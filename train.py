#region imports
import argparse
import os
import pandas as pd
import time
import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer
)

# Local modules
from src.candidateDataset import CandidateDataset
from src.rerankNet import RerankNet
from src.umls import Umls
import src.utils as utils
#endregion

#region parse_args()
def parse_args():
    "Parse input arguments"
    parser = argparse.ArgumentParser(description='Biomedical Entity Linker')
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--candidates',  type=int, default=5)
    parser.add_argument('--contextualized', type=int, default=0, help="use contextualized embeddings in candidate selection")
    parser.add_argument('--dev_dir', type=str, required=True, help='path to dev dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--epochs',  type=int, default=10)
    parser.add_argument('--loss_fn', type=str, required=True, choices=['ce','sce'])
    parser.add_argument('--lr',  type=float, default=1e-5)
    parser.add_argument('--max_length', default=25, type=int)
    parser.add_argument('--model_name_or_path', required=True, help='Directory for model')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--similarity_type', type=str, help='Similarity type for loss calculation')
    parser.add_argument('--test_dir', type=str, required=True, help='path to test dataset')
    parser.add_argument('--train_dir', type=str, required=True, help='path to training dataset')
    parser.add_argument('--umls_path', type=str, help='directory containing children.pickle and parents.pickle')
    args = parser.parse_args()
    return args
#endregion

def main(args):
    # Initialize
    start = time.time()
    LOGGER = utils.init_logging()
    LOGGER.info(args)
    utils.init_seed(42)
    bert = AutoModel.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Set loss function
    if args.loss_fn=='ce':
        loss_fn = utils.cross_entropy
    elif args.loss_fn=='sce':
        loss_fn = utils.similarity_cross_entropy
    else:
        raise Exception(f"Invalid loss function {args.loss_fn}")
        
    # Build model
    model = RerankNet(encoder=bert, 
                      tokenizer=tokenizer,
                      lr=args.lr,
                      device=args.device)
    
    # Load UMLS data
    umls = Umls('umls/processed')
    LOGGER.info("UMLS data loaded")

    # Load dictionary
    dictionary = utils.load_dictionary(args.dictionary_path)
    LOGGER.info("Dictionary loaded")

    # Load training data
    train_mentions = utils.load_mentions(args.train_dir)
    """
    Drop training records where the exact mention is mapped to a single CUI that is not the gold CUI
    Since encodings are non-contextualized, the network will always predict
    the highest similarity between exact mention matches. These training
    examples will only confuse the training.
    """
    name_cuis = utils.load_name_cuis(dictionary)
    consistent_mask = [utils.check_consistent(name_cuis, name,cui) for name,cui in train_mentions[:,:2]]
    LOGGER.info(f"Dropping {len(train_mentions)-sum(consistent_mask)} out of {len(train_mentions)} training records because of inconsistent exact mappings between annotation and dictionary CUIs")
    train_mentions = train_mentions[consistent_mask]
    train_set = CandidateDataset(train_mentions, dictionary, model.tokenizer, args.max_length, args.candidates, args.similarity_type, umls) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Load dev data for validation
    dev_mentions = utils.load_mentions(args.dev_dir)
    LOGGER.info("Mentions loaded")
    
    # Training loop
    epoch_results = pd.DataFrame([], columns=['acc@1','acc@5','umls_similarity', 'max_acc@1'])
    for epoch in range(args.epochs):
        ############## Candidate Generation ##############
        train_candidate_idxs = utils.get_topk_candidates(
                dict_names=list(dictionary[:,0]), 
                mentions=train_mentions, 
                tokenizer=model.tokenizer, 
                encoder=model.encoder, 
                max_length=args.max_length, 
                device=args.device, 
                topk=args.candidates)
                            
        # Add candidates to training dataset
        train_set.set_candidate_idxs(train_candidate_idxs)
        max_acc1 = train_set.max_acc1()
        LOGGER.info('Epoch {}: max possible acc@1 = {}'.format(epoch,max_acc1))
        
        ###################### Train ######################
        # Train encoder to properly rank candidates
        train_loss = 0
        train_steps = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training epoch {epoch}'):
            model.optimizer.zero_grad()
            batch_x, batch_y = data
            batch_pred = model(batch_x)
            loss = loss_fn(batch_pred, batch_y.to(args.device))
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        train_loss = train_loss / (train_steps + 1e-9)
        LOGGER.info('Epoch {}: loss/train_per_epoch={}/{}'.format(epoch,train_loss,epoch))
        
        #################### Evaluate ####################
        # Get candidates on dev dataset
        dev_candidate_idxs = utils.get_topk_candidates(
                dict_names=list(dictionary[:,0]), 
                mentions=dev_mentions, 
                tokenizer=model.tokenizer, 
                encoder=model.encoder, 
                max_length=args.max_length, 
                device=args.device, 
                topk=5) # Only need top five candidates to evaluate performance
        
        # Log performance on dev after each epoch
        results = utils.evaluate(dev_mentions, dictionary[dev_candidate_idxs], umls)
        epoch_results.loc[epoch] = (results['acc1'], results['acc5'], results['umls_similarity'], max_acc1)
        LOGGER.info("Epoch {}: acc@1={}".format(epoch,results['acc1']))
        LOGGER.info("Epoch {}: acc@5={}".format(epoch,results['acc5']))
        LOGGER.info("Epoch {}: umls_similarity={}".format(epoch,results['umls_similarity']))

        # Dump model after each training epoch
        model.dump(args.output_dir, epoch)
    
    # Training loop complete
    LOGGER.info('Training time: ' + utils.format_time(start,time.time()))
    LOGGER.info(f'\n{epoch_results.round(3)}')
    
    
    
    ###################### Predict ######################
    # Evaluate on test data using best training model
    best_epoch = epoch_results.umls_similarity.argmax()
    train_model_path = os.path.join(args.output_dir, "checkpoint_{}".format(best_epoch))
    LOGGER.info(f'Loading epoch {best_epoch} model from {train_model_path}')
    
    # Evaluate on test data using best training model
    start = time.time()
    best_epoch = epoch_results.umls_similarity.argmax()
    train_model_path = os.path.join(args.output_dir, "checkpoint_{}".format(best_epoch))
    LOGGER.info(f'Loading epoch {best_epoch} model from {train_model_path}')

    # Load training model
    train_bert = AutoModel.from_pretrained(train_model_path).to(args.device)
    train_tokenizer = AutoTokenizer.from_pretrained(train_model_path)

    # Load test mentions
    test_mentions = utils.load_mentions(args.test_dir)

    # Predict topk=5 candidates
    candidate_idxs = utils.get_topk_candidates(
            dict_names=list(dictionary[:,0]), 
            mentions=test_mentions, 
            tokenizer=train_tokenizer, 
            encoder=train_bert, 
            max_length=args.max_length, 
            device=args.device, 
            topk=5, # Only need top five candidates to evaluate performance
            doc_dir=None) # Update to allow contextualized embeddings

    # Log performance
    results = utils.evaluate(test_mentions, dictionary[candidate_idxs], umls)
    LOGGER.info("Test result: acc@1={}".format(results['acc1']))
    LOGGER.info("Test result: acc@5={}".format(results['acc5']))
    LOGGER.info("Test result: umls_similarity={}".format(results['umls_similarity']))

    LOGGER.info('Prediction time: ' + utils.format_time(start,time.time()))

    # Write output
    utils.dump_predictions(args.output_dir, results)

if __name__ == '__main__':
    args = parse_args()
    main(args)