import argparse
import json
import os
import torch
from transformers import (
    AutoModel,
    AutoTokenizer
)

# Local modules
from src.candidateDataset import CandidateDataset
from src.rerankNet import RerankNet
from src.umls import Umls
import src.utils as utils

def parse_args():
    "Parse input arguments"
    parser = argparse.ArgumentParser(description='Biomedical Entity Linker')
    parser.add_argument('--candidates',  type=int, default=5)
    parser.add_argument('--contextualized', action="store_true", help="use contextualized embeddings in candidate selection")
    parser.add_argument('--data_dir', type=str, required=True, help='data set to evaluate')
    parser.add_argument('--dictionary_path', type=str, required=True, help='dictionary path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_length', default=25, type=int)
    parser.add_argument('--model_name_or_path', required=True, help='Directory for model')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--umls_path', type=str, help='directory containing children.pickle and parents.pickle')
    args = parser.parse_args()
    return args

def main(args):
    # Initialize
    LOGGER = utils.init_logging()
    LOGGER.info(args)
    bert = AutoModel.from_pretrained(args.model_name_or_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load UMLS data
    umls = Umls('umls/processed')

    # Load dictionary
    dictionary = utils.load_dictionary(args.dictionary_path)
    LOGGER.info("Dictionary loaded")
    
    # Load mention data
    mentions = utils.load_mentions(args.data_dir)
    LOGGER.info("Mentions loaded")
    
    # Predict topk candidates
    candidate_idxs = utils.get_topk_candidates(
            dict_names=list(dictionary[:,0]), 
            mention_names=list(mentions[:,0]), 
            tokenizer=tokenizer, 
            encoder=bert, 
            max_length=args.max_length, 
            device=args.device, 
            topk=args.candidates)
    
    # Log performance on dev after each epoch
    results = utils.evaluate(mentions, dictionary[candidate_idxs], umls)
    if 'acc1' in results: LOGGER.info("Result: acc@1={}".format(results['acc1']))
    if 'acc5' in results: LOGGER.info("Result: acc@5={}".format(results['acc5']))
    if 'umls_similarity' in results: LOGGER.info("Result: umls_similarity={}".format(results['umls_similarity']))

    # Write output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file = os.path.join(args.output_dir,"predictions_eval.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)