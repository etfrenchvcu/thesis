import glob
import logging
import numpy as np
import os
import time
import torch
from tqdm import tqdm
from transformers import default_data_collator

# Local modules
from namesDataset import NamesDataset

def bulk_embed(names, tokenizer, encoder, max_length, device, show_progress=True):
    "Creates BERT embeddings for a bulk list of mention or dictionary names"
    # Tokenize and create dataset
    name_encodings = tokenizer(names, padding="longest", max_length=max_length, truncation=True, return_tensors="pt").to(device)
    name_dataset = NamesDataset(name_encodings)
    name_dataloader = torch.utils.data.DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=1024)

    # Create embeddings in batches
    embeds = []
    encoder.eval()
    for batch in tqdm(name_dataloader, disable=not show_progress, desc='Bulk embedding...'):
        outputs = encoder(**batch)
        batch_embeds = outputs[0][:,0].cpu().detach().numpy() # [CLS] representations
        embeds.append(batch_embeds)
    return np.concatenate(embeds, axis=0)

def get_topk_candidates(dict_names, mention_names, tokenizer, encoder, max_length, device, topk, show_progress=True):
    "Encodes dictionary and mention names, computes similarity, and returns topk candidates for each mention"
    # Create initial embeddings to identify candidates out of entire dictionary
    dict_embeds = bulk_embed(names=dict_names, tokenizer=tokenizer, encoder=encoder, max_length=max_length, device=device, show_progress=show_progress)
    query_embeds = bulk_embed(names=mention_names, tokenizer=tokenizer, encoder=encoder, max_length=max_length, device=device, show_progress=show_progress)

    # Matrix multiply and numpy magic to get top K similar candidates
    score_matrix = np.matmul(query_embeds, dict_embeds.T)
    candidate_idxs = retrieve_candidates(score_matrix, topk)
    return candidate_idxs

def retrieve_candidates(score_matrix, topk):
    """
    Given a matrix of similarity scores from multiplying dictionary and mention embeddings,
    ranks the similarity and returns the topk dictionary candidates. 
    """
    def indexing_2d(arr, cols):
        rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
        return arr[rows, cols]

    # get topk indexes without sorting
    topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

    # get topk indexes with sorting
    topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
    topk_argidxs = np.argsort(-topk_score_matrix) 
    topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

    return topk_idxs

#region Initialization/Data Loading
def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def init_logging():
    LOGGER = logging.getLogger()
    # Logger is already configured, remove all handlers
    if LOGGER.hasHandlers():
        LOGGER.handlers = []
        
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
    return LOGGER

def load_dictionary(dictionary_path):
    "Returns np.array([name, cui])"
    data = []
    with open(dictionary_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if line == "": continue
            cui, name = line.split("||")
            data.append((name,cui))
    
    return np.array(data)

def load_mentions(data_dir):
    "Returns np.array([mention, cui, offset, file])"
    data = []

    concept_files = sorted(glob.glob(os.path.join(data_dir, "*.concept")))
    for concept_file in tqdm(concept_files):
        with open(concept_file, "r", encoding='utf-8') as f:
            concepts = f.readlines()

        for concept in concepts:
            concept = concept.split("||")
            file = concept[0].strip()
            offset = concept[1].strip()
            mention = concept[3].strip()
            cui = concept[4].strip()
            data.append((mention,cui,offset,file))
    
    return np.array(data)
#endregion

#region Loss functions
def marginal_nll(score, target):
    "Marginal negative log likelihood loss"
    predict = torch.nn.functional.softmax(score, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)                   # sum all positive scores
    #TODO: this breaks on the mac, but maybe bring back
    # loss = loss[loss > 0]                     # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
    loss = -torch.log(loss)                   # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()                     # will return zero loss
    else:
        loss = loss.mean()
    return loss
#endregion

#region Evaluation functions
def log_topk(eval_mentions, candidates):
    "Creates a results dictionary containing the names, cuis, and label for all predicted candidates for each mention"
    dict_mentions = []
    for mention, candidates in zip(eval_mentions,candidates):
        mention_name = mention[0]
        golden_cui = mention[1]
        
        dict_candidates = []
        for candidate in candidates:
            dict_candidates.append({
                'name':candidate[0],
                'cui':candidate[1],
                'label':int(candidate[1]==golden_cui)
            })
        
        dict_mentions.append({
            'mention':mention_name,
            'golden_cui':golden_cui, # golden_cui can be composite cui
            'candidates':dict_candidates
        })
    return {'result':dict_mentions}

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    mentions = data['result']
    k = len(mentions[0]['candidates'])

    for i in range(0, k):
        hit = 0
        for mention in mentions:
            candidates = mention['candidates'][:i+1] # to get acc@(i+1)
            hit += int(np.any([candidate['label'] for candidate in candidates]))
        
        data['acc{}'.format(i+1)] = hit/len(mentions)

    return data

def evaluate_umls_similarity(data, umls):
    """
    Evaluate the average UMLS similarity between the golden_cui and the first prediction
    """
    mentions = data['result']
    gold_pred_similarity = [umls.similarity(m['golden_cui'],m['candidates'][0]['cui']) for m in mentions]
    data['umls_similarity'] = np.array(gold_pred_similarity).mean()
    return data

def evaluate(eval_mentions, candidates, umls=None):
    "Calculates performance metrics on the predicted candidates"
    results = log_topk(eval_mentions, candidates)
    results = evaluate_topk_acc(results)
    
    # Evaluate UMLS similarity if biosyn.umls is given
    if umls:
        results = evaluate_umls_similarity(results, umls)
        
    return results
#endregion