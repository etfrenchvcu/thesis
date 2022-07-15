import math
import numpy as np
import torch

class CandidateDataset(torch.utils.data.Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, mentions, dictionary, tokenizer, max_length, topk, similarity_type, umls):
        self.query_names = mentions[:,0]
        self.gold_cuis = mentions[:,1]

        self.dict_names = dictionary[:,0]
        self.dict_cuis = dictionary[:,1]

        self.topk = topk
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.umls = umls
        self.similarity_type = similarity_type

        self.candidate_idxs = None

    def set_candidate_idxs(self, candidate_idxs):
        self.candidate_idxs = candidate_idxs
    
    def __getitem__(self, query_idx):
        assert (self.candidate_idxs is not None)

        # Tokenize mention
        mention = self.query_names[query_idx]
        mention_tokens = self.tokenizer(mention, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        # Tokenize Candidates
        candidate_names = [self.dict_names[candidate_idx] for candidate_idx in self.candidate_idxs[query_idx]]
        candidate_tokens = self.tokenizer(candidate_names, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        # Get candidate labels
        gold_cui = self.gold_cuis[query_idx]
        candidate_cuis = self.dict_cuis[self.candidate_idxs[query_idx]]
        if self.similarity_type=='binary':
            # Binary labels [0,1] for exact CUI matches
            labels = (gold_cui==candidate_cuis).astype(np.float32)
        elif self.similarity_type=='linear':
            # Float labels (0,1] representing 1/1+dist, where dist is the UMLS distance between gold and candidate CUIs
            labels = np.array([self.umls.similarity(gold_cui, cui) for cui in candidate_cuis],np.float32)
        elif self.similarity_type=='log':
            # Float labels (0,1] representing 1/e^dist, where dist is the UMLS distance between gold and candidate CUIs
            labels = np.array([self.log_similarity(gold_cui, cui) for cui in candidate_cuis],np.float32)
        else:
            raise Exception(f"Invalid similarity_type={self.similarity_type}")
        
        return (mention_tokens, candidate_tokens), labels

    def __len__(self):
        return len(self.query_names)
    
    def log_similarity(self, gold_cui, candidate_cuis):
        dist = self.umls.dist(gold_cui, candidate_cuis)
        return 1 / math.exp(dist)