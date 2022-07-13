import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
from transformers import AdamW
LOGGER = logging.getLogger(__name__)


class RerankNet(nn.Module):
    def __init__(self, encoder, device, max_length, topk, batch_size, loss_fn):
        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.device = device
        self.max_length = max_length
        self.topk = topk
        self.batch_size = batch_size
        
        self.optimizer = AdamW(encoder.parameters(), lr=1e-5)
        
        #TODO: pass loss_fn?
        self.criterion = None #binary_cross_entropy if loss_fn == "bce" else marginal_nll
        
    def forward(self, x):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        # Split input into mentions and candidates
        mention_tokens, candidate_tokens = x
        
        # Embed mentions
        mention_tokens = mention_tokens.to(self.device)
        mention_embeds = self.encoder(
                    input_ids=mention_tokens['input_ids'].squeeze(1),
                    token_type_ids=mention_tokens['token_type_ids'].squeeze(1),
                    attention_mask=mention_tokens['attention_mask'].squeeze(1)
        )
        mention_embeds = mention_embeds[0][:,0].unsqueeze(1) # [CLS] embedding for mentions : [batch_size, 1, hidden]

        # Embed candidate names
        candidate_tokens = candidate_tokens.to(self.device)
        candidate_embeds = self.encoder(
                    input_ids=candidate_tokens['input_ids'].reshape(-1, self.max_length),
                    token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, self.max_length),
                    attention_mask=candidate_tokens['attention_mask'].reshape(-1, self.max_length)
        )
        candidate_embeds = candidate_embeds[0][:,0].reshape(self.batch_size, self.topk, -1) # [batch_size, topk, hidden]

        # Matrix multiply embeddings to score candidates
        return torch.bmm(mention_embeds, candidate_embeds.permute(0,2,1)).squeeze(1)

    def get_loss(self, outputs, targets):
        if self.device != 'cpu':
            targets = targets.to(self.device)
        loss = self.criterion(self, outputs, targets)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


# def marginal_nll(self, score, target):
#     """
#     sum all scores among positive samples
#     """
#     predict = F.softmax(score, dim=-1)
#     loss = predict * target
#     loss = loss.sum(dim=-1)                   # sum all positive scores
#     loss = loss[loss > 0]                     # filter sets with at least one positives
#     loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
#     loss = -torch.log(loss)                   # for negative log likelihood
#     if len(loss) == 0:
#         loss = loss.sum()                     # will return zero loss
#     else:
#         loss = loss.mean()
#     return loss

# def binary_cross_entropy(self, score, similarity):
#     """
#     Binary cross entropy loss. 
#     Ignores candidate scores and focuses on getting the similarity of all candidates as close to 1 as possible.
#     """
#     similarity = similarity.requires_grad_()
#     targets = torch.ones(similarity.shape).to(self.device)
#     return F.binary_cross_entropy(similarity, targets)