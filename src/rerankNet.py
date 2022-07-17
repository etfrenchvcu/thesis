import os
import torch
from tqdm import tqdm


class RerankNet(torch.nn.Module):
    def __init__(self, encoder, tokenizer, lr, device):
        super(RerankNet, self).__init__()
        self.encoder = encoder
        self.device = device
        self.tokenizer = tokenizer
        
        self.optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
        
    def forward(self, x):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        # Split input into mentions and candidates
        mention_tokens, candidate_tokens = x
        batch_size, candidates, max_length = candidate_tokens['input_ids'].shape
        
        # Embed mentions
        mention_embeds = self.encoder(
                    input_ids=mention_tokens['input_ids'].squeeze(1).to(self.device),
                    token_type_ids=mention_tokens['token_type_ids'].squeeze(1).to(self.device),
                    attention_mask=mention_tokens['attention_mask'].squeeze(1).to(self.device)
        )
        mention_embeds = mention_embeds[0][:,0].unsqueeze(1) # [CLS] embedding for mentions : [batch_size, 1, hidden]

        # Embed candidate names
        candidate_embeds = self.encoder(
                    input_ids=candidate_tokens['input_ids'].reshape(-1, max_length).to(self.device),
                    token_type_ids=candidate_tokens['token_type_ids'].reshape(-1, max_length).to(self.device),
                    attention_mask=candidate_tokens['attention_mask'].reshape(-1, max_length).to(self.device)
        )
        candidate_embeds = candidate_embeds[0][:,0].reshape(batch_size, candidates, -1) # [batch_size, candidates, hidden]

        # Matrix multiply embeddings to score candidates
        return torch.bmm(mention_embeds, candidate_embeds.permute(0,2,1)).squeeze(1)
    
    def dump(self, output_dir, epoch):
        checkpoint_dir = os.path.join(output_dir, "checkpoint_{}".format(epoch))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.encoder.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
    