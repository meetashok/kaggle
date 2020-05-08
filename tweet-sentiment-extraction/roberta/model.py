import transformers 
from tokenizers import ByteLevelBPETokenizer
import torch.nn as nn
import os
from config import Config
import torch

def initialize_tokenizer(vocab_file, merges_file):
    print("Tokenizer getting loaded...")

    tokenizer = ByteLevelBPETokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        lowercase=True,
        add_prefix_space=True)

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size = {vocab_size:,}")

    return tokenizer

class TweetModel(transformers.BertModel):
    def __init__(self, config):
        super(TweetModel, self).__init__(config)
        print("Importing model...")
        self.roberta = transformers.RobertaModel.from_pretrained("roberta-base", config=config)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(config.hidden_size*2, 2)

    def forward(self, ids, attention_mask, token_type_ids):
        _, _, out = self.roberta(
                        ids, 
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        
        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.dropout(out)
        logits = self.linear(out)

        start, end = logits[:,:,0], logits[:,:,1]

        return start, end

if __name__ == "__main__":
    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    
    tweet = "Feeling good today"
    encoded = tokenizer.encode(tweet)

    ids = torch.tensor([encoded.ids], dtype=torch.long)
    attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([encoded.type_ids], dtype=torch.long)

    model = TweetModel(Config.roberta_config)
    model = model.to(Config.device)

    start, end = model(ids, attention_mask, token_type_ids)

    print(start)
    print(end)
