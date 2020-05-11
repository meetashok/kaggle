import transformers 
from tokenizers import ByteLevelBPETokenizer
import torch.nn as nn
import os
from config import Config
import torch
import torch.nn as nn

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

class TweetModel(nn.Module):
    def __init__(self, config):
        super(TweetModel, self).__init__()
        print("Importing model...")
        self.roberta = transformers.RobertaModel.from_pretrained("roberta-base", config=config)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(config.hidden_size*2, 2)
        nn.init.normal_(self.linear.weight, std=0.02)


    def forward(self, ids, attention_mask, token_type_ids):
        _, _, out = self.roberta(
                        ids, 
                        attention_mask=attention_mask,
                        )
        
        x = torch.cat((out[-1], out[-2]), dim=-1)
        x = self.dropout(x)
        logits = self.linear(x)

        start, end = logits[:,:,0], logits[:,:,1]

        return start, end

class TweetModel2(nn.Module):
    def __init__(self, config):
        super(TweetModel2, self).__init__()
        print("Importing model...")
        self.roberta = transformers.RobertaModel.from_pretrained("roberta-base", config=config)
        self.dropout = nn.Dropout(0.4)
        self.conv1 = nn.Conv1d(config.hidden_size, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.linear = nn.Linear(64, 1)

    def forward(self, ids, attention_mask, token_type_ids):
        _, _, out = self.roberta(
                        ids, 
                        attention_mask=attention_mask,
                        )
        
        out1, out2 = out[-1], out[-2]
        out1, out2 = out1.transpose(1, 2), out2.transpose(1, 2)


        out1 = self.dropout(out1)
        
        out1 = self.conv1(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.dropout(out1)

        out1 = self.conv2(out1)
        out1 = nn.ReLU()(out1)
        
        out1 = out1.transpose(1, 2)
        out1 = self.linear(out1)

        out2 = self.dropout(out2)
        
        out2 = self.conv1(out2)
        out2 = nn.ReLU()(out2)
        out2 = self.dropout(out2)
        
        out2 = self.conv2(out2)
        out2 = nn.ReLU()(out2)
        
        out2 = out2.transpose(1, 2)
        out2 = self.linear(out2)
        
        start, end = out1.squeeze(), out2.squeeze()

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
