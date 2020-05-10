from config import Config
from model import TweetModel, TweetModel2
import torch 
import os
from utils import read_data
from dataset import TweetData
from model import initialize_tokenizer
from engine import get_selected_text
from torch.utils.data import DataLoader
import pandas as pd
from config import Config
import csv

def infer_model(modelrun):
    device = Config.device
    modelpath = os.path.join(Config.modelsdir, modelrun)
    models = [modelname for modelname in os.listdir(modelpath) if modelname[-4:] == ".bin"]

    _, test, _ = read_data()
    test["selected_text"] = test.text

    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    test_dataset = TweetData(test, tokenizer, Config.max_len)

    start_tokens_all = torch.zeros((len(models), len(test), Config.max_len))
    end_tokens_all = torch.zeros((len(models), len(test), Config.max_len))
    ids_all = torch.zeros((len(test), Config.max_len), dtype=torch.long)
    textIDs = []
    tweets = []

    for m, modelname in enumerate(models):
        print(f"Running {modelname}...")
        model = TweetModel2(Config.roberta_config)

        checkpoint = torch.load(os.path.join(modelpath, modelname))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

        start_tokens = torch.zeros((len(test), Config.max_len), dtype=torch.float32)
        end_tokens = torch.zeros_like(start_tokens)

        for i, data in enumerate(dataloader):
            model.eval()
            
            ids = data.get("ids")
            attention_mask = data.get("attention_mask")
            token_type_ids = data.get("token_type_ids")
            tweet = data.get("tweet")
            sentiment = data.get("sentiment")
            textID = data.get("textID")
            
            ids = ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            with torch.set_grad_enabled(False):
                start_logits, end_logits = model(ids, attention_mask, token_type_ids)
            
                start_tokens[i*Config.batch_size:(i+1)*Config.batch_size, :] = start_logits
                end_tokens[i*Config.batch_size:(i+1)*Config.batch_size, :] = end_logits
            
            if m == 0:
                ids_all[i*Config.batch_size:(i+1)*Config.batch_size, :] = ids
                textIDs.extend(textID)
                tweets.extend(tweet)
        
        start_tokens_all[m,:,:] = start_tokens
        end_tokens_all[m,:,:] = end_tokens

    token_start_pred = torch.argmax(torch.mean(start_tokens_all, dim=0), dim=-1).cpu().detach().numpy()
    token_end_pred = torch.argmax(torch.mean(end_tokens_all, dim=0), dim=-1).cpu().detach().numpy()
    ids_all = ids_all.cpu().detach().numpy()

    dataframe = {"textID": [], "selected_text": []}    
    for i in range(len(test)):
        selected_text = get_selected_text(ids_all[i], tweets[i], token_start_pred[i], token_end_pred[i], tokenizer)
        dataframe["textID"] += [textIDs[i]]
        dataframe["selected_text"] += [selected_text]

    print("Extracting submission...")
    pd.DataFrame(dataframe).to_csv("sample_submission.csv", index=False, quoting=csv.QUOTE_ALL)
    
if __name__ == "__main__":
    infer_model(modelrun="2020.05.09.16.54")
