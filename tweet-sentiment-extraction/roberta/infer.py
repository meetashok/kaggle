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
from tqdm import tqdm
import numpy as np 

def infer_model(modelrun, modelclass, inferdata):
    device = Config.device
    modelpath = os.path.join(Config.modelsdir, modelrun)
    models = [modelname for modelname in os.listdir(modelpath) if modelname[-4:] == ".bin"]

    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    dataset = TweetData(inferdata, tokenizer, Config.max_len)

    start_tokens_all = torch.zeros((len(models), len(inferdata), Config.max_len))
    end_tokens_all = torch.zeros((len(models), len(inferdata), Config.max_len))
    ids_all = torch.zeros((len(inferdata), Config.max_len), dtype=torch.long)
    textIDs = []
    tweets = []

    for m, modelname in enumerate(models):
        print(f"Running {modelname}...")
        model = modelclass(Config.roberta_config)

        checkpoint = torch.load(os.path.join(modelpath, modelname), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

        start_tokens = torch.zeros((len(inferdata), Config.max_len), dtype=torch.float32)
        end_tokens = torch.zeros_like(start_tokens)

        for i, data in tqdm(enumerate(dataloader)):
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

                start_index = i*Config.batch_size
                end_index = (i+1)*Config.batch_size
                
                start_tokens[start_index:end_index, :] = start_logits
                end_tokens[start_index:end_index, :] = end_logits
            
            if m == 0:
                ids_all[start_index:end_index, :] = ids
                textIDs.extend(textID)
                tweets.extend(tweet)
        
        start_tokens_all[m,:,:] = start_tokens
        end_tokens_all[m,:,:] = end_tokens

    # getting mean of all five folds
    token_start_mean = torch.mean(start_tokens_all, dim=0)
    token_end_mean = torch.mean(end_tokens_all, dim=0)

    # doing a softmax on the folds
    token_start_softmax = torch.softmax(token_start_mean, dim=1)
    token_end_softmax = torch.softmax(token_end_mean, dim=1)

    # getting the max values and index fo max_value
    token_start_values, token_start_pred = torch.max(token_start_mean, dim=1)
    token_end_values, token_end_pred = torch.max(token_end_mean, dim=1)

    # bringing data to cpu
    token_start_values = token_start_values.cpu().detach().numpy()
    token_start_pred = token_start_pred.cpu().detach().numpy()
    token_end_values = token_end_values.cpu().detach().numpy()
    token_end_pred = token_end_pred.cpu().detach().numpy()
    
    ids_all = ids_all.cpu().detach().numpy()

    dataframe = {"textID": [], "selected_text": [], "start_token": [], "start_softmax": [], "end_token": [], "end_softmax": []}
    for i in range(len(inferdata)):
        selected_text = get_selected_text(ids_all[i], tweets[i], token_start_pred[i], token_end_pred[i], tokenizer)
        dataframe["textID"] += [textIDs[i]]
        dataframe["selected_text"] += [selected_text]
        dataframe["start_token"] += [token_start_pred[i]]
        dataframe["start_softmax"] += [token_start_values[i]]
        dataframe["end_token"] += [token_end_pred[i]]
        dataframe["end_softmax"] += [token_end_values[i]]


    print("Extracting submission...")
    pd.DataFrame(dataframe).to_csv(f"submission_{Config.suffix}.csv", index=False, quoting=csv.QUOTE_ALL)
    
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train, test, _ = read_data()
    test["selected_text"] = test.text

    infer_model(modelrun="roberta-simple", modelclass=TweetModel, inferdata=train)
