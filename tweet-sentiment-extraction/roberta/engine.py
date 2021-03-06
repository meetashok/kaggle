from utils import jaccard_similarity
from model import TweetModel
from config import Config
from transformers import RobertaConfig
from tokenizers import ByteLevelBPETokenizer

import torch.optim as optim
from torch.optim import lr_scheduler
from utils import read_data
from dataset import TweetData
from torch.utils.data import Dataset, DataLoader
import time
import copy
import torch
import torch.nn as nn
from model import initialize_tokenizer
from transformers import AdamW
from torch.utils import tensorboard
import numpy as np
import torch
from tqdm import tqdm

def loss_criterion(start_logits, end_logits, start, end):
    weight = 0.6
    loss_fn = nn.CrossEntropyLoss()
    ones = torch.ones_like(start).to(Config.device)

    start_loss = weight*loss_fn(start_logits, start) + (1-weight)/2*loss_fn(start_logits, torch.min(start+1, Config.max_len*ones)) + (1-weight)/2*loss_fn(start_logits, torch.max(start-1, 4*ones)) 
    end_loss = weight*loss_fn(end_logits, end) + (1-weight)/2*loss_fn(end_logits, torch.min(end+1, Config.max_len*ones)) + (1-weight)/2*loss_fn(end_logits, torch.max(end-1, 4*ones)) 

    loss = torch.add(start_loss, end_loss) 
    return loss

def batch_jaccard_similarity(ids, token_start_pred, token_end_pred, tweet, selected_text, sentiment, tokenizer, th=3):
    jaccards = {"positive": 0, "negative": 0, "neutral": 0}
    counts =  {"positive": 0, "negative": 0, "neutral": 0}

    for i in range(len(sentiment)):
        if (sentiment[i] == "neutral") or len(tweet[i].split()) <= th:
            # pred_selected_text = tweet[i]
            pred_selected_text = get_selected_text(ids[i], tweet[i], token_start_pred[i], token_end_pred[i], tokenizer)
        else:
            pred_selected_text = get_selected_text(ids[i], tweet[i], token_start_pred[i], token_end_pred[i], tokenizer)

        jaccard = jaccard_similarity(pred_selected_text, selected_text[i])
        
        # print(f"Pred selected text: {pred_selected_text}")
        # print(f"Selected text: {selected_text[i]}")
        # print(f"Jaccard score: {jaccard}")

        jaccards[sentiment[i]] += jaccard
        counts[sentiment[i]] += 1
    
    return jaccards, counts

def get_selected_text(ids, tweet, token_start, token_end, tokenizer):
    selected_ids = ids[token_start: token_end+1]
    pred_selected_text = tokenizer.decode(selected_ids).strip()

    if len(pred_selected_text) > len(tweet):
        pred_selected_text = tweet.strip()

    if token_start > token_end:
        pred_selected_text = tweet.strip()

    return pred_selected_text

def train_model(model, dataloader, model_params):
    model.train()

    tokenizer = model_params.get("tokenizer")
    modelsdir = model_params.get("modelsdir")
    loss_criterion = model_params.get("loss_criterion")
    optimizer = model_params.get("optimizer")
    scheduler = model_params.get("scheduler")
    writer = model_params.get("writer")
    device = model_params.get("device")
        
    running_loss = 0.0

    for i, data in tqdm(enumerate(dataloader)):
        # print(model.linear.weight)       
        ids = data.get("ids")
        attention_mask = data.get("attention_mask")
        token_type_ids = data.get("token_type_ids")
        token_start = data.get("token_start")
        token_end = data.get("token_end")
        tweet = data.get("tweet")
        selected_text = data.get("selected_text")
        sentiment = data.get("sentiment")
        verbose = data.get("verbose")
        
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        token_start = token_start.to(device)
        token_end = token_end.to(device)

        with torch.set_grad_enabled(True):
            start_logits, end_logits = model(ids, attention_mask, token_type_ids)
            loss = loss_criterion(start_logits, end_logits, token_start, token_end)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if verbose: 
                token_start_pred = (torch.argmax(start_logits, dim=-1)
                        .cpu()
                        .detach()
                        .numpy())

                token_end_pred = (torch.argmax(end_logits, dim=-1)
                        .cpu()
                        .detach()
                        .numpy())

                ids = ids.cpu().detach().numpy()

                batch_jaccard = batch_jaccard_similarity(ids, 
                                            token_start_pred, 
                                            token_end_pred, 
                                            tweet, 
                                            selected_text, 
                                            sentiment, 
                                            tokenizer)

                print(f"Loss = {loss.item():7.4f}, Jaccard = {batch_jaccard:6.4f}")


            # writer.add_scalar("loss/train", loss, global_step=global_step)

            running_loss += loss.item()  * attention_mask.size(0)

    epoch_loss = running_loss / len(dataloader)
    # print('Train Loss: {:.4f}'.format(epoch_loss))


def eval_model(model, dataloader, model_params):
    tokenizer = model_params.get("tokenizer")
    loss_criterion = model_params.get("loss_criterion")
    writer = model_params.get("writer")
    device = model_params.get("device")

    running_loss = 0.0
    counts = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    jaccards = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}

    for _, data in tqdm(enumerate(dataloader)):
        model.eval()
        
        ids = data.get("ids")
        attention_mask = data.get("attention_mask")
        token_type_ids = data.get("token_type_ids")
        token_start = data.get("token_start")
        token_end = data.get("token_end")
        tweet = data.get("tweet")
        selected_text = data.get("selected_text")
        sentiment = data.get("sentiment")
        
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        token_start = token_start.to(device)
        token_end = token_end.to(device)

        with torch.set_grad_enabled(False):
            start_logits, end_logits = model(ids, attention_mask, token_type_ids)
            loss = loss_criterion(start_logits, end_logits, token_start, token_end)
            running_loss += loss.item() * attention_mask.size(0)

            token_start_pred = (torch.argmax(start_logits, dim=-1)
                            .cpu()
                            .detach()
                            .numpy())

            token_end_pred = (torch.argmax(end_logits, dim=-1)
                            .cpu()
                            .detach()
                            .numpy())

            ids = ids.cpu().detach().numpy()

            batch_jaccard, batch_counts  = batch_jaccard_similarity(ids, 
                                        token_start_pred, 
                                        token_end_pred, 
                                        tweet, 
                                        selected_text, 
                                        sentiment, 
                                        tokenizer)
            
            for sent in ["positive", "negative", "neutral"]:
                jaccards[sent] += batch_jaccard[sent]
                counts[sent] += batch_counts[sent]

    total_jaccard = jaccards["positive"] + jaccards["negative"] + jaccards["neutral"]
    total_count = counts["positive"] + counts["negative"] + counts["neutral"]

    loss_avg = running_loss / total_count
    jaccard_avg = total_jaccard / total_count

    print(f"Loss = {loss_avg:7.4f}, Jaccard = {jaccard_avg:6.4f}, Positive: {jaccards['positive'] / counts['positive']:6.4f}, Negative: {jaccards['negative'] / counts['negative']:6.4f}, Neutral: {jaccards['neutral']/counts['neutral']:6.4f}")
    # print(f"Loss = {loss_avg:7.4f}, Jaccard = {jaccard_avg:6.4f}, Positive: {jaccards['positive'] / counts['positive']:6.4f}")

    return (loss_avg, jaccard_avg)