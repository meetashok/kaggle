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

def loss_criterion(start_logits, end_logits, start, end):
    loss_fn = nn.CrossEntropyLoss()

    start_loss = loss_fn(start_logits, start)
    end_loss = loss_fn(end_logits, end)
    loss = torch.add(start_loss, end_loss)
    return loss

def batch_jaccard_similarity(ids, token_start_pred, token_end_pred, tweet, selected_text, sentiment, tokenizer, th=3):
    similarities = 0
    count_incorrect = 0

    for i in range(len(sentiment)):
        if (sentiment[i] == "neutral") or len(tweet[i].split()) <= th:
            pred_selected_text = tweet[i]
        else:
            pred_selected_text = tokenizer.decode(ids[i, token_start_pred[i]:token_end_pred[i]+1]).strip()

        if len(pred_selected_text) > len(tweet[i]):
            pred_selected_text = tweet[i]
            count_incorrect += 1

        if token_end_pred[i] < token_start_pred[i]:
            pred_selected_text = tweet[i]
            count_incorrect +=1 

        similarity = jaccard_similarity(pred_selected_text, selected_text[i])
        similarities += similarity
    
    return similarities / len(sentiment), len(sentiment), count_incorrect

def get_selected_text(ids, token_start, token_end, tokenizer):
    selected_ids = ids[token_start: token_end+1]
    selected_text = tokenizer.decode(selected_ids).strip()
    return selected_text

def train_model(model, model_params):
    model.train()

    tokenizer = model_params.get("tokenizer")
    dataloaders = model_params.get("dataloaders")
    modelsdir = model_params.get("modelsdir")
    loss_criterion = model_params.get("loss_criterion")
    optimizer = model_params.get("optimizer")
    scheduler = model_params.get("scheduler")
    writer = model_params.get("writer")
    num_epochs = model_params.get("num_epochs")
    device = model_params.get("device")

    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        running_loss = 0.0

        for i, data in enumerate(dataloaders["train"]):
            # print(model.linear.weight)       
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

            with torch.set_grad_enabled(True):
                start_logits, end_logits = model(ids, attention_mask, token_type_ids)
                loss = loss_criterion(start_logits, end_logits, token_start, token_end)

                token_start_pred = (torch.argmax(start_logits, dim=-1)
                            .cpu()
                            .detach()
                            .numpy())

                token_end_pred = (torch.argmax(end_logits, dim=-1)
                            .cpu()
                            .detach()
                            .numpy())

                ids = ids.cpu().detach().numpy()

                batch_jaccard, batch_count, batch_incorrect = batch_jaccard_similarity(ids, 
                                            token_start_pred, 
                                            token_end_pred, 
                                            tweet, 
                                            selected_text, 
                                            sentiment, 
                                            tokenizer)

                print(f"Loss = {loss.item():7.4f}, Jaccard = {batch_jaccard:6.4f}, Batch count: {batch_count:4,}, Batch incorrect: {batch_incorrect:4,}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # writer.add_scalar("loss/train", loss, global_step=global_step)

            running_loss += loss.item()  * attention_mask.size(0)

        eval_model(model, model_params)
        scheduler.step()
        epoch_loss = running_loss / len(dataloaders["train"])
        print('Train Loss: {:.4f}'.format(epoch_loss))


def eval_model(model, model_params):
    tokenizer = model_params.get("tokenizer")
    dataloaders = model_params.get("dataloaders")
    loss_criterion = model_params.get("loss_criterion")
    writer = model_params.get("writer")
    num_epochs = model_params.get("num_epochs")
    device = model_params.get("device")


    datatype = "valid" if "valid" in dataloaders else "train"
    print(f"Evaluating model on {datatype} data")

    running_loss = 0.0
    jaccard_score = 0.0
    total_count = 0
    count_incorrect = 0

    for _, data in enumerate(dataloaders[datatype]):
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

            batch_jaccard, total_count_batch, count_incorrect_batch  = batch_jaccard_similarity(ids, 
                                        token_start_pred, 
                                        token_end_pred, 
                                        tweet, 
                                        selected_text, 
                                        sentiment, 
                                        tokenizer)
            
            jaccard_score += batch_jaccard * total_count_batch
            total_count += total_count_batch
            count_incorrect += count_incorrect_batch

    loss_avg = running_loss / total_count
    jaccard_avg = jaccard_score / total_count

    print(f"Loss = {loss_avg:7.4f}, Jaccard = {jaccard_avg:6.4f}, Batch count: {total_count:4,}, Batch incorrect: {count_incorrect:4,}")