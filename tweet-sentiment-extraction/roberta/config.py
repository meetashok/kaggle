from transformers import RobertaConfig
import torch
import datetime
import os

class Config:
    datadir = "../data"
    
    roberta_vocab = "../pretrained/roberta-base-vocab.json"
    roberta_merges = "../pretrained/roberta-base-merges.txt"
    roberta_config_file = "../pretrained/roberta-base-config.json"
    roberta_config = RobertaConfig.from_pretrained("roberta-large")
    roberta_config.output_hidden_states = True
    
    max_len = 96
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device = {device}")

    batch_size = 4
    lr = 5e-5
    num_epochs = 3
    verbose = False
    nfolds = 5

    frac = 1

    # saving models
    modelsdir = "../models"
    suffix = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")
    