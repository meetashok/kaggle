from transformers import RobertaConfig
import torch
import datetime
import os

class Config:
    datadir = "../data"
    
    roberta_vocab = "../pretrained/roberta-base-vocab.json"
    roberta_merges = "../pretrained/roberta-base-merges.txt"
    roberta_config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
    
    max_len = 96
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device = {device}")

    batch_size = 32
    lr = 3e-5
    num_epochs = 1
    verbose = False

    frac = 0.001

    fold = 1

    # saving models
    modelsdir = "../models"
    suffix = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M")
    
    if not os.path.exists(os.path.join(modelsdir, suffix)):
        os.mkdir(os.path.join(modelsdir, suffix))
    