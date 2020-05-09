from transformers import RobertaConfig
import torch
import datetime

class Config:
    datadir = "../data"
    modelsdir = "../models"
    roberta_vocab = "../pretrained/roberta-base-vocab.json"
    roberta_merges = "../pretrained/roberta-base-merges.txt"
    roberta_config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
    
    max_len = 96
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device = {device}")

    batch_size = 32
    lr = 3e-5
    num_epochs = 10
    verbose = False

    frac = 0.1

    fold = 1

    # helpers
    suffix = datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S")