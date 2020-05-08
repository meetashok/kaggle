from transformers import RobertaConfig
import torch

class Config:
    datadir = "../data"
    modelsdir = "../models"
    roberta_vocab = "../pretrained/roberta-base-vocab.json"
    roberta_merges = "../pretrained/roberta-base-merges.txt"
    roberta_config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
    # roberta_config.output_hidden_states = True
    max_len = 96
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device = {device}")

    lr = 0.01
