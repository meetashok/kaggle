from config import Config
from model import TweetModel
import torch 
import os

def infer():
    model = TweetModel(Config.roberta_config)
    modelpath = os.path.join("../models/2020.05.09.10.10/fold_1.bin")

    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint["model_state_dict"])

if __name__ == "__main__":
    infer()