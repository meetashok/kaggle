from engine import train_model, loss_criterion, eval_model
from model import TweetModel, TweetModel2
from model import initialize_tokenizer
from config import Config
from utils import read_data
from dataset import TweetData
from transformers import get_linear_schedule_with_warmup

from torch.utils import tensorboard
from transformers import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import logging

def run(fold, modelclass):
    # loading and setting up model, optimizer
    model = modelclass(Config.roberta_config)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=Config.lr)
    
    model = model.to(Config.device)  
    
    # preparing data 
    data, _, _ = read_data(Config.frac)

    train = data.query("fold != @fold").reset_index(drop=True)
    valid = data.query("fold == @fold").reset_index(drop=True)

    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    
    train_dataset = TweetData(train, tokenizer, Config.max_len)
    valid_dataset = TweetData(valid, tokenizer, Config.max_len)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False),
        "valid": DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False)
    }

    writer = tensorboard.SummaryWriter(log_dir=f"runs/roberta_{Config.suffix}_fold={fold}")

    num_train_steps = int(len(train) / Config.batch_size * Config.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=num_train_steps
    )

    model_params = {
        "tokenizer": tokenizer,
        "loss_criterion": loss_criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "writer": writer,
        "device": Config.device,
        "verbose": Config.verbose
    }

    valid_jaccard_best = 0
    patience = 0

    for epoch in range(Config.num_epochs):
        print("-"*20)
        print('Epoch {}/{}'.format(epoch+1, Config.num_epochs))
        train_model(model, dataloaders["train"], model_params)
        
        print("Evaluating training data")
        train_loss, train_jaccard = eval_model(model, dataloaders["train"], model_params)

        print("Evaluating validation data")
        valid_loss, valid_jaccard = eval_model(model, dataloaders["valid"], model_params)

        writer.add_scalars(f"fold={fold}/loss",    {"train": train_loss, "valid": valid_loss}, global_step=epoch+1)
        writer.add_scalars(f"fold={fold}/jaccard", {"train": train_jaccard, "valid": valid_jaccard}, global_step=epoch+1)

        if valid_jaccard > valid_jaccard_best:
            valid_jaccard_best = valid_jaccard
            patience = 0

            print("Saving model...")
            modeloutput = os.path.join(Config.modelsdir, Config.suffix, f"fold{fold}.bin")
            torch.save({"model_state_dict": model.state_dict()}, modeloutput)
        else:
            print("Jaccard didn't improve...")
            patience += 1
            if patience == 3:
                print(f"Jaccard didn't improve for 2 iterations, breaking after {epoch+1} epochs")
                break

    return valid_loss, valid_jaccard_best

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if os.path.exists(os.path.join(Config.modelsdir, Config.suffix)):
        os.removedirs(os.path.join(Config.modelsdir, Config.suffix))
    
    os.mkdir(os.path.join(Config.modelsdir, Config.suffix))

    cv_loss, cv_jaccard = [], []
    
    for fold in range(1, Config.nfolds+1):
        print("-"*20)
        print(f"Running fold = {fold}/{Config.nfolds}")
        print("-"*20)

        loss, jaccard = run(fold, modelclass=TweetModel)
        cv_loss += [loss]
        cv_jaccard += [jaccard]

    print(f"CV loss = {np.mean(cv_loss):6.4f}, CV Jaccard = {np.mean(cv_jaccard):6.4f}")