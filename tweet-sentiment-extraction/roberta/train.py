from engine import train_model, loss_criterion, eval_model
from model import TweetModel, initialize_tokenizer
from config import Config
from utils import read_data
from dataset import TweetData
from transformers import get_linear_schedule_with_warmup

from torch.utils import tensorboard
from transformers import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

if __name__ == "__main__":
    model = TweetModel(Config.roberta_config)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=Config.lr)
    
    model = model.to(Config.device)  
    
    data, test, _ = read_data(Config.frac)

    train = data.query("fold != @Config.fold").reset_index(drop=True)
    valid = data.query("fold == @Config.fold").reset_index(drop=True)

    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    
    train_dataset = TweetData(train, tokenizer, Config.max_len)
    valid_dataset = TweetData(valid, tokenizer, Config.max_len)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False),
        "valid": DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False)
    }

    writer = tensorboard.SummaryWriter(log_dir=f"../runs/roberta_{Config.suffix}")

    num_train_steps = int(len(train) / Config.batch_size * Config.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    model_params = {
        "tokenizer": tokenizer,
        "modelsdir": Config.modelsdir,
        "loss_criterion": loss_criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "writer": writer,
        "device": Config.device,
        "verbose": Config.verbose
    }

    metrics = {
        "train": { "loss": [], "jaccard": [] },
        "valid": { "loss": [], "jaccard": [] },
    }

    for epoch in range(Config.num_epochs):
        print('Epoch {}/{}'.format(epoch+1, Config.num_epochs))
        train_model(model, dataloaders["train"], model_params)
        
        print("Evaluating training data")
        loss, jaccard = eval_model(model, dataloaders["train"], model_params)
        metrics["train"]["loss"] += [loss]
        metrics["train"]["jaccard"] += [jaccard]

        print("Evaluating validation data")
        loss, jaccard = eval_model(model, dataloaders["valid"], model_params)
        metrics["valid"]["loss"] += [loss]
        metrics["valid"]["jaccard"] += [jaccard]