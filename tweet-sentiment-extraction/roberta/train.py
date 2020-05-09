from engine import train_model, loss_criterion
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
    
    model = model.to(Config.device)  

    train, test, _ = read_data(Config.frac)
    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    train_dataset = TweetData(train, tokenizer, Config.max_len)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False)
    }

    # optimizer = optim.Adam(model.parameters(), 
    #         lr=Config.lr, 
    #         betas=(0.9, 0.999), 
    #         eps=1e-08, 
    #         weight_decay=1e-5)

    writer = tensorboard.SummaryWriter(log_dir=f"../runs/roberta_{Config.suffix}")

    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    num_train_steps = int(len(train) / Config.batch_size * Config.num_epochs)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    model_params = {
        "tokenizer": tokenizer,
        "dataloaders": dataloaders,
        "modelsdir": Config.modelsdir,
        "loss_criterion": loss_criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "writer": writer,
        "num_epochs": Config.num_epochs,
        "device": Config.device,
    }

    train_model(model, model_params)