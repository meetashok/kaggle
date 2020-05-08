from engine import train_model, loss_criterion
from model import TweetModel, initialize_tokenizer
from config import Config
from utils import read_data
from dataset import TweetData

from torch.utils import tensorboard
from transformers import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

if __name__ == "__main__":
    model = TweetModel(Config.roberta_config)
    model = model.to(Config.device)  

    train, test, _ = read_data(frac=1)
    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    train_dataset = TweetData(train, tokenizer, Config.max_len)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=Config.batch_size)
    }

    # optimizer = optim.Adam(model.parameters(), 
    #         lr=Config.lr, 
    #         betas=(0.9, 0.999), 
    #         eps=1e-08, 
    #         weight_decay=1e-5)

    writer = tensorboard.SummaryWriter(log_dir=f"../runs/roberta_{Config.suffix}")

    optimizer = AdamW(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_model(model, tokenizer, dataloaders, Config.modelsdir, loss_criterion, optimizer, scheduler, writer, num_epochs=Config.num_epochs)