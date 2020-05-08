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

def criterion(start_logits, end_logits, start, end):
    loss_fn=nn.CrossEntropyLoss()

    start_loss = loss_fn(start_logits, start)
    end_loss = loss_fn(end_logits, end)

    return (start_loss + end_loss) / 2

def batch_jaccard_similarity(ids, token_start, token_end, tweet, selected_text, sentiment, tokenizer, th=3):
    similarities = 0
    for i in range(len(sentiment)):
        if sentiment[i] == "neutral":
            pred_selected_text = tweet[i]
        elif len(tweet[i].split()) <= 3:
            pred_selected_text = tweet[i]
        else:
            pred_selected_text = tokenizer.decode(ids[i, token_start[i]:token_end[i]+1]).strip()

        if len(pred_selected_text) > len(tweet[i]):
            pred_selected_text = tweet[i]

        # print(tweet[i])
        # print(selected_text[i])
        # print(pred_selected_text)
        similarity = jaccard_similarity(pred_selected_text, selected_text[i])
        similarities += similarity
    
    return similarities / len(sentiment)

def get_selected_text(ids, token_start, token_end, tokenizer):
    selected_ids = ids[token_start: token_end+1]
    selected_text = tokenizer.decode(selected_ids).strip()
    return selected_text

def train_model(model, dataloaders, outdir, criterion, optimizer, scheduler, num_epochs, device=Config.device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 1e6

    # modelinfo = namedtuple("Model", ("time", "path", "auc"))
    # modelsinfo = [modelinfo(i, "output", 0.0) for i in range(maxmodels)]

    # if modelpath:
    #     checkpoint = torch.load(modelpath)
    #     model.load_state_dict(checkpoint["model_state_dict"])

    # class_names = dataloaders["train"].dataset.classes[P.classes_type]
    # n_classes = len(class_names)
    # global_step = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        running_loss = 0.0
        # all_probs, all_labels = [], []

        for i, data in enumerate(dataloaders["train"]):
            model.train()
            
            ids, attention_mask, token_type_ids = (
                                            data["ids"], 
                                            data["attention_mask"], 
                                            data["token_type_ids"]
                                            )

            token_start, token_end, tweet, selected_text, sentiment = (
                                    data["token_start"], 
                                    data["token_end"], 
                                    data["tweet"],
                                    data["selected_text"],
                                    data["sentiment"])
            
            ids = ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            token_start = token_start.to(device)
            token_end = token_end.to(device)
            
            # global_step += labels.size(0)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                start_logits, end_logits = model(ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)

                token_start_pred = torch.argmax(start_logits, dim=-1).detach().numpy()
                token_end_pred = torch.argmax(end_logits, dim=-1).detach().numpy()
                ids = ids.detach().numpy()

                batch_jaccard = batch_jaccard_similarity(ids, 
                                            token_start_pred, 
                                            token_end_pred, 
                                            tweet, 
                                            selected_text, 
                                            sentiment, 
                                            tokenizer)

                loss = criterion(start_logits, end_logits, token_start, token_end)
                print(f"Loss = {loss.item():10.4f}, Jaccard = {batch_jaccard:10.4f}")

                loss.backward()
                optimizer.step()

                # writer.add_scalar("loss/train", loss, global_step=global_step)

            running_loss += loss.item() * attention_mask.size(0)
            # running_corrects += torch.sum(preds == labels, 0)
            # if (i % P.printevery == 0) or (i==len(dataloaders["train"])-1):
            #     time_since = time.time() - since 
            #     print("Phase: Train, Epoch: {:2}, Iteration: {:2}/{}, Progress: {:.0f}%, Loss: {:.4f}, Time: {:.0f}m {:.0f}s".format( 
            #         epoch+1, i, len(dataloaders["train"]), 100. * (i) / len(dataloaders["train"]), loss.item(), time_since // 60, time_since % 60))
            #if i % P.evaluateevery == 0:
                # all_probs_valid, all_labels_valid, eval_loss = evaluate_singlemodel(model, criterion, dataloaders["valid"])
                # all_probs_valid = torch.cat(all_probs_valid, dim=0).detach().numpy()
                # all_labels_valid = torch.cat(all_labels_valid, dim=0).detach().numpy()
                # eval_metrics = compute_metrics(all_probs_valid, all_labels_valid)
                # eval_auc = roc_auc_score(all_labels_valid.ravel(), all_probs_valid.ravel())

                # for j in range(n_classes):
                #     writer.add_scalar("AUC_valid/{}".format(class_names[j]), eval_metrics["aucs"][j], global_step=global_step)
                #     print("AUC for {:30} = {:.3f}".format(class_names[j], eval_metrics["aucs"][j]))
                # writer.add_scalar("AUC_valid/overall", eval_auc, global_step=global_step)
                # writer.add_scalars("loss", {"train": loss.item(), "valid": eval_loss}, global_step=global_step)
            
                # if eval_auc > modelsinfo[-1].auc:
                #     if modelsinfo[-1].path != "output": # delete the worst existing model
                #         os.remove(os.path.join(outdir, modelsinfo[-1].path))
                    
                #     timenow = int(round(time.time(), 0))
                #     outfile = "epoch{}_itr{}.pt".format(epoch+1, i)
                #     modelsinfo[-1] = modelinfo(timenow, outfile, eval_auc)
                #     modelsinfo = sorted(modelsinfo, key=lambda model: model.auc, reverse=True)

                #     torch.save({
                #         "model_state_dict": model.state_dict(),
                #         "valid_loss": eval_loss,
                #         "epoch": epoch,
                #         "global_step": global_step,
                #         "params": P,
                #         "classes": class_names,
                #         "labels": all_labels_valid,
                #         "probs": all_probs_valid},
                #         # "models": modelsinfo}, 
                #         os.path.join(outdir, outfile))

        scheduler.step()

        epoch_loss = running_loss / len(dataloaders["train"])
        print('Train Loss: {:.4f}'.format(epoch_loss))

        print()

    best_auc = modelsinfo[0].auc
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_auc))

    # model.load_state_dict(best_model_wts)
    return modelsinfo


if __name__ == "__main__":
    model = TweetModel(Config.roberta_config)

    train, test, _ = read_data()
    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    train_dataset = TweetData(train, tokenizer, Config.max_len)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=32)
    }

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_model(model, dataloaders, Config.modelsdir, criterion, optimizer, scheduler, num_epochs=10)