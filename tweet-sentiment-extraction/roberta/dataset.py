from torch.utils.data import Dataset, DataLoader
from utils import read_data
from tokenizers import ByteLevelBPETokenizer
from config import Config
import torch
import numpy as np
from model import initialize_tokenizer

def process_tweet(tweet, selected_text, sentiment, tokenizer, max_len):
    sentiment_ids = {
            'positive': 1313, 
            'negative': 2430, 
            'neutral': 7974
    }
    
    # initializing ids, attention_mask, token_type_ids
    ids = np.ones((max_len), dtype=np.int32)
    attention_mask = np.zeros((max_len), dtype=np.int32)
    token_type_ids = np.zeros((max_len), dtype=np.int32)
    
    # removing extra spaces and encoding tweet
    tweet = " " + " ".join(tweet.split())
    selected_text = " ".join(selected_text.split())
    encoded_tweet = tokenizer.encode(tweet)
        
    # filling the ids and attention_mask
    ids_valid = [0] + encoded_tweet.ids + [2, 2] + [sentiment_ids[sentiment]] + [2]
    len_valid = len(ids_valid)
    attention_mask_valid = [1] * len_valid

    ids[:len_valid] = ids_valid
    attention_mask[:len_valid] = attention_mask_valid
    
    
    # finding the start and end of selected_text
    selected_text_len = len(selected_text)

    for idx, char in enumerate(tweet):
        if char == selected_text[0]:
            if tweet[idx:selected_text_len+idx] == selected_text:
                char_start = idx
                char_end = char_start + selected_text_len

    assert char_start is not None
    assert char_end is not None
    assert tweet[char_start:char_end] == selected_text
        
    # finding the start and end tokens
    for token_index, (offset_start, offset_end) in enumerate(encoded_tweet.offsets):
        if (char_start >= offset_start) and (char_start <= offset_end):
            token_start = token_index
        if (char_end-1 >= offset_start) and (char_end <= offset_end):
            token_end = token_index
        
    assert token_start is not None
    assert token_end is not None
    
    token_start += 4
    token_end += 4
    
    return (
        ids,
        attention_mask,
        token_type_ids,
        token_start,
        token_end,
        tweet,
        selected_text
    )

class TweetData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, item):
        row = self.dataframe.loc[item]

        tweet = row.text
        selected_text = row.selected_text
        sentiment = row.sentiment

        (
        ids,
        attention_mask,
        token_type_ids,
        token_start,
        token_end,
        tweet,
        selected_text
        ) = process_tweet(tweet, selected_text, sentiment, self.tokenizer, self.max_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "token_start": torch.tensor(token_start, dtype=torch.long),
            "token_end": torch.tensor(token_end, dtype=torch.long),
            "tweet": tweet,
            "selected_text": selected_text,
            "sentiment": sentiment
        }

if __name__ == "__main__":
    train, test, _ = read_data()

    tokenizer = initialize_tokenizer(Config.roberta_vocab, Config.roberta_merges)
    train_dataset = TweetData(train, tokenizer, Config.max_len)

    for batch in DataLoader(train_dataset, batch_size=100):
        print(batch["ids"])
        print(batch["attention_mask"])
        print(batch["token_type_ids"])
        break