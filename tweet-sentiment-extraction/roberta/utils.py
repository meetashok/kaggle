import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords
from config import Config

from sklearn.model_selection import StratifiedKFold
import numpy as np

def kfold_indices(dataframe, n_splits=5):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    df_fold = {"index": [], "fold": []}

    splits = kfold.split(np.arange(dataframe.index.size), y=dataframe.sentiment)
    for i, (_, split) in enumerate(splits):
        df_fold["index"].extend(split)
        df_fold["fold"].extend([i+1]*len(split))

    merged = (dataframe
              .merge(pd.DataFrame(df_fold).set_index("index"), left_index=True, right_index=True)
             )
    
    return merged

def read_data(frac=1):
    print("Reading data...")
    train = pd.read_csv(os.path.join(Config.datadir, "train.csv")).sample(frac=frac).dropna().reset_index(drop=True)
    # train = train.query("sentiment == 'positive'")
    train = kfold_indices(train)

    test = pd.read_csv(os.path.join(Config.datadir, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(Config.datadir, "sample_submission.csv"))

    return (train, test, sample_submission)

def jaccard_similarity(string1, string2):
    wordset = lambda x: set(x.lower().split())
    a, b = wordset(string1), wordset(string2)

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))


def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.
    Source: https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
    """
    text = str(text).lower()
    text = re.sub("\[.*?\]", "", text) # remove words in square brackets
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


def remove_stopwords(words):
    return [word for word in words if word not in stopwords.words("english")]


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }