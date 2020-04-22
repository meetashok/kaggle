import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords


def read_data(datadir):
    train = pd.read_csv(os.path.join(datadir, "train.csv"))
    test = pd.read_csv(os.path.join(datadir, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(datadir, "sample_submission.csv"))

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
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


def remove_stopwords(words):
    return [word for word in words if word not in stopwords.words("english")]
