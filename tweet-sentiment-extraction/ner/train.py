#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import utils
import datetime
import os
import re
import string
import pandas as pd
from tqdm import tqdm
import argparse
from config import Config
from dataset import Dataset

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting


def train_model(traindata, new_model_name, model=None, modelsdir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(Config.LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    # ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(traindata)
            batches = minibatch(traindata, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # # test the trained model
    # test_text = "i`d have responded, if i were going"
    # doc = nlp(test_text)
    # print("Entities in '%s'" % test_text)
    # for ent in doc.ents:
    #     print(ent.label_, ent.text)

    # save model to output directory
    if modelsdir is not None:
        modelsdir = Path(modelsdir)
        if not modelsdir.exists():
            modelsdir.mkdir()
        nlp.to_disk(modelsdir)
        print("Saved model to", modelsdir)

        # # test the saved model
        # print("Loading from", modelsdir)
        # nlp2 = spacy.load(modelsdir)
        # # Check the classes have loaded back consistently
        # assert nlp2.get_pipe("ner").move_names == move_names
        # doc2 = nlp2(test_text)
        # for ent in doc2.ents:
        #     print(ent.label_, ent.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello")

    parser.add_argument(
        "-i",
        "--iterations",
        dest="n_iter",
        help="Number of iterations for training",
        default=10,
        type=int,
    )

    parser.add_argument(
        "-m", "--modelsdir", dest="modelsdir", help="Location of models directory"
    )

    args = parser.parse_args()
    n_iter = args.n_iter
    modelsdir = args.modelsdir if args.modelsdir else Config.modelsdir

    train, test, _ = utils.read_data(Config.datadir)
    dataset = Dataset(train)
    pos_data, neg_data = dataset.build_training_data()

    print("Training positive model...")
    train_model(
        pos_data,
        "positive",
        modelsdir=os.path.join(modelsdir, "positive"),
        n_iter=n_iter,
    )

    print("Training negative model...")
    train_model(
        neg_data,
        "negative",
        modelsdir=os.path.join(modelsdir, "negative"),
        n_iter=n_iter,
    )

