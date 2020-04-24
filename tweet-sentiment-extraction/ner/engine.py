from config import Config
import argparse
import utils
import os
import spacy
import pandas as pd
from tqdm import tqdm
import datetime

def run_model(data, positivemodel, negativemodel, outputdir):
    print("Loading models...")
    positive_nlp = spacy.load(positivemodel)
    negative_nlp = spacy.load(negativemodel)

    data = data.dropna()

    textIDs, selected_texts = [], []
    jaccards = {
        "positive": [],
        "negative": [],
        "neutral": []
    }

    input_train = "selected_text" in data.columns

    print("Iterating data...")
    for _, row in tqdm(data.iterrows()):
        if row.sentiment == "positive":
            if len(row.text.split()) <= 2:
                selected_text = row.text
            else: 
                ents = positive_nlp(row.text).ents
                selected_text = ents[0].text if len(ents) > 0 else row.text
        elif row.sentiment == "negative":
            if len(row.text.split()) <= 2:
                selected_text = row.text
            else:
                ents = negative_nlp(row.text).ents
                selected_text = ents[0].text if len(ents) > 0 else row.text
        else:
            selected_text = row.text

        textIDs += [row.textID]
        selected_texts += [selected_text]

        if input_train:
            jaccard = utils.jaccard_similarity(row.text, selected_text)
            jaccards[row.sentiment] += [jaccard]

    output_df = pd.DataFrame({"textID": textIDs, "selected_text": selected_texts})
    
    if not input_train:
        suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        outputpath = os.path.join(outputdir, "submission_" + suffix + ".csv")
        
        print("Saving submission...")
        output_df.to_csv(outputpath, index=False)

    if input_train:
        nums, dens = [], []
        for key in ("positive", "negative", "neutral"):
            num = sum(jaccards[key])
            den = len(jaccards[key])
            jaccard_score = num / den
            print(f"Jaccard score for {key}: {num / den:.3f}")

            nums.append(num)
            dens.append(den)

        print(f"Jaccard score for overall: {sum(nums) / sum(dens):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument("-d", "--data-type", dest="data",
                        help="Type of data to be used",
                        default="train")

    parser.add_argument("-m", "--modelsdir", dest="modelsdir",
                        help="Location of models directory")

    args = parser.parse_args()
    
    modelsdir = args.modelsdir if args.modelsdir else Config.modelsdir

    train, test, _ = utils.read_data(Config.datadir)

    run_model(test, 
        os.path.join(modelsdir, "positive"),
        os.path.join(modelsdir, "negative"),
        outputdir=Config.outputdir
    )
    


