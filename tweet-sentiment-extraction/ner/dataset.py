import pandas as pd
from config import Config

class Dataset:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
        

    def clean_data(self):
        cleaned_data = (self.data
            .dropna()
            .reset_index(drop=True)
            .assign(
                text=lambda x: x.apply(lambda x: x.text.lower(), axis=1),
                selected_text=lambda x: x.apply(lambda x: x.selected_text.lower(), axis=1),
            )
        )

        self.cleaned_data = cleaned_data

    def build_training_data(self):
        print("Building positive and negative datasets for NER training...")
        if self.cleaned_data is None:
            self.clean_data()

        pos_data, neg_data = [], []

        for i, row in self.cleaned_data.iterrows():
            start = row.text.find(row.selected_text)
            end = start + len(row.selected_text)
            if end > start:
                train_row = (row.text, {"entities": [(start, end, Config.LABEL)]})

                if row.sentiment == "positive":
                    pos_data += [train_row]
                elif row.sentiment == "negative":
                    neg_data += [train_row]
                else:
                    pass

        print(f"Positive data size: {len(pos_data):,}")
        print(f"Negative data size: {len(neg_data):,}")

        return pos_data, neg_data

    
if __name__ == "__main__":
    import utils
    train, test, _ = utils.read_data(Config.datadir)

    dataset = Dataset(train)

    pos_data, neg_data = dataset.build_training_data()
