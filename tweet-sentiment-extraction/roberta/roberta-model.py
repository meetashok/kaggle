import tokenizers
import os
import utils
import numpy as np
from config import Config

import transformers

config = transformers.RobertaConfig.from_pretrained()

train, test, _ = utils.read_data(Config.datadir)

print(Config.roberta_merges, Config.roberta_vocab)


tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=Config.roberta_vocab, 
    merges_file=Config.roberta_merges, 
    lowercase=True,
    add_prefix_space=True
)

i = 1
text = train.iloc[i].text

encoded = tokenizer.encode(" ".join(text.split()))

encoded.ids[:20]
encoded.type_ids[:20]
encoded.tokens[:20]

nrecords, _ = train.shape
input_id = np.ones((nrecords, MAX_LEN), dtype="int32")
attention_mask = np.ones((nrecords, MAX_LEN), dtype="int32")
token_type_id = np.zeros((nrecords, MAX_LEN), dtype="int32")
start_token = np.zeros((nrecords, MAX_LEN), dtype="int32")
end_token = np.zeros((nrecords, MAX_LEN), dtype="int32")

print(encoded.tokens)
