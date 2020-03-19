#!/usr/bin/env python
# coding: utf-8

# # Sentence Similarity with Pretrained BERT
# In this notebook, we use a pretrained [BERT model](https://arxiv.org/abs/1810.04805) as a sentence encoder to measure sentence similarity. We use a [feature extractor](../../utils_nlp/bert/extract_features.py) that wraps [Hugging Face's PyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT) of Google's [BERT](https://github.com/google-research/bert). 
# 
# **Note: To learn how to do pre-training on your own, please reference the [AzureML-BERT repo](https://github.com/microsoft/AzureML-BERT) created by Microsoft.**

# ### 00 Global Settings

# In[2]:


import sys
import os
import torch
import itertools
import numpy as np
import pandas as pd
import scrapbook as sb
from collections import OrderedDict

sys.path.append("../../")
from utils_nlp.models.bert.common import Language, Tokenizer
from utils_nlp.models.bert.sequence_encoding import BERTSentenceEncoder, PoolingStrategy


# In[3]:


# device config
NUM_GPUS = 0

# model config
LANGUAGE = Language.ENGLISH
TO_LOWER = True
MAX_SEQ_LENGTH = 128
LAYER_INDEX = -2
POOLING_STRATEGY = PoolingStrategy.MEAN

# path config
CACHE_DIR = "./temp"


# In[4]:


if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)


# ### 01 Define the Sentence Encoder with Pretrained BERT

# The `BERTSentenceEncoder` defaults to Pretrained BERT.

# In[ ]:


se = BERTSentenceEncoder(
    language=LANGUAGE,
    num_gpus=NUM_GPUS,
    cache_dir=CACHE_DIR,
    to_lower=TO_LOWER,
    max_len=MAX_SEQ_LENGTH,
    layer_index=LAYER_INDEX,
    pooling_strategy=POOLING_STRATEGY,
)


# ### 02 Compute the Sentence Encodings

# The `encode` method of the sentence encoder accepts a list of text to encode, as well as the layers we want to extract the embeddings from and the pooling strategy we want to use. The embedding size is 768. We can also return just the values column as a list of numpy arrays by setting the `as_numpy` parameter to True.

# In[ ]:


result = se.encode(
    ["Coffee is good", "The moose is across the street"],
    as_numpy=False
)
result


# In[ ]:


# for testing
size_emb = len(result["values"].iloc[0])
sb.glue("size_emb", size_emb)


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to CPF Urgency Prediction Service!"

@app.route("/similarity", methods=["POST"])
def similarity():
    sim = request.json.get('sentence1','sentence2', None)
    if sim is None:
        abort(403)
    else:
        result = get_similarity(sim)
        return jsonify({
            'status': 'OK',
            'similarity': result,
        })
        
if __name__ == '__main__':
    app.run(debug=False, port=8668)