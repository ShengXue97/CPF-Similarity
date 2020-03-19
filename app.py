#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Hub Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from flask import Flask, request, abort, jsonify
import requests

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def get_correlation(labels, features, rotation):
  corr = np.inner(features, features)
  return corr[0][1]

def run_and_get_correlation(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(
      encoding_tensor, feed_dict={input_tensor_: messages_})
  return get_correlation(messages_, message_embeddings_, 90)

def get_similarity(input1, input2):
  messages = []
  messages.append(input1)
  messages.append(input2)

  similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
  similarity_message_encodings = embed(similarity_input_placeholder)
  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    return run_and_get_correlation(session, similarity_input_placeholder, messages,
                similarity_message_encodings)

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