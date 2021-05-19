"""
Sample from the trained model
"""

import trax
from model import biLSTMwithAttn
import numpy as np
from trax import fastmath
from trax import layers as tl

def stream(model, inputs, batch_size=1, temperature=1.0):
  input, flipped_input = inputs
  starting_seq = np.full((batch_size, 1), 0, dtype=np.int32)
  sample = None
  while sample != np.array([[0]]) and sample != np.array([[1]]):
    logits = model((input, flipped_input, starting_seq))[0]
    logits = tl.log_softmax(logits[:, -1, :])
    sample = tl.logsoftmax_sample(logits, temperature=temperature)
    starting_seq = np.concatenate((starting_seq, sample[None,:]), axis = 1)
  return starting_seq
