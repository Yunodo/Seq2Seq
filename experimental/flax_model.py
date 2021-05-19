"""
Status: not working, to be updated
Bidirectional LSTM Seq2Seq model with attention
Modelled after : https://github.com/google/flax/blob/master/examples/seq2seq/train.py
"""

import random
import functools

from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.core import init, apply

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np


vocab_size = 8000
learning_rate = 0.1
batch_size = 32
hidden_size = 512
n_train_steps = 10

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

max_input_len = 20
max_output_len = 20


class EncoderLSTM(nn.Module):
  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    return nn.OptimizedLSTMCell()(carry, x)

  @staticmethod
  def initialize_carry(hidden_size):
    # use dummy key since default state init fn is just zeros.
    return nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), (), hidden_size)


class DecoderLSTM(nn.Module):
  teacher_force: bool
  vocab_size: int

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, carry, x):
    rng, lstm_state, last_prediction = carry
    carry_rng, categorical_rng = jax.random.split(rng, 2)
    if not self.teacher_force:
      x = last_prediction
    lstm_state, y = nn.OptimizedLSTMCell()(lstm_state, x)
    logits = nn.Dense(self.vocab_size)(y)
    predicted_token = jax.random.categorical(categorical_rng, logits)
    prediction = jnp.array(
        predicted_token == jnp.arange(self.vocab_size), dtype=jnp.float32)
    return (carry_rng, lstm_state, prediction), (logits, prediction)


class Decoder(nn.Module):
  """LSTM decoder."""
  init_state: Tuple[Any]
  teacher_force: bool
  vocab_size: int

  @nn.compact
  def __call__(self, inputs):
    # inputs.shape = (seq_length, vocab_size).
    lstm = DecoderLSTM(teacher_force=self.teacher_force, vocab_size = self.vocab_size)
    first_token = jax.lax.slice_in_dim(inputs, 0, 1)[0]
    init_carry = (self.make_rng('lstm'), self.init_state, first_token)
    _, (logits, predictions) = lstm(init_carry, inputs)
    return logits, predictions


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture.
  Attributes:
    teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
    hidden_size: int, the number of hidden dimensions in the encoder and
      decoder LSTMs.
  """
  teacher_force: bool
  hidden_size: int
  vocab_size: int

  @nn.compact
  def __call__(self, encoder_inputs, decoder_inputs):
    """Run the seq2seq model.
    Args:
      encoder_inputs: masked input sequences to encode, shaped
        `[len(input_sequence), vocab_size]`.
      decoder_inputs: masked expected decoded sequences for teacher
        forcing, shaped `[len(output_sequence), vocab_size]`.
        When sampling (i.e., `teacher_force = False`), the initial time step is
        forced into the model and samples are used for the following inputs. The
        first dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
    Returns:
      Array of decoded logits.
    """
    # Encoder.
    encoder = EncoderLSTM()
    init_carry = encoder.initialize_carry(self.hidden_size)
    init_decoder_state, _ = encoder(init_carry, encoder_inputs)
    # Decoder.
    decoder_inputs = jax.lax.slice_in_dim(decoder_inputs, 0, -1)
    decoder = Decoder(
        init_state=init_decoder_state,
        teacher_force=self.teacher_force, vocab_size = vocab_size)
    logits, predictions = decoder(decoder_inputs)

    return logits, predictions


def model(teacher_force=True):
  return Seq2seq(teacher_force=teacher_force,
                 hidden_size = hidden_size, vocab_size = vocab_size)


def get_initial_params(key, vocab_size):
  """Creates a seq2seq model."""
  encoder_shape = jnp.ones((max_input_len, vocab_size), jnp.float32)
  decoder_shape = jnp.ones((max_output_len, vocab_size), jnp.float32)
  return model().init({'params': key, 'lstm': key},
                      encoder_shape, decoder_shape)['params']

def cross_entropy_loss(logits, labels, lengths):
  """Returns cross-entropy loss."""
  xe = jnp.sum(nn.log_softmax(logits) * labels, axis=-1)
  masked_xe = jnp.mean(mask_sequences(xe, lengths))
  return -masked_xe


IN_SHAPES = ['n,_', ['m,_']]
OUT_ELEM = f'(m, {vocab_size})'
OUT_SHAPE = (OUT_ELEM, OUT_ELEM)
def apply_model(inputs, targets, in_masks, out_masks, params, key, teacher_force=True):
  @functools.partial(jax.mask, in_shapes=IN_SHAPES, out_shape=OUT_SHAPE)
  def model_fn(input, target):
    logits, predictions = model(teacher_force=teacher_force).apply(
        {'params': params},
        input,
        target,
        rngs={'lstm': key})
    return logits, predictions
  return jax.vmap(model_fn)(inputs, targets], dict(n=in_masks, m=out_masks))
