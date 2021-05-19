"""
A bidirectional LSTM sequence-to-sequence model with attention that maps
from a source-target tokenized text pair to activations over vocabulary set

Modelled with LSTMSeq2SeqSeqAttn in mind:
https://github.com/google/trax/blob/master/trax/models/rnn.py
"""
import trax
import jax.numpy as jnp
from trax import layers as tl

def biLSTMwithAttn(input_vocab_size=256,
                    target_vocab_size=256,
                    d_model=512,
                    n_encoder_layers=2,
                    n_decoder_layers=2,
                    n_attention_heads=1,
                    attention_dropout=0.0,
                    mode='train'):

    """
    Inputs(3):
    source: rank 2 tensor representing a batch of text strings via token
          IDs plus padding markers; shape is (batch_size, sequence_length). The
          tensor elements are integers in `range(input_vocab_size)`, and `0`
          values mark padding positions.
    flipped source: same as source, but tokens are in reversed order
    target: rank 2 tensor representing a batch of text strings via token
          IDs plus padding markers; shape is (batch_size, sequence_length). The
          tensor elements are integers in `range(output_vocab_size)`, and `0`
          values mark padding positions.
    Output(1): rank 3 tensor representing a batch of log-probability
      distributions for each sequence position over possible token IDs;
      shape is (batch_size, sequence_length, `vocab_size`).
    Args(8):
    input_vocab_size: Input vocabulary size -- each element of the input tensor
        should be an integer in `range(vocab_size)`. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    target_vocab_size: Target vocabulary size.
    d_model: Final dimension of tensors at most points in the model, including
        the initial embedding output.
    n_encoder_layers: Number of LSTM layers in the encoder.
    n_decoder_layers: Number of LSTM layers in the decoder after attention.
    n_attention_heads: Number of attention heads.
    attention_dropout: Stochastic rate (probability) for dropping an activation
        value when applying dropout within an attention block.
    mode: If `'predict'`, use fast inference. If `'train'`, each attention block
        will include dropout; else, it will pass all values through unaltered.
    """

    input_encoder = tl.Serial(
      tl.Embedding(input_vocab_size, d_model),
      [tl.LSTM(d_model) for _ in range(n_encoder_layers)]
    )

    pre_attention_decoder = tl.Serial(
      tl.ShiftRight(mode=mode),
      tl.Embedding(target_vocab_size, d_model),
      tl.LSTM(d_model, mode = mode)
    )

    def PrepareAttentionInputs():
    # Layer that prepares queries, keys, values and mask for attention
        def F(encoder_activations, flipped_encoder_activations, decoder_activations,
          input_tokens, flipped_input_tokens):
          keys = values = jnp.concatenate((encoder_activations,
                        flipped_encoder_activations), axis = 1)
          queries = decoder_activations
# Mask is 1 where inputs are not padding (0) and 0 where they are padding.
          mask = jnp.concatenate((input_tokens != 0,
                        flipped_input_tokens !=0), axis = 1)
# We need to add axes to the mask for attention heads and decoder length.
          mask = jnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
# Broadcast so mask is [batch, 1 for heads, decoder-len, encoder-len].
          mask = mask + jnp.zeros((1, 1, decoder_activations.shape[1], 1))
          mask = mask.astype(jnp.float32)
          return queries, keys, values, mask
        return tl.Fn('PrepareAttentionInputs', F, n_out=4)

    return tl.Serial(
        tl.Select([0, 1, 2, 0, 1, 2]), # [in-toks, flip-in-toks, target-toks] * 2
        tl.Parallel(input_encoder, input_encoder, pre_attention_decoder, None, None, None),
        PrepareAttentionInputs(),  # q, k, v, mask, in-toks, flip-in-toks, target-toks
        tl.Residual(
                      tl.AttentionQKV(d_model, n_heads=n_attention_heads,
                               dropout=attention_dropout, mode=mode,
                               cache_KV_in_predict=True)
                ),     # decoder-vecs, mask, target-toks
        tl.Select([0, 2]),
        [tl.LSTM(d_model, mode = mode) for _ in range(n_decoder_layers)],
        tl.Dense(target_vocab_size),
        tl.LogSoftmax()
        )
