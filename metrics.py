"""
Metrics that are used during training for optimisation and
evaluation of progress. Modelled as Trax layers
"""
import trax
import jax.numpy as jnp

def SequenceLoss():
# Cross-entropy loss for sequence data

  def category_cross_entropy(model_output, targets):

    def one_hot(x, n_categories, dtype=jnp.float32):
      indices_less_than_n = jnp.arange(n_categories)
      return jnp.array(x[..., jnp.newaxis] == indices_less_than_n, dtype)

    n_categories = model_output.shape[2]
    target_distributions = one_hot(targets, n_categories)
    model_log_distributions = trax.layers.core.log_softmax(model_output)
    return jnp.average (-jnp.sum(target_distributions * model_log_distributions,
        axis = -1), axis = - 1)

  def f(model_output, targets):
        cross_entropies = category_cross_entropy(
        model_output, targets)
        return jnp.average(cross_entropies)
  return trax.layers.base.Fn('SequenceLoss', f)

def SequenceAccuracy():
# Accuracy metric for sequence, compares % of words that exactly match
  def f(model_output, targets):
    predictions = jnp.argmax(model_output, axis = -1)
    position_is_accurate = jnp.equal(predictions, targets)
    return jnp.average(jnp.average(position_is_accurate, axis = -1), axis = -1)

  return trax.layers.base.Fn('SequenceAccuracy', f)
