import random
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pennylane as qml

from jax import numpy as jnp
from jax import vmap
import jax
import optax
from jax.config import config
from jax.scipy.special import logsumexp

config.update("jax_enable_x64", True)

from sklearn.utils import shuffle

# 神经网络
# A helper function to randomly initialize weights and biases for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
'''
    输入 -> 输出 
    layer_sizes = [2, 10, 10, 6]
'''
def init_network_params(sizes, key, scale=1e-2):
  keys = jax.random.split(key, len(sizes))
  return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
  return jnp.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def neural_network_mapping(params, x):
  # per-example predictions
  activations = x
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = sigmoid(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

