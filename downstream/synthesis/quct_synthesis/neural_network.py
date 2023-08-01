import random

import jax
import optax
from jax import numpy as jnp
from jax import vmap
from jax.config import config
from sklearn.model_selection import train_test_split

config.update("jax_enable_x64", True)

from sklearn.utils import shuffle

# 神经网络
# A helper function to randomly initialize weights and biases for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = jax.random.split(key)
  return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,), dtype=jnp.complex128)


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

  return sigmoid(logits).imag #(logits).imag  # 看下输出的区间  - logsumexp(logits)


def cost(params, x, y):
  return optax.l2_loss(neural_network_mapping(params, x), y)

def cost_batch(params, x, y):
  return vmap(cost, in_axes=(None, 0, 0))(params, x, y).sum()

class NeuralNetworkModel():
  def __init__(self, layer_sizes):
    self.layer_sizes = layer_sizes
    return
  
  def fit(self, X, Y, max_epoch = 100, batch_size = 20, test_size = 0.2):
    self.params =  init_network_params(self.layer_sizes, jax.random.PRNGKey(random.randint(0, 100)), scale=1e-2)
    opt = optax.adamw(learning_rate=1e-2)
    opt_state = opt.init(self.params)
    
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size)

    self.best_loss = 1e10
    
    print('fit:', len(X), 'data')
    for epoch in range(max_epoch):
        total_loss = 0
        
        X, Y = shuffle(X, Y)
        
        for start in range(0, len(X), batch_size):
            _X, _Y = X[start: start+batch_size], Y[start: start+batch_size]
            
            loss_value, gradient = jax.value_and_grad(cost_batch)(self.params, _X, _Y)
            updates, opt_state = opt.update(gradient, opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates) 
            total_loss += loss_value    
            
        # if epoch%10 ==0:
        test_loss = cost_batch(self.params, X_test, Y_test)
        print('epoch  %i\t| train loss = %.5f\t|  test loss = %.5f' % (epoch, total_loss, test_loss))
        
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.best_mapping = self.params
            
    return self.best_mapping
            