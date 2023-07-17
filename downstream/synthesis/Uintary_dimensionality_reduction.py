import random

import jax
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import optax
# 还得传参数进去
from jax import grad, jit
from jax import numpy as jnp
from jax import pmap, vmap
from sklearn.utils import shuffle
# from tensorcircuit.backends.jax_backend import optax_optimizer

def sparse_vec2vec(sparse_vec, vec_size): 
    vec = np.zeros((vec_size, 1))
    for index1 in sparse_vec:
        vec[index1][0] = 1
    return vec

def batch(X, Y = None, batch_size = 100, should_shuffle = False):
    if Y is not None:
        X = np.array(X)
        if should_shuffle:
            X,Y = shuffle(X, Y)
        for start in range(0, X.shape[0], batch_size):
            yield X[start: start+batch_size], Y[start: start+batch_size]
    else:
        X = np.array(X)
        if should_shuffle:
            X = shuffle(X)
        for start in range(0, X.shape[0], batch_size):
            yield X[start: start+batch_size]

def batch_sp(X, Y = None, batch_size = 100):
    pass


@jax.jit
def mds_reduce(parameters, x):
    return parameters @ x

def v_mds_reduce(parameters, X):
    return vmap(mds_reduce, in_axes=(None, 0), out_axes=0)(parameters, X)

@jax.jit
def dist(x1, x2):
    delta = x1-x2
    return jax.numpy.sum(delta*delta)

def normalization(x):
    """"
    归一化到区间{0,1]
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / (_range+1)

# 非常ugly
def vvmap_dist(X1, X2):
    vmap_dist = lambda x, X : vmap(dist, in_axes=(None, 0), out_axes=0)(x, X) #.mean()
    return vmap(vmap_dist, in_axes=(0, None), out_axes=0)(X1, X2) #.mean()

def batch_loss(parameters, X):
    reduced_X = vmap(mds_reduce, in_axes=(None, 0), out_axes=0)(parameters, X) # in_axes应该是传几个参数就扫几个

    X_dist = vvmap_dist(X, X)
    reduce_X_dist = vvmap_dist(reduced_X, reduced_X)

    return optax.l2_loss(normalization(X_dist), normalization(reduce_X_dist)).mean()

def MDS(vecs, reduced_dim, epoch_num = 10, print_interval = 10):
    vec_size = vecs.shape[1]
    
    assert vecs[0].shape[1] == 1, vecs[0].shape

    optimizer = optax.adam(learning_rate=1e-2)
    
    params = jax.random.normal(shape=(reduced_dim, vec_size), key=jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    best_parms = None
    best_loss = 1e10

    donot_decearse_epoch = 0

    for epoch in range(epoch_num):
        loss_values = []
        for X in batch(vecs, batch_size = 100,):
            # print(X)
            loss_value, gradient = jax.value_and_grad(batch_loss)(params, X)
            updates, opt_state = optimizer.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            loss_values.append(loss_value)
            # print(loss_value)
        mean_loss = np.array(loss_values).mean()
        
        if epoch % print_interval == 0:
            print(f'mds, epoch: {epoch}, mean loss: {mean_loss}')
            
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_parms = params
            donot_decearse_epoch = 0
        else:
            donot_decearse_epoch += 1
            if donot_decearse_epoch > 5:
                break

    print(f'Finishd at epoch {epoch} with mean loss {best_loss}.')
    reduced_vecs = vmap(mds_reduce, in_axes=(None, 0), out_axes=0)(best_parms, vecs)
    return np.array(best_parms), np.array(reduced_vecs)

if __name__ == "__main__":
    pass