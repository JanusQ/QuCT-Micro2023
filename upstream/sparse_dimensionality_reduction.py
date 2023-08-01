import jax

# import tensorflow as tf
import numpy as np
import optax

# 还得传参数进去
from jax import numpy as jnp
from jax import vmap
from sklearn.utils import shuffle


# from tensorcircuit.backends.jax_backend import optax_optimizer
# import ray


def sparse_vec2vec(sparse_vec, vec_size):
    vec = np.zeros((vec_size, 1))
    for index1 in sparse_vec:
        vec[index1][0] = 1
    return vec


def batch(
        X,
        Y=None,
        batch_size=100,
        should_shffule=False,
):
    if Y is not None:
        X = np.array(X)
        if should_shffule:
            X, Y = shuffle(X, Y)
        for start in range(0, X.shape[0], batch_size):
            yield X[start: start + batch_size], Y[start: start + batch_size]
    else:
        X = np.array(X)
        if should_shffule:
            X = shuffle(X)
        for start in range(0, X.shape[0], batch_size):
            yield X[start: start + batch_size]


@jax.jit
def mds_reduce(parameters, x):
    return parameters @ x


def v_mds_reduce(parameters, X):
    return vmap(mds_reduce, in_axes=(None, 0), out_axes=0)(parameters, X)


@jax.jit
def dist(x1, x2):
    delta = x1 - x2
    return jax.numpy.sum(delta * delta)


def normalization(x):
    """ "
    归一化到区间{0,1]
    """
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / (_range + 1)


# 非常ugly

def vvmap_dist(X1, X2):
    def vmap_dist(x, X):
        return vmap(dist, in_axes=(None, 0), out_axes=0)(x, X)  # .mean()

    return vmap(vmap_dist, in_axes=(0, None), out_axes=0)(X1, X2)  # .mean()


def batch_loss(parameters, X):
    reduced_X = vmap(mds_reduce, in_axes=(None, 0), out_axes=0)(
        parameters, X
    )  # in_axes应该是传几个参数就扫几个

    X_dist = vvmap_dist(X, X)
    reduce_X_dist = vvmap_dist(reduced_X, reduced_X)

    return optax.l2_loss(normalization(X_dist), normalization(reduce_X_dist)).mean()


def MDS(vecs, reduced_dim, epoch_num=10, print_interval=10):
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
        for X in batch(
                vecs,
                batch_size=10,
                should_shffule=True,
        ):
            # print(X)
            loss_value, gradient = jax.value_and_grad(batch_loss)(params, X)
            updates, opt_state = optimizer.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            loss_values.append(loss_value)
            # print(loss_value)
        mean_loss = np.array(loss_values).mean()

        if epoch % print_interval == 0:
            print(f"mds, epoch: {epoch}, mean loss: {mean_loss}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_parms = params
            donot_decearse_epoch = 0
        else:
            donot_decearse_epoch += 1
            if donot_decearse_epoch > 5:
                break

    print(f"Finishd at epoch {epoch} with mean loss {best_loss}.")
    reduced_vecs = vmap(mds_reduce, in_axes=(None, 0), out_axes=0)(best_parms, vecs)
    return np.array(best_parms), np.array(reduced_vecs)


mds_scale = 1e3


def _a2b_minus(a_index, a_value, b):
    b_indexs = b[0]
    b_values = b[1]
    correspond_b_index = (b_indexs == a_index)[0:]

    return jnp.array([a_index, a_value - jnp.sum(b_values * correspond_b_index)])


def _b2a_minus(a, b_index,b_value):
    a_indexs = a[0]
    correspond_a_index = (a_indexs == b_index)[0:]

    return jnp.array([b_index, (1 - jnp.sum(correspond_a_index)) * -b_value])

@jax.jit
def sp_minus(a, b):
    """a-b"""
    a_minus = vmap(_a2b_minus, in_axes=(0, 0, None), out_axes=0)(a[0], a[1], b)
    b_minus = vmap(_b2a_minus, in_axes=(None, 0,0), out_axes=0)(a, b[0], b[1])
    return jnp.concatenate((a_minus, b_minus))

def _a2b_pluse(a_index, a_value, b):
    b_indexs = b[0]
    b_values = b[1]
    correspond_b_index = (b_indexs == a_index[0])[0:]

    return jnp.array([a_index, a_value + jnp.sum(b_values * correspond_b_index)])

def _b2a_pluse(a, b_index, b_value):
    a_indexs = a[0]
    correspond_a_index = (a_indexs == b_index)[0:]

    return jnp.array([b_index, (1 - jnp.sum(correspond_a_index)) * b_value])

@jax.jit
def sp_pluse(a, b):
    """a-b"""
    a_plus = vmap(_a2b_pluse, in_axes=(0, 0, None), out_axes=0)(a[0], a[1], b)
    b_plus = vmap(_b2a_pluse, in_axes=(None, 0, 0), out_axes=0)(a, b[0], b[1])
    res = jnp.concatenate((a_plus, b_plus))

    # assert len(res.shape) == 3

    return res


@jax.jit
def sp_dist(x1, x2):
    delta = sp_minus(x1, x2)
    # print(delta.tolist())

    # delta = np.array(delta)

    return jnp.sum(delta[0:, 1] ** 2)


reduced_scaling = 1000  # 必须和RandomwalkModel里面的一样


def normalize(x):
    L = jnp.sqrt(jnp.sum(x[:, 1] ** 2))
    x = x.at[:, 1].set(x[:, 1] * reduced_scaling / L)
    # x.at[:,1].set(0)
    # x[:, 1]/L
    return x


@jax.jit
def sp_cos_dist(x1, x2):
    nx1, nx2 = normalize(x1), normalize(x2)
    # dnx1, dnx2 = construct_dense(nx1, 400000), construct_dense(nx2, 400000)

    # print(sum((dnx1 - dnx2)**2))

    dist = sp_dist(nx1, nx2) / reduced_scaling / reduced_scaling
    return (2 - dist) / 2


# def sp_vvmap_dist(X1, X2):
#     vmap_dist = lambda x, X : vmap(sp_dist, in_axes=(None, 0), out_axes=0)(x, X) #.mean()
#     return vmap(vmap_dist, in_axes=(0, None), out_axes=0)(X1, X2) #.mean()

@jax.jit
def sp_multi_constance(sparse_vec, constance):
    # sparse_vec[:,1] = sparse_vec[:,1] * constance
    sparse_vec = sparse_vec.at[:, 1].set(sparse_vec[:, 1] * constance)
    return sparse_vec


@jax.jit
def sp_dot(a, b):
    x_indexs = jnp.array(b[0], dtype=jnp.int64)

    x_values = jnp.array(b[1], dtype=jnp.int64)

    x_parameters = a[0:, x_indexs]  # (reduced_dim, nonzero_num)
    return x_parameters @ x_values


@jax.jit
def sp_mds_reduce(parameters, x):
    """
    parameters: (reduced_dim, vec_size)
    x: [
        [nonzero_indexes], [nonzero_values]
    ]
    return parameters @ x
    """
    return sp_dot(parameters, x)
    # x_indexs = x[0:,0]  # (nonzero_num, )

    # x_values = x[0:,1]
    # x_values = jnp.reshape.reshape(x_values, (x.shape[0], 1)) # (nonzero_num, 1)

    # x_parameters = parameters[0:,x_indexs] # (reduced_dim, nonzero_num)
    # return x_parameters @ x_values


@jax.jit
def sp_batch_loss(parameters, X):
    reduced_X = vmap(sp_mds_reduce, in_axes=(None, 0), out_axes=0)(
        parameters, X
    )  # in_axes应该是传几个参数就扫几个

    reduce_X_dist = vvmap_dist(reduced_X, reduced_X)
    X_dist = sp_vvmap_dist(X)
    # reduce_X_dist = jnp.zeros((10,10))

    # return jnp.array((10,)).mean()
    return optax.l2_loss(normalization(X_dist), normalization(reduce_X_dist)).mean()


# @ray.remote


def dist_vec(x, X):
    # print(x)
    vec = np.zeros((X.shape[0],))
    for i2, x2 in enumerate(X):
        vec[i2] = sp_dist(x, x2)

    return vec


def dist_matrix(X):
    # D = [None] * X.shape[0]
    D = np.zeros((X.shape[0], X.shape[0]))
    for i1, x1 in enumerate(X):
        # print(i1)
        # D[i1] = dist_vec.remote(x1, X)
        for i2, x2 in enumerate(X[i1:]):
            i2 += i1
            D[i1][i2] = sp_dist(x1, x2)
            D[i2][i1] = D[i1][i2]
    # D = [ray.get(future) for future in D]
    return np.array(D)


# 非常ugly


def sp_vvmap_dist(X):
    def vmap_dist(x, X):
        return vmap(sp_dist, in_axes=(None, 0), out_axes=0)(x, X)  # .mean()

    return vmap(vmap_dist, in_axes=(0, None), out_axes=0)(X, X)  # .mean()


def construct_dense(spare_vec, dim):
    vec = np.zeros((dim, 1))
    for i, v in spare_vec:
        vec[i] = v
    return vec


def reconstruct_dist_matrix(D, I):
    return D[I][0:, I]


# D[I][I]

def sp_MDS(vecs, vec_size, reduced_dim, epoch_num=10, print_interval=10, batch_size=10):
    optimizer = optax.adam(learning_rate=1e-2)

    params = jax.random.normal(shape=(reduced_dim, vec_size), key=jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    best_parms = None
    best_loss = 1e10

    donot_decearse_epoch = 0

    # indecies = np.arange(0, vecs.shape[0])
    # D = dist_matrix(vecs)
    # D = sp_vvmap_dist(vecs)

    for epoch in range(epoch_num):
        loss_values = []
        for X in batch(
                vecs,
                batch_size=batch_size,
                should_shffule=True,
        ):
            X = jnp.array(X, dtype=jnp.int64)
            # print(X)
            # print(1)
            # X_dist = reconstruct_dist_matrix(D, I)
            # print(2)
            # MX = np.array()

            loss_value, gradient = jax.value_and_grad(sp_batch_loss)(params, X)
            updates, opt_state = optimizer.update(gradient, opt_state, params)
            params = optax.apply_updates(params, updates)
            loss_values.append(loss_value)
            # print(loss_value)
        mean_loss = np.array(loss_values).mean()

        if epoch % print_interval == 0:
            print(f"mds, epoch: {epoch}, mean loss: {mean_loss}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_parms = params
            donot_decearse_epoch = 0
        else:
            donot_decearse_epoch += 1
            if donot_decearse_epoch > 5:
                break

    print(f"Finishd at epoch {epoch} with mean loss {best_loss}.")
    reduced_vecs = vmap(sp_mds_reduce, in_axes=(None, 0), out_axes=0)(best_parms, vecs)
    return np.array(best_parms), np.array(reduced_vecs)


def pad_to(sparse_vecs, path_values, size):
    pointer = 0
    vec_indexs = sparse_vecs.copy()
    vec_values = path_values
    for _ in range(size - len(vec_indexs)):
        while pointer in vec_indexs:
            pointer += 1
        vec_indexs.append(pointer)
        vec_values.append(0)
        pointer += 1
    vec_indexs = np.array(vec_indexs)
    vec_values = np.array(vec_values)
    sort_index = np.argsort(vec_indexs)
    vec_indexs = vec_indexs[sort_index]
    vec_values = vec_values[sort_index]
    return np.array([vec_indexs, vec_values])


def make_same_size(vecs, max_nonzero_num=None):
    """得每个稀疏向量得长度一样才能用vmap"""
    if max_nonzero_num is None:
        max_nonzero_num = 0
        for vec in vecs:
            vec.sort(key=lambda elm: elm[0][0])
            if len(vec) > max_nonzero_num:
                max_nonzero_num = len(vec)

    for vec in vecs:
        pointer = 0
        vec_indexs = [index[0] for index, elm in vec]
        for _ in range(max_nonzero_num - len(vec)):
            while pointer in vec_indexs:
                pointer += 1
            vec.append([[pointer], [0]])
            pointer += 1
        vec.sort(key=lambda elm: elm[0][0])

    return vecs, max_nonzero_num


# np.sum()
# @jax.jit
def test(a, b):
    index = b[0:, 0] == 5
    # return b.sum(where=index, axis=1) #
    # index = index.tolist() #.index(True)
    # index = np.array(index)
    # index = jnp.where(index, jnp.arange(a.shape[0]))
    # temp = jnp.take(a, index)
    # # temp = a[index]
    # print(index)
    # print(temp)
    # return temp.sum()
    ta = a[0:, 1] * index
    print(ta)
    return


if __name__ == "__main__":
    import random

    # a = jnp.array([
    #     [[1], [2]],
    #     [[3], [4]],
    # ])

    # b = jnp.array([
    #     [[2], [4]],
    #     [[5], [6]],
    # ])

    # print(jnp.array([[1],[2]]) * jnp.array([[False],[True]]))

    # a = jnp.array([
    #     [1, 2],
    #     [3, 4],
    # ])

    # b = jnp.array([
    #     [2, 4],
    #     [5, 6],
    # ])

    # c = test(a, b)
    # print(c.tolist())
    # print(sp_dist(a, b))

    vecs = []

    vec_size = 0

    # import ray
    # ray.init()

    for i in range(10000):
        vec = []
        i = 0
        while i < 100:
            i += random.randint(1, 20)
            # vec.append([[i], [random.random() * mds_scale]])
            vec.append([[i], [(random.random() * 3 + 1) * mds_scale]])

            if vec_size < i:
                vec_size = i

        vecs.append(vec)

    vecs = make_same_size(vecs)
    vecs = np.array(vecs)
    vecs = vecs.astype(np.int64)

    # vec_num = 50 #len(reduced_vecs)
    # print_nearest_num = 10
    # for i in range(vec_num):
    #     for j, vec in enumerate(vecs):
    #         _dist = float(sp_dist(vecs[i], vec))
    #         print(_dist)

    vec_size = np.max(vecs[0:, 0:, 0])
    vec_size += 1

    # vec1 = [[[0], [0]],
    #         [[2], [1]],
    #         [[5], [2]]]
    # vec2 = [[[0], [0]],
    #         [[4], [1]],
    #         [[6], [2]]]

    # vec1 = vecs[0] #np.array(vec1)
    # vec2 = vecs[1] #np.array(vec2)

    # dense_vec1 = construct_dense(vec1, vec_size)
    # dense_vec2 = construct_dense(vec2, vec_size)

    # # print(vecs[0])
    # # print(vecs[1])
    # print(sp_dist(vec1, vec2))

    # print(dist(dense_vec1, dense_vec2))

    params, reduced_vecs = sp_MDS(vecs, vec_size, 30, epoch_num=1000, batch_size=100)

    vec_num = 50  # len(reduced_vecs)
    print_nearest_num = 10
    for i in range(vec_num):
        dis1 = []
        dis2 = []
        for j, vec in enumerate(vecs):
            dis1.append([j, float(sp_dist(vecs[i], vec))])
        for j, vec in enumerate(reduced_vecs):
            dis2.append([j, float(dist(reduced_vecs[i], vec))])
        dis1.sort(key=lambda x: (x[1], x[0]))
        dis2.sort(key=lambda x: (x[1], x[0]))
        print(
            f"""向量{i}
        降维前最近{print_nearest_num}个向量：{[tuple(dis1[i]) for i in range(print_nearest_num * 5)]}
        降维后最近{print_nearest_num}个向量：{[tuple(dis2[i]) for i in range(print_nearest_num)]}
        交集：{set([dis1[i][0] for i in range(print_nearest_num * 5)]) & set([dis2[i][0] for i in range(print_nearest_num)])}
        """
        )

    print(reduced_vecs.shape)
