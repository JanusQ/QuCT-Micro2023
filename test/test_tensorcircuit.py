import numpy as np
import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")
ctype, rtype = tc.set_dtype("complex128")

def rx0(theta):
    return K.kron(
        K.cos(theta) * K.eye(2) + 1.0j * K.sin(theta) * tc.gates._x_matrix, K.eye(2)
    )


def rx1(theta):
    return K.kron(
        K.eye(2), K.cos(theta) * K.eye(2) + 1.0j * K.sin(theta) * tc.gates._x_matrix
    )


def ry0(theta):
    return K.kron(
        K.cos(theta) * K.eye(2) + 1.0j * K.sin(theta) * tc.gates._y_matrix, K.eye(2)
    )


def ry1(theta):
    return K.kron(
        K.eye(2), K.cos(theta) * K.eye(2) + 1.0j * K.sin(theta) * tc.gates._y_matrix
    )


def rz0(theta):
    return K.kron(
        K.cos(theta) * K.eye(2) + 1.0j * K.sin(theta) * tc.gates._z_matrix, K.eye(2)
    )


def rz1(theta):
    return K.kron(
        K.eye(2), K.cos(theta) * K.eye(2) + 1.0j * K.sin(theta) * tc.gates._z_matrix
    )


def cnot01():
    return K.cast(K.convert_to_tensor(tc.gates._cnot_matrix), ctype)


def cnot10():
    return K.cast(
        K.convert_to_tensor(
            np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ),
        ctype,
    )


ops_repr = ["rx0", "rx1", "ry0", "ry1", "rz0", "rz1", "cnot01", "cnot10"]


n, p, ch = 2, 3, 8
# number of qubits, number of layers, size of operation pool

target = tc.array_to_tensor(np.array([1, 0, 0, 1.0]) / np.sqrt(2.0))
# target wavefunction, we here use GHZ2 state as the objective target


def ansatz(params, structures):
    c = tc.Circuit(n)
    params = K.cast(params, ctype)
    structures = K.cast(structures, ctype)
    for i in range(p):
        c.any(
            0,
            1,
            unitary=structures[i, 0] * rx0(params[i, 0])
            + structures[i, 1] * rx1(params[i, 1])
            + structures[i, 2] * ry0(params[i, 2])
            + structures[i, 3] * ry1(params[i, 3])
            + structures[i, 4] * rz0(params[i, 4])
            + structures[i, 5] * rz1(params[i, 5])
            + structures[i, 6] * cnot01()
            + structures[i, 7] * cnot10(),
        )
    s = c.state()
    loss = K.sum(K.abs(target - s))
    return loss


vag1 = K.jit(K.vvag(ansatz, argnums=0, vectorized_argnums=1))


def sampling_from_structure(structures, batch=1):
    prob = K.softmax(K.real(structures), axis=-1)
    return np.array([np.random.choice(ch, p=K.numpy(prob[i])) for i in range(p)])


@K.jit
def best_from_structure(structures):
    return K.argmax(structures, axis=-1)


@K.jit
def nmf_gradient(structures, oh):
    """
    compute the Monte Carlo gradient with respect of naive mean-field probabilistic model
    """
    choice = K.argmax(oh, axis=-1)
    prob = K.softmax(K.real(structures), axis=-1)
    indices = K.transpose(K.stack([K.cast(tf.range(p), "int64"), choice]))
    prob = tf.gather_nd(prob, indices)
    prob = K.reshape(prob, [-1, 1])
    prob = K.tile(prob, [1, ch])

    return tf.tensor_scatter_nd_add(
        tf.cast(-prob, dtype=ctype),
        indices,
        tf.ones([p], dtype=ctype),
    )


nmf_gradient_vmap = K.vmap(nmf_gradient, vectorized_argnums=1)


verbose = False
epochs = 400
batch = 256
lr = tf.keras.optimizers.schedules.ExponentialDecay(0.06, 100, 0.5)
structure_opt = tc.backend.optimizer(tf.keras.optimizers.Adam(0.12))
network_opt = tc.backend.optimizer(tf.keras.optimizers.Adam(lr))
nnp = K.implicit_randn(stddev=0.02, shape=[p, 6], dtype=rtype)
stp = K.implicit_randn(stddev=0.02, shape=[p, 8], dtype=rtype)
avcost1 = 0
print('start')
for epoch in range(epochs):  # iteration to update strcuture param
    avcost2 = avcost1
    costl = []
    batched_stuctures = K.onehot(
        np.stack([sampling_from_structure(stp) for _ in range(batch)]), num=8
    )
    infd, gnnp = vag1(nnp, batched_stuctures)
    gs = nmf_gradient_vmap(stp, batched_stuctures)  # \nabla lnp
    gstp = [K.cast((infd[i] - avcost2), ctype) * gs[i] for i in range(infd.shape[0])]
    gstp = K.real(K.sum(gstp, axis=0) / infd.shape[0])
    avcost1 = K.sum(infd) / infd.shape[0]
    nnp = network_opt.update(gnnp, nnp)
    stp = structure_opt.update(gstp, stp)

    # if epoch % 40 == 0 or epoch == epochs - 1:
    print("----------epoch %s-----------" % epoch)
    print(
        "batched average loss: ",
        np.mean(avcost1),
    )

    if verbose:
        print(
            "strcuture parameter: \n",
            stp.numpy(),
            "\n network parameter: \n",
            nnp.numpy(),
        )

    cand_preset = best_from_structure(stp)
    print("best candidates so far:", [ops_repr[i] for i in cand_preset])
    print(
        "corresponding weights for each gate:",
        [K.numpy(nnp[j, i]) if i < 6 else 0.0 for j, i in enumerate(cand_preset)],
    )