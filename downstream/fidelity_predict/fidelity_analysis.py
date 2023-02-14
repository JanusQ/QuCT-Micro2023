from collections import defaultdict

from qiskit import QuantumCircuit

from upstream.dimensionality_reduction import batch
import numpy as np
from jax import numpy as jnp
import jax
from jax import vmap
import optax

error_param_rescale = 10000


class FidelityModel():
    def __init__(self, model):
        self.error_params = None
        self.model = model
        return

    def train(self):
        error_params = jnp.zeros(shape=(1, self.model.reduced_dim))
        optimizer = optax.adamw(learning_rate=1e-2)
        opt_state = optimizer.init(error_params)
        train_dataset = self.model.dataset
        error_params, opt_state = train_error_params(train_dataset, error_params, opt_state, optimizer, epoch_num=30)
        self.error_params = error_params

    def predict_fidelity(self, qc: QuantumCircuit):
        veced = self.model.vectorize(qc)
        error_params = self.error_params
        circuit_predict = smart_predict(error_params, veced['reduced_vecs'])
        gate_errors = np.array([
            jnp.dot(error_params / error_param_rescale, vec)
            for vec in veced['reduced_vecs']
        ])[:, 0]
        veced['gate_errors'] = gate_errors
        veced['circuit_predict'] = circuit_predict
        return circuit_predict, veced, gate_errors


def train_error_params(train_dataset, params, opt_state, optimizer, epoch_num=10):
    # 如果同时训练的数组大小不一致没办法使用vmap加速
    n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(train_dataset)
    print(gate_nums)
    for gate_num in gate_nums:
        best_loss_value = 1e10
        best_params = None
        for epoch in range(epoch_num):
            loss_values = []
            # print(n_instruction2circuit_infos[gate_num])
            n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
            for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100):
                loss_value, params, opt_state = epoch_train(circuit_infos, params, opt_state, optimizer)
                loss_values.append(loss_value)

            mean_loss = np.array(loss_values).mean()
            if mean_loss < best_loss_value:
                best_loss_value = mean_loss
                best_params = params

            if epoch % 10 == 0:
                print(f'gate num: {gate_num}, epoch: {epoch}, train mean loss: {mean_loss}')

        params = best_params

        # test_mean_losses = []
        # for circuit_info in test_dataset:
        #     test_x = np.array(circuit_info['reduced_vecs'], dtype=np.float32)
        #     test_y = np.array([circuit_info['ground_truth_fidelity']], dtype=np.float32)
        #     test_mean_loss = loss_func(best_params, test_x, test_y)
        #     test_mean_losses.append(test_mean_loss)

        # test_mean_loss = np.array(test_mean_losses).mean()
        # losses_hisotry.append(test_mean_loss)
        # gate_num_loss[gate_num] = test_mean_loss
        # print(gate_num, )
        # print(f'gate num: {gate_num}, test mean loss: {test_mean_loss}')

    print(f'taining error params finishs')
    return params, opt_state


@jax.jit
def smart_predict(params, reduced_vecs):
    '''预测电路的保真度'''
    errors = vmap(lambda params, reduced_vec: jnp.dot(params / error_param_rescale, reduced_vec), in_axes=(None, 0),
                  out_axes=0)(params, reduced_vecs)
    return jnp.product(1 - errors, axis=0)[0]  # 不知道为什么是[0][0]
    # return 1-jnp.sum([params[index] for index in sparse_vector])
    # np.array(sum(error).primal)


def loss_func(params, reduced_vecs, true_fidelity):
    # predict_fidelity = naive_predict(circ) # 对于电路比较浅的预测是准的，大概是因为有可能翻转回去吧，所以电路深了之后会比理论的保真度要高一些
    predict_fidelity = smart_predict(params, reduced_vecs)
    # print(predict_fidelity)
    # print(true_fidelity)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def batch_loss(params, X, Y):
    losses = vmap(loss_func, in_axes=(None, 0, 0), out_axes=0)(params, X, Y)
    return losses.mean()


# 在训练中逐渐增加gate num
def epoch_train(circuit_infos, params, opt_state, optimizer):
    # print(circuit_infos[0].keys())
    X = np.array([circuit_info['reduced_vecs'] for circuit_info in circuit_infos], dtype=np.float32)
    Y = np.array([[circuit_info['ground_truth_fidelity']] for circuit_info in circuit_infos], dtype=np.float32)

    loss_value, gradient = jax.value_and_grad(batch_loss)(params, X, Y)
    updates, opt_state = optimizer.update(gradient, opt_state, params)
    params = optax.apply_updates(params, updates)

    params = params.at[params > error_param_rescale / 10].set(error_param_rescale / 10)  # 假设一个特征对error贡献肯定小于0.1
    params = params.at[params < 0].set(0)

    return loss_value, params, opt_state


def get_n_instruction2circuit_infos(dataset):
    n_instruction2circuit_infos = defaultdict(list)
    for circuit_info in dataset:
        # qiskit_circuit = circuit_info['qiskit_circuit']
        gate_num = len(circuit_info['instructions'])
        n_instruction2circuit_infos[gate_num].append(circuit_info)

    # print(n_instruction2circuit_infos[gate_num])
    gate_nums = list(n_instruction2circuit_infos.keys())
    gate_nums.sort()

    return n_instruction2circuit_infos, gate_nums
