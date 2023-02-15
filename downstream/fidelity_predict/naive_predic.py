@jax.jit
def naive_predict(naive_params, instructions):
    cal_fidelity = lambda qubits: jnp.where(qubits[1] == -1, naive_params['single'][qubits[0]],
                                            naive_params['double'][qubits[0]][qubits[1]]) / error_param_rescale
    fidelity = jnp.product(vmap(cal_fidelity, in_axes=(0,), out_axes=0)(instructions), axis=0)
    # for qubits in instructions:
    #     fidelity *= jnp.where(qubits[1] == -1, naive_params['single'][qubits[0]] / error_param_rescale, naive_params['double'][qubits[0]][qubits[1]] / error_param_rescale )
    # if qubits[1] == -1:
    #     fidelity *= naive_params['single'][qubits[0]] / error_param_rescale
    # else:
    #     fidelity *= naive_params['double'][qubits[0]][qubits[1]] / error_param_rescale
    return fidelity


def naive_loss(naive_params, instructions, true_fidelity):
    predict_fidelity = naive_predict(naive_params, instructions)
    return optax.l2_loss(true_fidelity - predict_fidelity) * 100


def naive_batch_loss(naive_params, X, Y):
    # losses = jnp.array([naive_loss(naive_params, x, y) for x, y in zip(X, Y)])
    losses = vmap(naive_loss, in_axes=(None, 0, 0), out_axes=0)(naive_params, X, Y)
    return losses.mean()


def naive_prase_circuit(circuit_info):
    return np.array([
        instruction['qubits'] if len(instruction['qubits']) == 2 else instruction['qubits'] + [-1]
        for instruction in circuit_info['gates']
    ])


# 在训练中逐渐增加gate num
def naive_epoch_train(circuit_infos, naive_params, naive_opt_state, naive_optimizer):
    # print(circuit_infos[0].keys())

    # print(circuit_infos[0]['qiskit_circuit'])

    X = np.array([naive_prase_circuit(circuit_info) for circuit_info in circuit_infos])

    Y = np.array([[circuit_info['ground_truth_fidelity']] for circuit_info in circuit_infos], dtype=np.float32)

    loss_value, gradient = jax.value_and_grad(naive_batch_loss)(naive_params, X, Y)
    updates, naive_opt_state = naive_optimizer.update(gradient, naive_opt_state, naive_params)
    naive_params = optax.apply_updates(naive_params, updates)

    naive_params['single'] = naive_params['single'].at[naive_params['single'] < 1 / error_param_rescale].set(
        1 / error_param_rescale)  # 假设一个特征对error贡献肯定小于0.1
    naive_params['single'] = naive_params['single'].at[naive_params['single'] < 0].set(0)

    naive_params['double'] = naive_params['double'].at[naive_params['double'] < 1 / error_param_rescale].set(
        1 / error_param_rescale)  # 假设一个特征对error贡献肯定小于0.1
    naive_params['double'] = naive_params['double'].at[naive_params['double'] < 0].set(0)

    return loss_value, naive_params, naive_opt_state


def naive_train(dataset, naive_params, naive_opt_state, naive_optimizer, epoch_num=10):
    # 如果同时训练的数组大小不一致没办法使用vmap加速
    n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(dataset)
    print(gate_nums)
    for gate_num in gate_nums:
        best_loss_value = 1e10
        best_params = None
        for epoch in range(epoch_num):
            loss_values = []
            n_instruction2circuit_infos[gate_num] = np.array(n_instruction2circuit_infos[gate_num])
            for circuit_infos in batch(n_instruction2circuit_infos[gate_num], batch_size=100):
                loss_value, naive_params, opt_state = naive_epoch_train(circuit_infos, naive_params, naive_opt_state,
                                                                        naive_optimizer)
                loss_values.append(loss_value)

            mean_loss = np.array(loss_values).mean()
            if mean_loss < best_loss_value:
                best_loss_value = mean_loss
                best_params = naive_params

            if epoch % 10 == 0:
                print(f'gate num: {gate_num}, epoch: {epoch}, mean loss: {mean_loss}')

        naive_params = best_params

        test_mean_losses = []
        for circuit_info in circuit_infos:
            test_x = naive_prase_circuit(circuit_info)
            test_y = np.array([circuit_info['ground_truth_fidelity']], dtype=np.float32)
            test_mean_loss = naive_loss(best_params, test_x, test_y)
            test_mean_losses.append(test_mean_loss)

        print(f'gate num: {gate_num}, test mean loss: {np.array(test_mean_losses).mean()}')

    print(f'taining finishs')
    return best_loss_value, naive_params, opt_state
