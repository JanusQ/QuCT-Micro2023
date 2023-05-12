# Import general libraries (needed for functions)
from upstream.randomwalk_model import extract_device
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb
from simulator import NoiseSimulator, get_random_erroneous_pattern
# Import Qiskit classes
import qiskit
from simulator.noise_simulator import add_pattern_error_path
from utils.backend import devide_chip, gen_grid_topology, get_grid_neighbor_info, Backend, topology_to_coupling_map
from collections import defaultdict
from utils.backend import default_basis_single_gates, default_basis_two_gates
import copy
import os
from upstream import RandomwalkModel
import cloudpickle as pickle
from qiskit import assemble, transpile
import ray

from utils.ray_func import wait
# Generate RB circuits (2Q RB)

# number of qubits
n_totoal_qubits = 18
n_qubits = n_totoal_qubits

with open(f"execute_18bit/error_params_predicts_execute_18bits_train_0_2500_step2.pkl", "rb")as f:
    downstream_model, predicts, reals, durations ,test_dataset = pickle.load(f)
upstream_model = downstream_model.upstream_model

backend: Backend = upstream_model.backend
backend = Backend(backend.n_qubits, topology=backend.topology,)

erroneous_pattern: dict = upstream_model.erroneous_pattern
basis_gates: list = backend.basis_gates

simulator = NoiseSimulator(backend)


def run_rb(n_totoal_qubits, backend, rb_pattern, rb_circs, xdata, target_qubits, upstream_model=None, plot=False, involved_qubits=None):

    if upstream_model is not None:
        assert upstream_model.backend == backend

    basis_gates = backend.basis_gates  # ['u1', 'u2', 'u3', 'cx']
    transpiled_circs_list = []

    rb_fit = rb.RBFitter(None, xdata, rb_pattern)
    shots = 500

    jobs = []
    for rb_index, rb_circ in enumerate(rb_circs):
        # 其实是一组电路, 应该是同一个电路的不同长度

        # ibm 只能有u1, u2, u3和 cx 垃圾玩意
        fit_rn_circ = transpile(rb_circ, basis_gates=['u2', 'u3', 'cx'])
        real_rb_circ = transpile(rb_circ, basis_gates=basis_gates)  # 实际执行的电路

        if upstream_model is not None:

            error_rb_circ = []
            for index, elm in enumerate(real_rb_circ):
                elm, n_error = add_pattern_error_path(
                    elm, n_totoal_qubits, upstream_model, upstream_model.erroneous_pattern)

                elm2 = fit_rn_circ[index]

                elm.name = elm2.name
                new_creg = elm._create_creg(len(target_qubits), "cr")
                elm.add_register(new_creg)
                for cbit, qubit in enumerate(target_qubits):
                    elm.barrier()
                    elm.measure(qubit, cbit)

                error_rb_circ.append(elm)
                # if n_error != 0:
                #     print('wow')

            # print(error_rb_circ[-1])
        else:
            error_rb_circ = real_rb_circ
            # print(error_rb_circ[-1])

        transpiled_circs_list.append(fit_rn_circ)

        '''TODO: 看下很多比特的时候的时间'''
        job = simulator.simulate_noise(error_rb_circ, shots, get_count=False)
        jobs.append(job)

    for job in jobs:
        # Add data to the fitter
        rb_fit.add_data(job.result())
        # print('After seed %d, alpha: %f, EPC: %f' %
        #     (rb_index, rb_fit.fit[0]['params'][1], rb_fit.fit[0]['epc']))

    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)

        # Plot the essence by calling plot_rb_data
        rb_fit.plot_rb_data(0, ax=ax, add_label=True, show_plt=False)

        # Add title and label
        # ax.set_title('%d Qubit RB'%(nQ), fontsize=18)
        fig.savefig(
            f'./simulate_50_350/{n_totoal_qubits}/{involved_qubits}_rb.svg')
        plt.close()
        # plt.show()

    gpc = rb.rb_utils.gates_per_clifford(
        transpiled_circuits_list=transpiled_circs_list,
        clifford_lengths=xdata[0],
        basis=['u2', 'u3', 'cx'],
        qubits=target_qubits)

    epc = rb_fit.fit[0]['epc']
    return gpc, epc


def get_error_1q(target_qubit, backend, length_range=[20, 1500], upstream_model=None):
    # 先搞单比特的rb
    rb_pattern = [[target_qubit]]
    target_qubits = [target_qubit]
    rb_circs, xdata = rb.randomized_benchmarking_seq(
        rb_pattern=rb_pattern, nseeds=5, length_vector=np.arange(length_range[0], length_range[1], (length_range[1] - length_range[0]) // 10), )  # seed 越多跑的越久

    # print(rb_circs[0][-1])

    gpc, epc = run_rb(n_totoal_qubits, backend, rb_pattern, rb_circs, xdata, target_qubits,
                      upstream_model=upstream_model, plot=True, involved_qubits=target_qubit)

    # calculate 1Q EPGs
    epg = rb.calculate_1q_epg(
        gate_per_cliff=gpc, epc_1q=epc, qubit=target_qubit)

    return epg  # epg['u3'] 作为单比特门误差    #sum(epg.values()) / len(epg)

# 贼多比特应该也能跑
# _backend = Backend(n_qubits=200, basis_single_gates=default_basis_single_gates, basis_two_gates=default_basis_two_gates)
# print(get_error_1q(189, _backend, upstream_model = upstream_model)))

# for qubit in range(n_totoal_qubits):
#     error_1q = get_error_1q(qubit, backend, upstream_model = upstream_model)
#     print(qubit, error_1q)

# TODO: 现在的噪音太小了


def get_error_2q(target_qubits, error_1qs, backend, length_range=[20, 600], upstream_model=None):
    assert len(target_qubits) == 2 and len(error_1qs) == 2

    # 先搞单比特的rb
    rb_pattern = [target_qubits]
    target_qubits = target_qubits

    # 这个好像特别慢
    rb_circs, xdata = rb.randomized_benchmarking_seq(
        rb_pattern=rb_pattern, nseeds=5, length_vector=np.arange(length_range[0], length_range[1], (length_range[1] - length_range[0]) // 10), )  # seed 越多跑的越久

    # print(rb_circs[0][-1])

    gpc, epc = run_rb(n_totoal_qubits, backend, rb_pattern, rb_circs, xdata, target_qubits,
                      upstream_model=upstream_model, plot=True, involved_qubits=target_qubits)

    # calculate 1Q EPGs
    epg = rb.calculate_2q_epg(
        gate_per_cliff=gpc,
        epc_2q=epc,
        qubit_pair=target_qubits,
        list_epgs_1q=error_1qs)

    return epg


p1Q = 0.002
epg_q0 = {'u1': 0, 'u2': p1Q/2, 'u3': 2 * p1Q/2}
epg_q1 = {'u1': 0, 'u2': p1Q/2, 'u3': 2 * p1Q/2}

# print(get_error_2q([3, 4], [epg_q0, epg_q1], backend, upstream_model = upstream_model))

# [0, 1]
# [1, 2]
# [3, 4]
# [0, 3]
# [1, 4]
upstream_model.dataset = None


@ray.remote
def get_error_2q_remote(target_qubits, error_1qs, backend, upstream_model):
    return get_error_2q(target_qubits, error_1qs, backend, upstream_model=upstream_model)


@ray.remote
def get_error_1q_remote(qubit, backend, upstream_model):
    return get_error_1q(qubit, backend, upstream_model=upstream_model)


def get_errors(backend: Backend, upstream_model: RandomwalkModel = None, multi_process=False):
    qubits = list(range(backend.n_qubits))
    couplers = list(backend.coupling_map)

    if upstream_model is not None and upstream_model.dataset is not None:
        print('WARNING: upstream_model.dataset is not None when calling get_errors', )

    # TODO: ray.remote
    qubit_errors = wait([
        get_error_1q_remote.remote(qubit, backend, upstream_model=upstream_model) if multi_process else get_error_1q(
            qubit, backend, upstream_model=upstream_model)
        for qubit in qubits
    ])

    coupler_errors = wait([
        get_error_2q_remote.remote([q1, q2], [qubit_errors[q1], qubit_errors[q2]], backend, upstream_model=upstream_model) if multi_process else get_error_2q(
            [q1, q2], [qubit_errors[q1], qubit_errors[q2]], backend, upstream_model=upstream_model)
        for q1, q2 in couplers
    ])

    return [
        error['u3']
        for error in qubit_errors
    ], coupler_errors

retrain = True
if retrain:
    all_errors = get_errors(backend, upstream_model=None,
                        multi_process=True)  # upstream_model
    print(all_errors)
    single_average_error = {}
    couple_average_error = {}
    for q, e in enumerate(all_errors[0]):
        single_average_error[q] = e
    for c, e in zip(list(backend.coupling_map), all_errors[1]):
        couple_average_error[tuple(c)] = e
        
    print(single_average_error, couple_average_error)
    with open(f"execute_18bit/rb/rb_error.pkl", "wb")as f:
        pickle.dump((single_average_error, couple_average_error), f)
else:
    with open(f"execute_18bit/rb/rb_error.pkl", "rb")as f:
        single_average_error, couple_average_error = pickle.load(f)



plot = False
if plot:
    def get_xeb_fidelity(circuit_info):
        fidelity = 1
        for gate in circuit_info['gates']:
            device = extract_device(gate)
            if isinstance(device, tuple):
                device = (circuit_info['map'][device[0]],
                        circuit_info['map'][device[1]])
                fidelity = fidelity * (1 - couple_average_error[device])
            else:
                device = circuit_info['map'][device]
                fidelity = fidelity * (1 - single_average_error[device])
        # * np.product((measure0_fidelity + measure1_fidelity) / 2)
        return fidelity




    xebs = []
    for cir in test_dataset:
        xebs.append(get_xeb_fidelity(cir))

    xebs = np.array(xebs)

    print('average inaccuracy = ', np.abs(xebs - reals).mean())
    print('average inaccuracy = ', np.abs(xebs - reals).std())
    
    fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
    axes.axis([0, 1, 0, 1])
    axes.scatter(reals, xebs)
    axes.set_xlabel('real')
    axes.set_ylabel('rb')
    axes.plot([[0, 0], [1, 1]])
    fig.savefig(f"execute_18bit/rb/real_rb_{n_qubits}_step1.svg")
