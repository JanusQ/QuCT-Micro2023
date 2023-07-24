# Import general libraries (needed for functions)
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


def run_rb(backend, simulator, rb_pattern, rb_circs, xdata, target_qubits, upstream_model = None, plot = False, involved_qubits = None):
    # if upstream_model is not None:
    #     assert upstream_model.backend == backend
    
    n_totoal_qubits = backend.n_qubits
    basis_gates = backend.basis_gates #['u1', 'u2', 'u3', 'cx']
    transpiled_circs_list = []
    
    rb_fit = rb.RBFitter(None, xdata, rb_pattern)
    shots = 500
    
    jobs = []
    for rb_index, rb_circ in enumerate(rb_circs):
        # 其实是一组电路, 应该是同一个电路的不同长度
        
        fit_rn_circ = transpile(rb_circ, basis_gates=['u2', 'u3', 'cx'])  # ibm 只能有u1, u2, u3和 cx 垃圾玩意
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
        else:
            error_rb_circ = real_rb_circ

        transpiled_circs_list.append(fit_rn_circ)
        
        '''TODO: 看下很多比特的时候的时间'''
        job = simulator.simulate_noise(error_rb_circ, shots, get_count=False)
        jobs.append(job)


        # Add data to the fitter
    rb_fit.add_data([job.result() for job in jobs])
        # print('After seed %d, alpha: %f, EPC: %f' %
        #     (rb_index, rb_fit.fit[0]['params'][1], rb_fit.fit[0]['epc']))

    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(1, 1, 1)

        # Plot the essence by calling plot_rb_data
        rb_fit.plot_rb_data(0, ax=ax, add_label=True, show_plt=False)
            
        # Add title and label
        # ax.set_title('%d Qubit RB'%(nQ), fontsize=18)
        # fig.savefig(f'./simulate_50_350/{n_totoal_qubits}/{involved_qubits}_rb.svg')
        plt.close()
        # plt.show()

    gpc = rb.rb_utils.gates_per_clifford(
        transpiled_circuits_list=transpiled_circs_list,
        clifford_lengths=xdata[0],
        basis=['u2', 'u3', 'cx'],
        qubits=target_qubits)

    epc = rb_fit.fit[0]['epc']
    return gpc, epc

def get_error_1q(target_qubit, backend, simulator, length_range = [20, 1500], upstream_model = None):
    # 先搞单比特的rb
    rb_pattern = [[target_qubit]]
    target_qubits = [target_qubit]
    rb_circs, xdata = rb.randomized_benchmarking_seq(
        rb_pattern=rb_pattern, nseeds=5, length_vector=np.arange(length_range[0], length_range[1], (length_range[1] - length_range[0]) //10 ), )  # seed 越多跑的越久

    # print(rb_circs[0][-1])

    gpc, epc = run_rb(backend, simulator, rb_pattern, rb_circs, xdata, target_qubits, upstream_model = upstream_model, plot = False, involved_qubits = target_qubit)

    # calculate 1Q EPGs
    epg = rb.calculate_1q_epg(gate_per_cliff=gpc, epc_1q=epc, qubit=target_qubit)

    return epg  # epg['u3'] 作为单比特门误差    #sum(epg.values()) / len(epg)


# TODO: 现在的噪音太小了
def get_error_2q(target_qubits, error_1qs, backend, simulator, length_range = [20, 600], upstream_model = None):
    assert len(target_qubits) == 2  and len(error_1qs) == 2
    
    # 先搞单比特的rb
    rb_pattern = [target_qubits]
    target_qubits = target_qubits
    
    # 这个好像特别慢
    rb_circs, xdata = rb.randomized_benchmarking_seq(
        rb_pattern=rb_pattern, nseeds=5, length_vector=np.arange(length_range[0], length_range[1], (length_range[1] - length_range[0]) //10), )  # seed 越多跑的越久

    # print(rb_circs[0][-1])

    gpc, epc = run_rb(backend, simulator, rb_pattern, rb_circs, xdata, target_qubits, upstream_model = upstream_model, plot = False, involved_qubits = target_qubits)

    # calculate 1Q EPGs
    epg = rb.calculate_2q_epg(
        gate_per_cliff=gpc,
        epc_2q=epc,
        qubit_pair=target_qubits,
        list_epgs_1q=error_1qs)
    
    return epg

@ray.remote
def get_error_2q_remote(target_qubits, error_1qs, backend, simulator, upstream_model):
    return get_error_2q(target_qubits, error_1qs, backend, simulator, upstream_model = upstream_model)

@ray.remote
def get_error_1q_remote(qubit, backend, simulator, upstream_model):
    return get_error_1q(qubit, backend, simulator, upstream_model=upstream_model)

def get_errors(backend: Backend, simulator: NoiseSimulator, upstream_model: RandomwalkModel = None, multi_process = False):
    qubits = list(range(backend.n_qubits))
    couplers = list(backend.coupling_map)
    
    if upstream_model is not None and upstream_model.dataset is not None:
        print('WARNING: upstream_model.dataset is not None when calling get_errors', )
    
    # TODO: ray.remote
    qubit_errors = wait([
        get_error_1q_remote.remote(qubit, backend, simulator, upstream_model=upstream_model)  
        if multi_process else  get_error_1q(qubit, backend, simulator, upstream_model=upstream_model)
        for qubit in qubits
    ])
    
    # q1, q2 = 2, 3
    # get_error_2q([q1, q2], [qubit_errors[q1], qubit_errors[q2]], backend, simulator, upstream_model = upstream_model)
    # multi_process  = False
    coupler_errors = wait([
        get_error_2q_remote.remote([q1, q2], [qubit_errors[q1], qubit_errors[q2]], backend, simulator, upstream_model = upstream_model)
        if multi_process else  get_error_2q([q1, q2], [qubit_errors[q1], qubit_errors[q2]], backend, simulator, upstream_model = upstream_model)
        for q1, q2 in couplers
    ])
    # 报过ValueError: `x0` is infeasible.的错
    return [
        error['u3']
        for error in qubit_errors
    ], coupler_errors
    