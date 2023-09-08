from circuit.dataset_loader import gen_algorithms
import pickle
import random

from circuit import gen_random_circuits
from circuit.dataset_loader import gen_algorithms
from circuit.formatter import layered_circuits_to_qiskit
from circuit.parser import qiskit_to_layered_circuits
from circuit.random_circuit import random_1q_layer
from circuit.utils import get_extra_info, stitching
from utils.backend import default_basis_single_gates, default_basis_two_gates
from utils.backend import get_devide_qubit, Backend


def gen_uncut_dataset(n_qubits, topology, neighbor_info, coupling_map):
    backend = Backend(n_qubits=n_qubits, topology=topology, neighbor_info=neighbor_info, basis_single_gates=default_basis_single_gates,
                      basis_two_gates=default_basis_two_gates, divide=False, decoupling=False)

    dataset = gen_random_circuits(min_gate=40, max_gate=120, n_circuits=15, two_qubit_gate_probs=[
                                  2, 5], gate_num_step=10, backend=backend, multi_process=True)

    dataset_machine = []
    for cir in dataset:
        cir['layer2gates'].reverse()
        max_layer = len(cir['layer2gates']) -1 
        cir['gate2layer'] = [max_layer - layer for layer in cir['gate2layer'] ] 
        cir['max_layer'] = max_layer
        dataset_machine.append(cir['layer2gates'])

    return dataset


def gen_cut_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size, min_cut_qubit = 5, align = False, devide_size=5,circuit_type = 'random'):
    covered_couplng_map = set()
    dataset = []

    devide_cut_backends = []
    devide_maps = []
    deivde_reverse_maps = []
    all_devide_qubits = []
    while True:
        before = len(covered_couplng_map)
        devide_qubits = get_devide_qubit(topology, devide_size)
        
        '''不能有只有一个比特的'''
        '''这明明是全等于5'''
        # has_1_qubit = False
        # for devide_qubit in devide_qubits:
        #     if len(devide_qubit) < min_cut_qubit:
        #         # print(len(devide_qubit))
        #         has_1_qubit = True
        #         break
        # if has_1_qubit:
        #     continue
        _devide_qubits = []
        for devide_qubit in devide_qubits:
            if len(devide_qubit) >= min_cut_qubit:
                _devide_qubits.append(devide_qubit)
        
        devide_qubits = _devide_qubits
        
        cut_backends = []
        maps = []
        reverse_maps = []
        for i in range(len(devide_qubits)):
            _map = {}
            _reverse_map = {}
            for idx, qubit in enumerate(devide_qubits[i]):
                _map[idx] = qubit
                _reverse_map[qubit] = idx
            maps.append(_map)
            reverse_maps.append(_reverse_map)

            cut_coupling_map = []
            for ele in coupling_map:
                if ele[0] in devide_qubits[i] and ele[1] in devide_qubits[i]:
                    cut_coupling_map.append(
                        (_reverse_map[ele[0]], _reverse_map[ele[1]]))
                    covered_couplng_map.add(tuple(ele))

            cut_backends.append(Backend(n_qubits=len(devide_qubits[i]), topology=topology, neighbor_info=neighbor_info, coupling_map=cut_coupling_map,
                                basis_single_gates=default_basis_single_gates, basis_two_gates=default_basis_two_gates, divide=False, decoupling=False))

        if before == len(covered_couplng_map):
            continue

        print(devide_qubits)
        all_devide_qubits.append(devide_qubits)
        devide_cut_backends.append(cut_backends)
        devide_maps.append(maps)
        deivde_reverse_maps.append(reverse_maps)

        if len(covered_couplng_map) == len(coupling_map):
            break

    # dataset_5qubit = []
    n_circuits = dataset_size // 60 // len(devide_cut_backends)
    for cut_backends, devide_qubits, maps,  reverse_maps in zip(devide_cut_backends, all_devide_qubits, devide_maps,  deivde_reverse_maps):
        cut_datasets = []
        for cut_backend in cut_backends:
            _dataset = gen_random_circuits(min_gate=20, max_gate=170, n_circuits=n_circuits, two_qubit_gate_probs=[
                                            1, 5], gate_num_step=10, backend=cut_backend, multi_process=True,circuit_type=circuit_type)
            cut_datasets.append(_dataset)

        # def get_n_instruction2circuit_infos(dataset):
        #     n_instruction2circuit_infos = defaultdict(list)
        #     for circuit_info in dataset:
        #         # qiskit_circuit = circuit_info['qiskit_circuit']
        #         gate_num = len(circuit_info['gates'])
        #         n_instruction2circuit_infos[gate_num].append(circuit_info)

        #     # print(n_instruction2circuit_infos[gate_num])
        #     gate_nums = list(n_instruction2circuit_infos.keys())
        #     gate_nums.sort()

        #     return n_instruction2circuit_infos, gate_nums

        # n_instruction2circuit_infos, gate_nums = get_n_instruction2circuit_infos(cut_datasets[0])
        # for gate in gate_nums:
        #     print(gate, len(n_instruction2circuit_infos[gate]))
        
        dataset += stitching(n_qubits, cut_datasets,
                             devide_qubits, maps, reverse_maps, align)

    print(len(dataset), "circuit generated")
    # print(len(dataset_5qubit), "5bit circuit generated")
    
    if align:
        dataset_machine = []
        for cir in dataset:
            dataset_machine.append(cir['layer2gates'])

        
    return dataset
    
    


def gen_algorithm_dataset(n_qubits, topology, neighbor_info, coupling_map, mirror):
    algos = gen_algorithms(n_qubits, coupling_map, mirror)

    get_extra_info(algos)
    for algo in algos:
        # print(layered_circuits_to_qiskit(18, algo['layer2gates']))
        print(algo['id'], len(algo['layer2gates']), len(
            algo['gates']), algo['duration'], algo['prop'])
    print(len(algos))
    algos_machine = []
    for cir in algos:
        algos_machine.append(cir['layer2gates'])

    title = "_mirror" if mirror else ""
    with open(f'execute_18bits_algos{title}.pkl', 'wb')as f:
        pickle.dump(algos_machine, f)

    with open(f'execute_18bits_algos_more_info{title}.pkl', 'wb')as f:
        pickle.dump(algos, f)

def gen_various_input_validate():
    with open('execute_18bits_validate_more_info_3000.pkl','rb')as f:
        dataset = pickle.load(f)
    
    random.shuffle(dataset)
    pick_10 = []
    for gate_num in range(200,400,20):
        for cir in dataset:
            if cir['gate_num'] == gate_num:
                pick_10.append(cir)
                break
    
    res = []
    for _cir in pick_10:
        for i in range(20):
            qc = layered_circuits_to_qiskit(18,_cir['layer2gates'],barrier = False)
            input = random_1q_layer(18, default_basis_single_gates)
            qc = input.compose(qc)
            qc = qc.compose(input.inverse())
            circuit_info = qiskit_to_layered_circuits(qc, False, False)
            circuit_info['id'] = _cir['id']+f'input_{i}'
            circuit_info['duration'] = _cir['duration'] + 60
            circuit_info['gate_num'] = _cir['gate_num'] + 36
            
            circuit_info['layer2gates'].reverse()
            max_layer = len(circuit_info['layer2gates']) -1 
            circuit_info['gate2layer'] = [max_layer - layer for layer in circuit_info['gate2layer'] ] 
            circuit_info['max_layer'] = max_layer
            
            res.append(circuit_info)
    
    
    dataset_machine = []
    for cir in res:
        dataset_machine.append(cir['layer2gates'])
    with open('execute_18bits_validate_various_input_200.pkl','wb')as f:
        pickle.dump(dataset_machine, f)

    with open('execute_18bits_validate_various_input_200_more_info.pkl','wb')as f:
        pickle.dump(res, f)
            


# size = 6
# n_qubits = 18
# topology = gen_grid_topology(size)  # 3x3 9 qubits
# new_topology = defaultdict(list)
# for qubit in topology.keys():
#     if qubit < n_qubits:
#         for ele in topology[qubit]:
#             if ele < n_qubits:
#                 new_topology[qubit].append(ele)
# topology = new_topology
# neighbor_info = copy.deepcopy(topology)
# coupling_map = topology_to_coupling_map(topology)

# # gen_cut_dataset(n_qubits, topology, neighbor_info, coupling_map, dataset_size = 1500,circuit_type = 'random')
# gen_validate_dataset(n_qubits, topology, neighbor_info, coupling_map)
# # gen_algorithm_dataset(n_qubits, topology, neighbor_info, coupling_map, True)
# # gen_algorithm_dataset(n_qubits, topology, neighbor_info, coupling_map, False)
# # gen_various_input_validate()