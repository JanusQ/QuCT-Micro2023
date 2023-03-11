from utils.backend import *


def count_path(topology, neigh_info, step):
    
    qubit_sizes  = []
    for qubit in topology.keys():
        coupling_size = 0
        for neibor in neigh_info[qubit]:
             coupling_size+= len(neigh_info[neibor])
             
        single_size = len(topology[qubit]) + 1 
        
        set_size = single_size * 3 + coupling_size - 1
        
        qubit_size = set_size**step * 2**step #fommer and latter
        
        # parallel
        set_size_parallel = [set_size]
        for i in range(1, step + 1):
            single_size_step = single_size - i
            single_size_step = single_size_step if single_size_step > 0 else 0
            coupling_size_step = coupling_size - i * coupling_size / single_size
            coupling_size_step = coupling_size_step if coupling_size_step > 0 else 0
            set_size_parallel.append(single_size_step * 3 + coupling_size_step - 1)
        
        if step == 1:
            qubit_size += set_size_parallel[1]            
        elif step == 2:
            qubit_size += set_size_parallel[1] * set_size_parallel[2] + 4 * set_size_parallel[0] * set_size_parallel[1]       
        elif step ==3:
            qubit_size += set_size_parallel[1] * set_size_parallel[2] * set_size_parallel[3] + 6 * set_size_parallel[0] * set_size_parallel[1] * set_size_parallel[2] + 12 * set_size_parallel[0]**2 * set_size_parallel[1]
        
        qubit_sizes.append(qubit_size)
        # print(qubit,qubit_size)

    return np.array(qubit_sizes).mean()
    
    
# for size in range(2,3):
#     for step  in range(1,4):
#         topology = gen_grid_topology(size) # 3x3 9 qubits
#         neigh_info = get_grid_neighbor_info(size, 1)
#         coupling_map = topology_to_coupling_map(topology)
#         print(neigh_info)
#         n_qubits = size**2
#         count = count_path(topology, neigh_info, step)
#         print((n_qubits,step),count)
    