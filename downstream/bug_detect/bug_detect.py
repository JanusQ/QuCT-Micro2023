import statistics


def find_bug(circuit):  # bug_instructions是手动添加的bug的id

    circuit_info = model_bug.vectorize(circuit)
    gate_vecs = circuit_info['sparse_vecs']  # .reshape(-1, 100)
    num_qubits = circuit_info['num_qubits']
    instruction2nearest_circuits = []

    for analyzed_vec_index, analyzed_vec in enumerate(gate_vecs):
        dists = np.array(vmap(func_dist, in_axes=(0, None), out_axes=0)(
            num_qubits2positive_vecs[num_qubits], analyzed_vec))

        dist_indexs = np.argsort(dists)[:3]  # 小的在前面
        nearest_dists = dists[dist_indexs]

        dist_indexs = dist_indexs[nearest_dists < 2]

        nearest_positive_instructions = [
            num_qubits2instructions[num_qubits][_index]
            for _index in dist_indexs
        ]
        nearest_circuits = [
            elm[2]['id']
            for elm in nearest_positive_instructions
        ]

        nearest_circuits = set(nearest_circuits)

        instruction2nearest_circuits.append(nearest_circuits)

    bug_positions = []
    for index, nearest_circuits in enumerate(instruction2nearest_circuits):

        neighbor_nearest_circuits = []
        pre = 0 if (index - 6) < 0 else index - 6
        for nearest_circuit_set in instruction2nearest_circuits[pre:index] + instruction2nearest_circuits[
                                                                             index + 1: index + 6]:
            neighbor_nearest_circuits += list(nearest_circuit_set)

        # isbug = True
        # for nearest_circuit in nearest_circuits:
        #     if nearest_circuit in neighbor_nearest_circuits:
        #         isbug = False
        #         break
        # if isbug:
        #     bug_indexes.append(index)

        neighbor_mode_nearest_circuit1 = statistics.mode(neighbor_nearest_circuits)
        neighbor_nearest_circuits = [elm for elm in neighbor_nearest_circuits if elm != neighbor_mode_nearest_circuit1]

        if len(neighbor_nearest_circuits) > 0:
            neighbor_mode_nearest_circuit2 = statistics.mode(neighbor_nearest_circuits)
        else:
            neighbor_mode_nearest_circuit2 = None

        if neighbor_mode_nearest_circuit1 not in nearest_circuits:
            if neighbor_mode_nearest_circuit2 is None:
                bug_positions.append((circuit_info['gates'][index]['qubits'][1] if len(
                    circuit_info['gates'][index]['qubits']) == 2
                                      else circuit_info['gates'][index]['qubits'][0]
                                      , circuit_info['gate2layer'][index]))
            elif neighbor_mode_nearest_circuit2 not in nearest_circuits:
                bug_positions.append((circuit_info['gates'][index]['qubits'][1] if len(
                    circuit_info['gates'][index]['qubits']) == 2
                                      else circuit_info['gates'][index]['qubits'][0]
                                      , circuit_info['gate2layer'][index]))

    return bug_positions

def func_dist(vec1, vec2):
    return jnp.sqrt(sp_dist(vec1, vec2) / 1000000)
