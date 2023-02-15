import numpy as np


def get_extra_info(dataset):
    def get_layer_type_divide(layer2instructions):
        """
        返回layer2instructions的每层的类型，是单比特门层则为1；否则则为2
        """
        return [len(layer[0]['qubits']) for layer in layer2instructions]

    def get_couple_prop(circuit_info):
        couple = 0
        for ins in circuit_info['instructions']:
            if len(ins['qubits']) == 2:
                couple += 1
        return couple / (len(circuit_info['instructions']))

    def get_xeb_fidelity(circuit_info):
        fidelity = 1
        for instruction in circuit_info['instructions']:
            if len(instruction['qubits']) == 2:
                q0_id, q1_id = instruction['qubits'][0], instruction['qubits'][1]
                if q0_id > q1_id:
                    fidelity = fidelity * (1 - couple_average_error[q1_id])
                else:
                    fidelity = fidelity * (1 - couple_average_error[q0_id])
            else:
                q0_id = instruction['qubits'][0]
                fidelity = fidelity * (1 - single_average_error[q0_id])
        return fidelity * np.product((measure0_fidelity + measure1_fidelity) / 2)

    def get_circuit_duration(layer2instructions):
        single_gate_time = 30
        two_gate_time = 60
        layer_types = get_layer_type_divide(layer2instructions)

        duration = 0
        for layer_type in layer_types:
            if layer_type == 1:
                duration += single_gate_time
            elif layer_type == 2:
                duration += two_gate_time
            else:
                raise Exception(layer_type)

        return duration

    qubits = ['q3_15', 'q3_17', 'q3_19', 'q5_19', 'q7_19', ]

    single_average_error = np.array([0.084, 0.040, 0.083, 0.025, 0.037, ]) / 100  # 原先是带%的

    couple_average_error = np.array([0.6, 0.459, 0.537, 0.615, ]) / 100  # q1_q2, q2_q3, ..., q9_q10

    t1s = np.array([94.5, 124.1, 117.1, 124.8, 136.3])  # q1, q2, ..., q5

    t2s = np.array([5.04, 7.63, 5.67, 6.97, 4.40, ])  # q1, q2, ..., q10

    measure0_fidelity = np.array([0.97535460, 0.97535460, 0.9645634, 0.9907482, 0.96958333])
    measure1_fidelity = np.array([0.955646258, 0.97572327, 0.950431034, 0.9629411764, 0.9570833333])

    for cir in dataset:
        cir['xeb_fidelity'] = get_xeb_fidelity(cir)
        cir['duration'] = get_circuit_duration(cir['layer2instructions'])
        cir['prop'] = get_couple_prop(cir)
        cir['gate_num'] = len(cir['instructions'])
    return dataset

def label_ground_truth_fidelity(dataset, labels):
    assert len(dataset) == len(labels)
    assert isinstance(dataset[0], dict)
    for idx, cir in enumerate(dataset):
        cir['ground_truth_fidelity'] = labels[idx]

