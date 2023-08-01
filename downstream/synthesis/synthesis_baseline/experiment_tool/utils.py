import time
import traceback
from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dagdependency, circuit_to_dag
from qiskit.dagcircuit.dagdepnode import DAGDepNode
from scipy.stats import unitary_group


def random_unitary(num_qubits):
    return unitary_group.rvs(2 ** num_qubits)


def timer(fn):
    """
    A decorator used for timing execution time of a function

    Usage:
        @timer
        def compile():
            ...

    """

    def wrapper(*args, **kwargs):
        start = time.time()  # 包括睡眠状态
        # process_start = time.process_time() # float（以小数表示的秒为单位）返回当前进程的系统和用户 CPU 时间的总计值。 它不包括睡眠状态所消耗的时间。 根据定义它只作用于进程范围。
        # thread_start = time.thread_time()# （以小数表示的秒为单位）返回当前线程的系统和用户 CPU 时间的总计值。 它不包括睡眠状态所消耗的时间。 根据定义它只作用于线程范围
        r, cpu_time = fn(*args, **kwargs)
        execution_time = time.time() - start
        # execution_process_time = time.process_time() - process_start
        # execution_thread_time = time.thread_time() - thread_start
        return execution_time, r, execution_time + cpu_time

    return wrapper


def get_cnot_cnt(qc):
    cnt = 0
    for ins in qc:
        if ins.operation.name in ['cx', 'cz', 'cnot']:
            cnt += 1
    return cnt


def get_depth(qc):
    dag = circuit_to_dag(qc)
    return len([l for l in dag.layers()])


def _get_op_nodes_by_level(nodes: List[DAGDepNode], qubit_num: int) -> List[List[DAGDepNode]]:
    # Filter warnings for back-reference index of a qubit
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Create graph for topological sort
    nodes = list(nodes)
    predecessors = [n.predecessors.copy() for n in nodes]
    successors = [n.successors.copy() for n in nodes]
    node_num = len(nodes)
    is_added = [False for i in range(node_num)]
    left_node = node_num

    # Result
    nodes_by_level = []

    # Topological sort by level
    while left_node > 0:
        # Find all nodes which have no predecessors
        # One qubit can only be used by one gate in every level
        is_used = [False for _ in range(qubit_num)]
        level = []

        for n in nodes:
            if not is_added[n.node_id] and len(predecessors[n.node_id]) == 0:
                # Check node gate type
                if n.op.num_qubits == 2:  # CZ gate
                    q0_id, q1_id = n.qargs[0].index, n.qargs[1].index

                    # Both q0 and q1 are not used by other gates in this level
                    if is_used[q0_id] or is_used[q1_id]:
                        continue
                    else:
                        is_used[q0_id] = is_used[q1_id] = True
                else:  # RX, RY, RZ gate
                    q0_id = n.qargs[0].index

                    # q0 is not used by other gates in this level
                    if is_used[q0_id]:
                        continue
                    else:
                        is_used[q0_id] = True

                # Add node `n` to this level
                level.append(n)
                is_added[n.node_id] = True
                left_node -= 1

        # Clear edges
        for n in level:
            for succ in successors[n.node_id]:
                predecessors[succ].remove(n.node_id)

        nodes_by_level.append(level)

    return nodes_by_level


def cal_circuit_parallelism(qc: QuantumCircuit):
    """
    Calculate average parallelism of a QuantumCircuit Obj
    """
    dag_dep = circuit_to_dagdependency(qc)
    nodes_by_level = _get_op_nodes_by_level(dag_dep.get_nodes(), qc.num_qubits)

    sum_p = 0
    for level in nodes_by_level:
        sum_p += len(level)
    return sum_p / len(nodes_by_level)


class ISynthesiser:
    def __init__(self, unitary: np.ndarray):
        self.unitary = unitary
        self.metrics = None

    def get_synthesiser_name(self) -> str:
        raise NotImplementedError('Override this method to get synthesiser name')

    def synthesis(self, unitary: np.ndarray):
        raise NotImplementedError('Override this method to implement synthesis')

    @timer
    def timed_synthesis(self):
        try:
            return self.synthesis(self.unitary)
        except Exception as e:
            print(f'{self.get_synthesiser_name()}: Synthesis Error!')
            traceback.print_exc()

    def get_synthesis_metrics(self):
        if not self.metrics:
            execution_time, qc, cpu_time = self.timed_synthesis()

            if qc:
                '''大的电路这些时间会很慢'''
                self.metrics = {
                    'Synthesiser': self.get_synthesiser_name(),
                    'Circuit': qc.qasm(),
                    'Execution Time': execution_time,
                    'CNOT Count': get_cnot_cnt(qc),
                    'Depth': get_depth(qc),
                    'CPU time': cpu_time,
                    'gate_num': len(qc),
                }
        return self.metrics


if __name__ == '__main__':
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.z(2)
    qc.cy(1, 2)
    qc.cx(0, 1)
    qc.h(2)
    qc.cy(0, 1)
    print(qc)

    # print(f'Parallelism: {cal_circuit_parallelism(qc)}')
