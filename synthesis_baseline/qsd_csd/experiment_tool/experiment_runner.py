from experiment_tool.adapters import get_synthesise_adapter
from typing import List
from qiskit import QuantumCircuit
import numpy as np
import json
import pickle


class ExperimentRunner:
    def __init__(self, name: str, unitary: np.ndarray, synthesiser_list: List[str]):
        self.name = name
        self.unitary = unitary
        self.num_qubits = int(np.log2(len(unitary)))

        cls_list = [get_synthesise_adapter(cls_name) for cls_name in synthesiser_list]
        self.synthesisers = [cls(unitary) for cls in cls_list]

    def _write_results(self, metrics: List[dict]):
        # file_path = f'../experiment_results/{self.name}.json'
        file_path = f'/home/olcus/synthesis/quantum-circuit-synthesis/experiment_results/{self.name}.json'
        with open(file_path, mode='w') as f:
            serialized_unitary = json.dumps(pickle.dumps(self.unitary).decode('latin-1'))
            result_dict = {
                'Experiment Name': self.name,
                'Unitary': serialized_unitary,
                'Num of Qubits': self.num_qubits,
                'Metrics': metrics
            }
            json.dump(result_dict, f, indent=2)

    def run(self):
        print(f'Experiment {self.name} Start...')
        metrics = []
        for s in self.synthesisers:
            print(f'Start synthesising: {s.get_synthesiser_name()}')
            m = s.get_synthesis_metrics()
            metrics.append(m)
        self._write_results(metrics)
        print('Finish!')
        print()

    @staticmethod
    def print_result_file(exp_name: str):
        # file_path = f'../experiment_results/{exp_name}.json'
        file_path = f'/home/olcus/synthesis/quantum-circuit-synthesis/experiment_results_old/{exp_name}.json'
        with open(file_path, mode='r') as f:
            result_dict = json.load(f)

            # Deserialize fields
            picked_unitary = json.loads(result_dict['Unitary']).encode('latin-1')
            unitary = pickle.loads(picked_unitary)

            # Print Result
            print(f"Experiment Name: {result_dict['Experiment Name']}")
            print()
            print("Unitary:")
            print(np.round(unitary, 2))
            print()
            print(f"Num of Qubits: {result_dict['Num of Qubits']}")
            print()
            
            for metrics in result_dict['Metrics']:
                print('============================')
                print(f"Synthesiser: {metrics['Synthesiser']}")
                print('Circuit:')
                print(QuantumCircuit.from_qasm_str(metrics["Circuit"]))
                print(f"Execution time: {metrics['Execution Time']}s")
                print(f"CNOT Count: {metrics['CNOT Count']}")
                print(f"Depth: {metrics['Depth']}")
                print(f"Parallelism: {metrics['Parallelism']}")
                print('============================')
                print('')
        return result_dict
