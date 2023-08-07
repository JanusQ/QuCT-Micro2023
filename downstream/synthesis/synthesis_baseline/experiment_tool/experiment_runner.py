import json
import os
import pickle
from typing import List

import numpy as np
from experiment_tool.adapters import get_synthesise_adapter


def check_exist(path):
    if os.path.exists(path):
        path
class ExperimentRunner:
    def __init__(self, name: str, unitary: np.ndarray, synthesiser_list: List[str], type: str):
        self.name = name
        self.unitary = unitary
        self.num_qubits = int(np.log2(len(unitary)))
        cls_list = [get_synthesise_adapter(cls_name) for cls_name in synthesiser_list]
        self.synthesisers = [cls(unitary) for cls in cls_list]
        self.type = type

    def _get_write_path(self):
        dir_path = f'ae/synthesis/baseline/{self.num_qubits}/{self.type}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, f'{self.name}.json')
        cnt = 1
        while os.path.exists(file_path):
            file_path = os.path.join(dir_path, f'{self.name}_{cnt}.json')
            cnt += 1
        return file_path

    def _write_results(self, metrics: List[dict]):
        file_path = self._get_write_path()
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

