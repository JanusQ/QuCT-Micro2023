import numpy.random

from circuit import gen_random_circuits, label_ground_truth_fidelity
from upstream import RandomwalkModel
from downstream import FidelityModel
from simulator import get_error_results

train_dataset = gen_random_circuits(5, 20, 60, 2, True, True)
upstream_model = RandomwalkModel(1, 15)
upstream_model.batch_train(train_dataset)
upstream_model.load_reduced_vecs()


get_error_results(train_dataset, upstream_model)
# label_ground_truth_fidelity(train_dataset,numpy.random.rand(64))
downstream_model = FidelityModel()
downstream_model.train(train_dataset)

test_dataset = gen_random_circuits(5, 20, 60, 1, True, True)
for cir in test_dataset:
    cir = upstream_model.vectorize(cir)
    predict, circuit_info, gate_errors = downstream_model.predict_fidelity(cir)
    print(predict)
