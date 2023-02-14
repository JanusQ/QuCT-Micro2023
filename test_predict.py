from circuit import gen_random_circuits
from upstream import  RandomwalkModel
from downstream import FidelityModel

train_dataset = gen_random_circuits(5,20,200,10,True,True)
upstream_model = RandomwalkModel(1,15)
upstream_model.batch_train(train_dataset)
upstream_model.load_reduced_vecs()

downstream_model =FidelityModel(upstream_model)
downstream_model.train()

veceds = []
test_dataset = gen_random_circuits(5,20,200,1,True,True)
for cir in test_dataset:
    predict, veced, gate_errors = downstream_model.predict_fidelity(cir['qiskit_circuit'])
    veceds.append(veced)
    print(predict,cir['xeb_fidelity'],gate_errors)
    
