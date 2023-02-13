from quct import generate_circtuit, predict_fidelity


test_dataset = generate_circtuit(5,20,400,1,True,True)
veceds  = [] 
for cir in test_dataset:
    veced, predict, gate_errors = predict_fidelity(cir['qiskit_circuit'])
    veceds.append(veced)
    print(veced['predict'],cir['xeb_fidelity'],gate_errors)
    
