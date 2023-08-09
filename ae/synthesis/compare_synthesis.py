
from ae.synthesis.load_syn_result import load_baseline, load_baseline_Unitary, load_quct
import  matplotlib.pyplot as plt
import numpy as np

gate_nums, times = [], []
for n_qubits in range(3,6):
    type = 'random'

    gate_num, cnot_num, depths, time= load_baseline(n_qubits, type, 'CSD Synthesiser')
    gate_nums.append(gate_num)
    times.append(time)
    gate_num, cnot_num, depths, time = load_quct(n_qubits, type, )
    gate_nums.append(gate_num)
    times.append(time)
    gate_num, cnot_num, depth, time= load_baseline(n_qubits, type, 'QFast Synthesiser')
    gate_nums.append(gate_num)
    times.append(time)



gate_nums = np.array(gate_nums).reshape((-1,3)).T
times = np.array(times).reshape((-1,3)).T

fig, ax = plt.subplots(figsize= (12,10))
x =  np.arange(3,6) * 10
ax.bar(x-2, gate_nums[0], width = 2, label = 'ccd')
ax.bar(x, gate_nums[1], width = 2, label = 'quct')
ax.bar(x+2, gate_nums[2], width = 2, label = 'qfast')
ax.legend()
ax.set_xlabel('qubit')
ax.set_ylabel('gate')
ax.set_xticks(x)
ax.set_xticklabels(np.arange(3,6))

ax2 = ax.twinx()
ax2.plot(x, times[0], linewidth = 4, marker = '^', markersize = 12, label = 'ccd')
ax2.plot(x, times[1], linewidth = 4, marker = '^', markersize = 12, label = 'quct')
ax2.plot(x, times[2], linewidth = 4, marker = '^', markersize = 12, label = 'qfast')
ax2.set_ylabel('time')
fig.savefig('ae/synthesis/compare_synthesis.svg')

