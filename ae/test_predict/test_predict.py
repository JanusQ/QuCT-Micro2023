import  pickle
import numpy as np
import random
with open('dataset_ibm.pkl', 'rb')as f:
    train_dataset, test_dataset = pickle.load(f)

from utils.backend import Backend
# coupling_map = [[4, 3], [3, 4], [2, 3], [3, 2], [1, 2], [2, 1], [0, 1], [1, 0]]
coupling_map = [[3, 4], [2, 3], [1, 2] ,[0, 1]]
topology = {0: [1], 1: [0, 1], 2: [1,3], 3: [2,4], 4: [3]}
backend = Backend(n_qubits=5, topology=topology, neighbor_info=None, coupling_map=coupling_map,
basis_single_gates = ['id', 'rz', 'sx', 'x'], basis_two_gates = ['cx'], divide = False, decoupling = False)

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random
def plot_scaater(reals, predicts, durations ,name):
    par = np.polyfit(reals, predicts, 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    x1 = [0.4, 1.0]
    y1 = [slope*xx + intercept  for xx in x1]
    #定义颜色
    colors = ["#FF3636", '#277C8E' ,"#1F77B4"]
    '''xia <- shang'''
    # colors.reverse()
    # colors = np.array(colors) / 256
    # 定义颜色的位置
    pos = [0, .5, 1]
    # 创建colormap对象
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(pos, colors)))

    normalied_durations = (durations - durations.min())/(durations.max() - durations.min())

    # cmap_name = 'Blues'
    # cmap_name = 'viridis'
    # cmap_name = 'plasma'
    # cmap_name = 'winter'

    random_index = list(range(len(reals)))
    random.shuffle(random_index)
    random_index = random_index[:1500]
    reals = np.array(reals)
    predicts = np.array(predicts)
    fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象
    axes.axis([0, 1, 0, 1])
    axes.scatter(reals[random_index], predicts[random_index], c= normalied_durations[random_index], cmap=cmap,alpha = 0.6, s=80 )
    axes.plot(x1,y1)
    axes.set_xlim(.2, 1)
    axes.set_ylim(.2, 1)
    axes.set_xlabel('real ')
    axes.set_ylabel('predict')
    axes.plot([[0,0],[1,1]])
    # fig.colorbar(cm.ScalarMappable( cmap=cmap))
    fig.savefig(name)
    print(slope, intercept)


from upstream import RandomwalkModel
delta_steps = []
for n_steps in range(4):
    upstream_model = RandomwalkModel(n_steps, 20, backend=backend, travel_directions=('parallel', 'former'))
    upstream_model.train(train_dataset+test_dataset, multi_process=True, remove_redundancy=n_steps > 1)

    from downstream import FidelityModel

    downstream_model = FidelityModel(upstream_model)
    downstream_model.train(train_dataset, epoch_num = 300)

    test_predicts, test_reals, durations = [] , [], []
    for circuit_info in test_dataset:

        predict = downstream_model.predict_fidelity(circuit_info)
        circuit_info['predict'] = predict
        durations.append(circuit_info['duration'])
        test_reals.append(circuit_info['ground_truth_fidelity'])
        test_predicts.append(predict)
    
    test_reals = np.array(test_reals)
    test_predicts = np.array(test_predicts)
    durations = np.array(durations)
    plot_scaater(test_reals, test_predicts, durations, name = f'scatter_step{n_steps}.png')
    print('average inaccuracy = ', np.abs(test_predicts - test_reals).mean())
    delta_steps.append(np.abs(np.array(test_reals)-np.array(test_predicts)).tolist())

pickle.dump(delta_steps,open('delta_steps.pkl','wb'))