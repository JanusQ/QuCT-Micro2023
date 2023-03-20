import numpy as np
def get_duration2circuit_infos(durations,step,max_duration):
    durations = np.array(durations)
    if max_duration == 0:
        max_duration = durations.max()
    else:
        max_duration = max_duration if max_duration <= durations.max() else durations.max()
    left, right = durations.min(), durations.min() + step
    duration2circuit_index = []
    duration_X = []
    while right <= max_duration:
        duration_index = np.where( (durations>left)&(durations<=right))[0]
        left+= step
        right += step
        if len(duration_index) == 0:
            continue
        duration2circuit_index.append(duration_index)
        duration_X.append((left+right)/2)
        
        

    return duration_X, duration2circuit_index

def plot_duration_fidelity(fig, axes,dataset,step = 100 ,max_duration =0):
    predicts,reals, durations = [],[],[]
    for cir in dataset:
        predicts.append(cir['circuit_predict'])
        reals.append(cir['ground_truth_fidelity'])
        durations.append(cir['duration'])
        
    durations = np.array(durations)    
    reals = np.array(reals)    
    predicts = np.array(predicts)
    duration_X, duration2circuit_index = get_duration2circuit_infos(durations,step,max_duration)
    
    real_y,predict_y = [],[]
    for circuit_index in duration2circuit_index:
        real_y.append(reals[circuit_index].mean())
        predict_y.append(predicts[circuit_index].mean())
        
    axes.plot(duration_X, real_y ,markersize = 12,linewidth = 2, label='real',marker = '^' )
    axes.plot(duration_X, predict_y ,markersize = 12,linewidth = 2, label='predict',marker = '^' )
    axes.set_xlabel('duration ')
    axes.set_ylabel('fidelity')
    axes.legend() # 添加图例
    fig.show()
    return  duration_X, duration2circuit_index


def plot_top_ratio(upstream_model, erroneous_pattern_weight):
    x ,y = [],[]
    for top in range(1,100,1):
        top /= 100
        total_find = 0
        for device, pattern_weights in erroneous_pattern_weight.items():
            path_table_size = len(upstream_model.device2path_table[device].keys())
            for pattern_weight in pattern_weights:
                if  pattern_weight[1] < top * path_table_size:
                    total_find += 1

        find_ratio = total_find / (len(upstream_model.erroneous_pattern.keys()) * upstream_model.error_pattern_num_per_device)
        print(top,find_ratio)
        x.append(top)
        y.append(find_ratio)
        
        
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=(20,6)) # 创建一个图形对象和一个子图对象
    axes.plot(x, y ,markersize = 12,linewidth = 2, label='ratio',marker = '^' )
    axes.set_xlabel('top ')
    axes.set_ylabel('find_ratio')
    axes.legend() # 添加图例
    fig.show()
    num_qubits = upstream_model.dataset[0]['num_qubits']
    fig.savefig(f"find_ratio_{num_qubits}.svg")

def find_error_path(upstream_model, error_params):
    error_params = np.array(error_params)
    erroneous_pattern = upstream_model.erroneous_pattern
    
    device_index2device = {} #两比特门与但单比特门映射为一维下标
    for device  in upstream_model.device2path_table.keys():
        device_index = list(upstream_model.device2path_table.keys()).index(device)
        device_index2device[device_index] = device
        
    error_params_path_weight = {} #训练好的参数对应的path及其权重
    error_params_path = {}
    for idx, device_error_param in enumerate(error_params):
        device = device_index2device[idx]
        sort = np.argsort(device_error_param)
        sort = sort[::-1]
        device_error_params_path_weight = []
        device_error_params_path = []
        for i in sort:
            if int(i) in upstream_model.device2reverse_path_table[device].keys():
                path = upstream_model.device2reverse_path_table[device][int(i)]
                if isinstance(path,str):
                    device_error_params_path_weight.append((path,device_error_param[i]))
                    device_error_params_path.append(path)
        error_params_path_weight[device] = device_error_params_path_weight
        error_params_path[device] = device_error_params_path
        
    erroneous_pattern_weight = {} #手动添加的error_path在训练完参数中的排位
    for device, patterns in erroneous_pattern.items():
        device_error_params_path = error_params_path[device]
        device_erroneous_pattern_weight = []
        for pattern in patterns:
            if pattern in device_error_params_path:
                k = device_error_params_path.index(pattern)
                device_erroneous_pattern_weight.append((pattern,k))
        erroneous_pattern_weight[device] = device_erroneous_pattern_weight
        
    plot_top_ratio(upstream_model, erroneous_pattern_weight)
