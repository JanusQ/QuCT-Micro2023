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


def get_prop2circuit_infos(props,step = 0.1,max_props = 0):
    props = np.array(props)
    if max_props == 0:
        max_props = props.max()
    else:
        max_props = max_props if max_props <= props.max() else props.max()
    left, right = props.min(), props.min() + step
    prop2circuit_index = []
    prop_X = []
    while right <= max_props:
        prop_index = np.where( (props>left)&(props<=right))[0]
        left+= step
        right += step
        if len(prop_index) == 0:
            continue
        prop2circuit_index.append(prop_index)
        prop_X.append((left+right)/2)
        
    return prop_X, prop2circuit_index

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
    
    
    axes.scatter(duration_X, real_y)
    axes.plot(duration_X, real_y ,markersize = 12,linewidth = 2, label='real',marker = '^' )
    axes.plot(duration_X, predict_y ,markersize = 12,linewidth = 2, label='predict',marker = '^' )
    axes.set_xlabel('duration ')
    axes.set_ylabel('fidelity')
    axes.legend() # 添加图例
    fig.show()
    return  duration_X, duration2circuit_index


def plot_real_predicted_fidelity(fig, axes, dataset,step = 100 ,max_duration =0):
    predicts,reals, durations = [],[],[]
    for cir in dataset:
        predicts.append(cir['circuit_predict'])
        reals.append(cir['ground_truth_fidelity'])
        durations.append(cir['duration'])
        
    axes.axis([0, 1, 0, 1])
    axes.scatter(reals, predicts)
    axes.set_xlabel('real ')
    axes.set_ylabel('predict')
    fig.show()
    return fig


def plot_top_ratio(upstream_model, erroneous_pattern_weight, name):
    x ,y = [],[]
    
    erroneous_pattern_num = 0
    for device, erroneous_patterns in upstream_model.erroneous_pattern.items():
        erroneous_pattern_num += len(erroneous_patterns)
        
    total_path_num = sum([len(upstream_model.device2path_table[device]) for device in upstream_model.device2path_table])
        
    for top in range(1, 100,1):
        top /= 100
        total_find = 0
        for device, pattern_weights in erroneous_pattern_weight.items():
            path_table_size = len(upstream_model.device2path_table[device].keys())
            for pattern_weight in pattern_weights:
                if  pattern_weight[1] < top * path_table_size:
                    total_find += 1

        # find_ratio = total_find / erroneous_pattern_num
        # find_path = total_find
        # print(top, find_ratio)
        
        # x.append(top)
        # y.append(find_ratio)
        
        x.append(total_path_num * top)
        y.append(total_find)
        
        
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=(20,6)) # 创建一个图形对象和一个子图对象
    axes.plot(x, y ,markersize = 12,linewidth = 2, label='ratio',marker = '^' )
    axes.set_xlabel('top')
    axes.set_ylabel('#find_path')
    # axes.set_ylabel('find_ratio')
    axes.legend() # 添加图例
    fig.show()
    num_qubits = upstream_model.backend.n_qubits
    
    if name is None:
        fig.savefig(f"find_ratio_{num_qubits}.svg")
    else:
        fig.savefig(name)
    return fig

def find_error_path(upstream_model, error_params, name = None):
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
        
    plot_top_ratio(upstream_model, erroneous_pattern_weight, name)

import pandas as pd
import seaborn as sns
def plot_correlation(data, feature_names, color_features = None, name = 'correlation'):
    df = pd.DataFrame(data, columns=feature_names)
    
    if color_features is not None:
        df["class"] = pd.Series(color_features)
        sns_plot = sns.pairplot(df, hue='class', palette="tab10")
    else:
        sns_plot = sns.pairplot(df, hue=None, palette="tab10")
    # sns_plot = sns.pairplot(df, hue=color_feature, palette="tab10")
    fig = sns_plot.fig
    fig.savefig(name)
    return