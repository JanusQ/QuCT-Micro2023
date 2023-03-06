import numpy as np
def get_duration2circuit_infos(durations,step,max_duration):
    durations = np.array(durations)
    if max_duration == 0:
        max_duration = durations.max()
    
    left, right = 0,step
    duration2circuit_index = []
    duration_X = []
    while right <= max_duration:
        duration_index = np.where( (durations>left)&(durations<=right))
        duration2circuit_index.append(duration_index)
        duration_X.append((left+right)/2)
        left+= step
        right += step
        

    return duration_X, duration2circuit_index

def plot_duration_fidelity(fig, axes,dataset,step,max_duration):
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