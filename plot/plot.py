import numpy as np
def get_duration2circuit_infos(duration,step,max_duration = 0):
    duration = np.array(duration)
    if max_duration == 0:
        max_duration = duration.max()
    
    left, right = 0,step
    duration2circuit_index = []
    duration_X = []
    while right <= max_duration:
        duration_index = np.where( (duration>left)&(duration<=right))
        duration2circuit_index.append(duration_index)
        duration_X.append((left+right)/2)
        left+= step
        right += step
        

    return duration_X, duration2circuit_index

