import time
import ray
from ray.util.multiprocessing import Pool

ray.init()
# @ray.remote
def fun(index):
    for _ in range(5):
        print(index)
        time.sleep(1)
   
    
if __name__ == '__main__':
    start = time.time()
    futures =[]
    for i in range(6):
        futures.append(fun.remote(i))
    
    for future in futures:
        ray.get(future)
    print(time.time() - start)