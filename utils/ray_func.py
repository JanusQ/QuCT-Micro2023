import ray
import threading
import inspect
import uuid
from random import random
import concurrent.futures
from concurrent.futures._base import Future
from collections import Iterable
from inspect import isgeneratorfunction
from collections import defaultdict

# 是不是需要远程执行的函数
def is_ray_func(func):
    for name, f in inspect.getmembers(func, lambda f: hasattr(f, '_remote')):
        return True
    return False

def is_ray_future(obj):
    return isinstance(obj, ray._raylet.ObjectRef)

def wait(future):
    # TODO: 可能会导致循环递归
    if isinstance(future, (list, set)):
        futures = future
        return [wait(future) for future in futures]
    if is_ray_future(future):
        return ray.get(future)
    elif isinstance(future, Future):
        return future.result()
    elif isinstance(future, (dict, defaultdict)):
        return {
            key: wait(item)
            for key, item in future.items()
        }
    else:
        # raise Exception(future, 'is not future type')
        return future

# def wait(future, wait_list = None):
#     # TODO: 不知道这样能不能避免递归
#     if wait_list is None:
#         wait_list = {}
#     elif str(future) in wait_list:
#         return wait_list[str(future)]

#     if isinstance(future, (list, set)):
#         futures = future
#         result = [wait(future, wait_list) for future in futures]
#     elif is_ray_future(future):
#         result = ray.get(future)
#     elif isinstance(future, Future):
#         result = future.result()
#     elif isinstance(future, (dict, defaultdict)):
#         result = {
#             key: wait(item, wait_list)
#             for key, item in future.items()
#         }
#     else:
#         # raise Exception(future, 'is not future type')
#         result =  future
    
#     wait_list[str(future)] = result
#     return result
    