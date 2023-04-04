import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from utils.backend import gen_grid_topology, get_grid_neighbor_info, Backend, gen_linear_topology, get_linear_neighbor_info, gen_fulllyconnected_topology
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from downstream.synthesis.synthesis_model_pca_unitary_jax_可以跑但是最后会一直插一个 import SynthesisModelRandom, find_parmas, pkl_dump, pkl_load, matrix_distance_squared, SynthesisModel, synthesize
import jax
from scipy.stats import unitary_group

# U = unitary_group.rvs(2**3)
# print(matrix_distance_squared(U, U))

n_qubits = 3
topology = gen_fulllyconnected_topology(n_qubits)
neigh_info = gen_fulllyconnected_topology(n_qubits)

synthesis_data_path = f'./temp_data/{n_qubits}_synthesis_data.pkl'
with open(synthesis_data_path, 'rb') as f:
    Us, Vs = pickle.load(f)
    
print(matrix_distance_squared(Us[0], Us[0]))
# X, y = make_multilabel_classification(n_classes=3, random_state=0)

_Us = []
for U in Us:
    U = U.reshape(U.size)
    U = np.concatenate([U.real, U.imag])
    _Us.append(U)

Us = np.array(_Us, dtype= np.float64)
# Us.reshape((len(Us), Us[0].size))
# Us += 1


# @jax.jit
def to_unitary(x):
    x_real = x[:len(x)//2]
    x_imag = x[len(x)//2:]
    return x_real.reshape((2**n_qubits, 2**n_qubits)) + 1j * x_imag.reshape((2**n_qubits, 2**n_qubits))
    

def _matrix_distance_squared(x1, x2):
    x1 = to_unitary(x1) #x1 .reshape((2**n_qubits, 2**n_qubits))
    x2 = to_unitary(x2) #x2.reshape((2**n_qubits, 2**n_qubits))
    return matrix_distance_squared(x1, x2)

print(_matrix_distance_squared(Us[0], Us[0]))

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric= _matrix_distance_squared, n_jobs = -1).fit(Us)


n_test = 5

distances, indices = nbrs.kneighbors(Us[-n_test:])
for test_i, v in enumerate(Vs[-n_test:]):
    print(v, ':')
    for dist, index in zip(distances[test_i], indices[test_i]):
        print(Vs[index], dist)

# clf = MultiOutputClassifier(MultinomialNB()).fit(Us, Vs) # n_jobs=6
# clf = RandomForestClassifier(max_depth=5, random_state=0, n_jobs = 6, verbose= 1).fit(Us, Vs)
# predict = clf.predict(Us[-n_test:])
# for predict, real in zip(predict, Vs[-n_test:]):
#     print(predict, real)
# print(Vs[-2:])

print()