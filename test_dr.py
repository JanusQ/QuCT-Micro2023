import numpy as np
from scipy.stats import unitary_group
from sklearn.decomposition import IncrementalPCA as PCA

def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return np.abs(1 - np.abs(np.sum(np.multiply(A, np.conj(B)))) / A.shape[0])

class PCA():
    def __init__(self, X, k = None, reduced_prop = None) -> None:
        X = np.concatenate([m.reshape((-1, m.shape[-1])) for m in X], axis=0)

        # 对 X 做中心化处理
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # 计算 X 的协方差矩阵
        C = np.cov(X_centered.T)

        # 对 C 做特征值分解
        eigvals, eigvecs = np.linalg.eig(C)

        
        sorted_indices = np.argsort(eigvals)[::-1]
        sorted_eigen_values = eigvals[sorted_indices]
        

        sum_eigen_values = np.sum(sorted_eigen_values)
        if reduced_prop is not None:
            k = 0
            target_eigen_values = sum_eigen_values * reduced_prop
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values:
                accumulated_eigen_value += eigen_value
                k = k + 1
                if accumulated_eigen_value > target_eigen_values:
                    break
        if k is not None:
            accumulated_eigen_value = 0
            for eigen_value in sorted_eigen_values[:k]:
                accumulated_eigen_value += eigen_value
            reduced_prop = accumulated_eigen_value/sum_eigen_values
            print('reduced_prop = ' , reduced_prop)

    
        # 取前 k 个最大的特征值对应的特征向量
        
        self.k = k
        
        self.V_k = eigvecs[:, sorted_indices[:k]]
        self.eigvecs = eigvecs
        self.sorted_indices = sorted_indices[:k]
        self.X_mean = X_mean
        pass

    def transform(self, X) -> np.array:
        V_k = self.V_k

        # 对每个输入矩阵都做降维，并且保持距离相似性
        reduced_matrices = []
        for m in X:
            m_centered = m - self.X_mean[np.newaxis, :]
            m_reduced = np.dot(m_centered, V_k)
            # 对降维后的矩阵做幺正正交化，保证在降维前后距离有相似性
            q, r = np.linalg.qr(m_reduced)
            reduced_matrices.append(q)
        
        return reduced_matrices



def unitary_dim_reduction(*matrices, k = None, reduced_prop = None):
    # Input:
    # - *matrices: 可变数量的酉矩阵
    # - k: 降维后的维度
    #
    # Output:
    # - reduced_matrices: 降维后的矩阵列表

    # 将所有输入矩阵的列向量拼接在一起，得到数据矩阵 X
    X = np.concatenate([m.reshape((-1, m.shape[-1])) for m in matrices], axis=0)

    # 对 X 做中心化处理
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # 计算 X 的协方差矩阵
    C = np.cov(X_centered.T)

    # 对 C 做特征值分解
    eigvals, eigvecs = np.linalg.eig(C)

    
    sorted_indices = np.argsort(eigvals)[::-1]
    sorted_eigen_values = eigvals[sorted_indices]
    

    sum_eigen_values = np.sum(sorted_eigen_values)
    if reduced_prop is not None:
        k = 0
        target_eigen_values = sum_eigen_values * reduced_prop
        accumulated_eigen_value = 0
        for eigen_value in sorted_eigen_values:
            accumulated_eigen_value += eigen_value
            k = k + 1
            if accumulated_eigen_value > target_eigen_values:
                break
    if k is not None:
        accumulated_eigen_value = 0
        for eigen_value in sorted_eigen_values[:k]:
            accumulated_eigen_value += eigen_value
        print('reduced_prop = ' ,accumulated_eigen_value/sum_eigen_values)

 
    # 取前 k 个最大的特征值对应的特征向量
    
    V_k = eigvecs[:, sorted_indices[:k]]

    # 对每个输入矩阵都做降维，并且保持距离相似性
    reduced_matrices = []
    for m in matrices:
        m_centered = m - X_mean[np.newaxis, :]
        m_reduced = np.dot(m_centered, V_k)
        # 对降维后的矩阵做幺正正交化，保证在降维前后距离有相似性
        q, r = np.linalg.qr(m_reduced)
        reduced_matrices.append(q)

    return reduced_matrices

# 输入多个酉矩阵
n_qubits = 1
m1 =  unitary_group.rvs(2**n_qubits) # np.array([[0.6+0.4j, 0.0+0.8j], [0.0+0.8j, 0.6-0.4j]])

assert np.allclose(m1 @ m1.T.conj(), np.eye(2))

m2 = unitary_group.rvs(2**n_qubits) #np.array([[0.5-0.5j, 0.5+0.5j], [0.5+0.5j, 0.5-0.5j]])

# 降维后的维度

ms = np.array([m1, m2])

# pca = PCA(n_components=1, batch_size=1)
# pca.fit(ms)
# print(pca.transform([m1]))

pca = PCA(ms, k = 1)
print(pca.transform(ms))

# 对输入的矩阵做降维，保证在降维前后距离有相似性
reduced_ms = unitary_dim_reduction(m1, m2, k = 1)

print(reduced_ms)