import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.getcwd())

data_path = os.path.join(os.path.dirname(__file__), 'data', 'traffic', 'PEMS07')
label_path = os.path.join(data_path, 'DAY', 'label.npy')
adjacency_path = os.path.join(data_path, 'adjacency_matrix.npy')
signal_path = os.path.join(data_path, 'graph_signals.npy')

print(data_path)
print(label_path)
print(adjacency_path)
print(signal_path)


class GraphScattering:
    def __init__(self, label_path, adjacency_path, signal_path):
        self.label_path = label_path
        self.adjacency_path = adjacency_path
        self.signal_path = signal_path

    def relu(self, x):
        return np.maximum(0, x)

    def load_data(self):
        self.labels = np.load(self.label_path) if self.label_path is not None else None
        self.A = np.load(self.adjacency_path) if self.adjacency_path is not None else None
        self.X = np.load(self.signal_path) if self.signal_path is not None else None

        if self.A is not None:
            print(f"Loaded adjacency matrix A with shape: {self.A.shape}")
        if self.X is not None:
            print(f"Loaded graph signals X with shape: {self.X.shape}")

        return self.A, self.X

    def get_P(self, A: np.ndarray) -> np.ndarray:
        d_arr = np.sum(A, axis=0)
        d_arr_inv = np.divide(1.0, d_arr, out=np.zeros_like(d_arr, dtype=float), where=d_arr != 0)
        D_inv = np.diag(d_arr_inv)
        P = 0.5 * (np.eye(A.shape[0]) + A @ D_inv)
        return P

    def compute_W_2_transform(self, A, X, largest_scale, low_pass_as_wavelet=True):
        X = np.asarray(X, dtype=float)
        print(f"this is the shape of X before conversion to array: {X.shape}")
        A = np.asarray(A, dtype=float)

        # if X.ndim == 2:
        #     X = X[:, :, None]

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")

        N = X.shape[0]            # number of samples
        p = X.shape[1]            # number of nodes
        print(f"this is p: {p}")
        print(f"this is N: {N}")

        # if N != A.shape[0]:
        #     raise ValueError(f"Node mismatch: X has {N} nodes but A is {A.shape[0]}x{A.shape[1]}")

        # X = X.reshape(N,p)
        # print(f"this is the shape of X after reshape: {X.shape}")

        P = self.get_P(A)
        print(f"this is the shape of P: {P.shape}")

        I = np.eye(N)
        print(f"this is the shape of I: {I.shape}")
        #print(f"this is the shape of X before applying (I - P) @ X: {X.shape}")

        coeffs = []
        C0=(I-P) @ X
        #C0 = np.einsum('ab,bi->ai', (I - P), X)
        print(f"this is the shape of C0: {C0.shape}")
        coeffs.append(C0)

        a = C0.copy()

        for j in range(1, largest_scale + 1):
            pow_val = 2 ** (j - 1)

            prev = a.copy()
            for _ in range(pow_val):
                prev=P@prev
                #prev = np.einsum('ab,bi->ai', P, prev)

            curr = prev.copy()
            for _ in range(pow_val):
                curr= P @ curr
                #curr = np.einsum('ab,bi->ai', P, curr)

            a = prev + curr
            print(f"this is the shape of a at scale {j}: {a.shape}")
            coeffs.append(a)
            print(f'this is the length of coeffs: {len(coeffs)}')

        if low_pass_as_wavelet:
            low_pass = a.copy()
            for _ in range(2 ** (largest_scale - 1)):
                low_pass = P @ low_pass
                #low_pass = np.einsum('ab,bi->ai', P, low_pass)
                print(f'this is the shape of low_pass: {low_pass.shape}')
            coeffs.append(low_pass)

        out = np.concatenate(coeffs, axis=1)
        return out

    def blis_action(self, X):
        X1 = self.relu(X)
        X2 = self.relu(-X)
        return np.concatenate([X1, X2], axis=1)

    def blis_coeff(self, A, X, J, L, low_pass_as_wavelet=True):
        B = X.copy()
        for l in range(L):
            B = self.compute_W_2_transform(A, B, largest_scale=J, low_pass_as_wavelet=low_pass_as_wavelet)
            print(f"this is the shape of B after layer {l+1}: {B.shape}")
            B = self.blis_action(B)
        return B


A = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
], dtype=float)

X = np.array([
    [1, 2],
    [3, 4],
    [4, 3],
    [2, 1]
], dtype=float)

gs = GraphScattering(None, None, None)

W = gs.compute_W_2_transform(A, X, largest_scale=2, low_pass_as_wavelet=True)
print('-'*40)
#print(W)
print(f"this is the shape of W: {W.shape}")

B = gs.blis_coeff(A, X, J=2, L=3, low_pass_as_wavelet=True)
print(f'this is the shape of B: {B.shape}')

gs2 = GraphScattering(label_path, adjacency_path, signal_path)
A, X = gs2.load_data()
X=X[:,:,-1].T
W = gs2.compute_W_2_transform(A, X, largest_scale=4)
B = gs2.blis_coeff(A, X, J=2, L=2)

print(f'this is the shape of W: {W.shape}')
print(f'this is the shape of B: {B.shape}')