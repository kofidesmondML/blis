import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.getcwd())

data_path = os.path.join(os.path.dirname(__file__), 'data', 'traffic', 'PEMS08')
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

    def reverse_relu(self, x):
        return np.maximum(0, -x)

    def load_data(self):
        self.labels = np.load(self.label_path) if self.label_path is not None else None
        self.A = np.load(self.adjacency_path) if self.adjacency_path is not None else None
        self.X = np.load(self.signal_path) if self.signal_path is not None else None

        if self.A is not None:
            print(f"Loaded adjacency matrix A with shape: {self.A.shape}")
        if self.X is not None:
            print(f"Loaded graph signals X with shape: {self.X.shape}")
        if self.labels is not None:
            print(f"Loaded labels with shape: {self.labels.shape}")

        return self.A, self.X, self.labels

    def get_P(self, A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")

        d_arr = np.sum(A, axis=0)
        d_arr_inv = np.divide(
            1.0,
            d_arr,
            out=np.zeros_like(d_arr, dtype=float),
            where=d_arr != 0,
        )
        D_inv = np.diag(d_arr_inv)
        P = 0.5 * (np.eye(A.shape[0]) + A @ D_inv)
        return P

    def compute_W_2_transform(self, A, X, largest_scale, low_pass_as_wavelet=True):
        A = np.asarray(A, dtype=float)
        X = np.asarray(X, dtype=float)

        if largest_scale < 0:
            raise ValueError("largest_scale must be nonnegative.")

        if X.ndim == 1:
            X = X[:, None]

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")

        n_nodes = A.shape[0]

        if X.shape[0] != n_nodes:
            raise ValueError(
                f"Node mismatch: X has {X.shape[0]} rows, but A is {n_nodes}x{n_nodes}. "
                f"X must have shape (n_nodes, n_features)."
            )

        P = self.get_P(A)
        I = np.eye(n_nodes)

        coeffs = []

        C0 = (I - P) @ X
        coeffs.append(C0)

        a = X.copy()

        for j in range(1, largest_scale):
            pow_val = 2 ** (j - 1)

            prev = a.copy()
            for _ in range(pow_val):
                prev = P @ prev

            curr = prev.copy()
            for _ in range(pow_val):
                curr = P @ curr

            a_j = prev - curr
            coeffs.append(a_j)

        if low_pass_as_wavelet:
            if largest_scale == 0:
                low_pass_steps = 0
            else:
                low_pass_steps = 2 ** (largest_scale - 1)

            low_pass = X.copy()
            for _ in range(low_pass_steps):
                low_pass = P @ low_pass

            coeffs.append(low_pass)

        out = np.concatenate(coeffs, axis=1)
        return out

    def blis_action(self, X):
        print(f"this is the shape of X before BLIS action: {X.shape}")
        X1 = self.relu(X)
        X2 = self.relu(-X)
        X_new = np.concatenate([X1, X2], axis=1)
        print(f"this is the shape of X_new after BLIS action: {X_new.shape}")
        return X_new

    def blis_coeff(self, A, X, J, L, low_pass_as_wavelet=True):
        B = np.asarray(X, dtype=float)

        if B.ndim == 1:
            B = B[:, None]

        for l in range(L):
            B = self.compute_W_2_transform(
                A,
                B,
                largest_scale=J,
                low_pass_as_wavelet=low_pass_as_wavelet,
            )
            print(f"this is the shape of B after W_2 layer {l + 1}: {B.shape}")

            B = self.blis_action(B)
            print(f"this is the shape of B after BLIS action {l + 1}: {B.shape}")

        return B

    def global_pool_l1(self, B):
        B = np.asarray(B, dtype=float)

        if B.ndim != 2:
            raise ValueError(f"B must have shape (n_nodes, H), got {B.shape}")

        return np.sum(np.abs(B), axis=0)

    def compute_moments(self, B, Q=4, flatten=True):
        B = np.asarray(B, dtype=float)

        if B.ndim != 2:
            raise ValueError(f"B must have shape (n_nodes, H), got {B.shape}")

        absB = np.abs(B)
        moments = [np.sum(absB ** q, axis=0) for q in range(1, Q + 1)]
        moments = np.stack(moments, axis=0)

        if flatten:
            return moments.reshape(-1)

        return moments

    def blis_global_features(self, A, X, J, L, Q=4, low_pass_as_wavelet=True, return_B=False):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[:, None]

        B = self.blis_coeff(
            A,
            X,
            J=J,
            L=L,
            low_pass_as_wavelet=low_pass_as_wavelet,
        )

        pooled_l1 = self.global_pool_l1(B)
        moment_matrix = self.compute_moments(B, Q=Q, flatten=False)
        moment_features = moment_matrix.reshape(-1)

        print(f"this is the shape of B: {B.shape}")
        print(f"this is the shape of pooled_l1: {pooled_l1.shape}")
        print(f"this is the shape of moment_matrix: {moment_matrix.shape}")
        print(f"this is the shape of moment_features: {moment_features.shape}")

        if return_B:
            return pooled_l1, moment_features, moment_matrix, B

        return pooled_l1, moment_features, moment_matrix

    def calculate_scattering(self, A, X, J, Q, layer_num=2):
        if J < 0:
            raise ValueError("J must be nonnegative.")

        if Q < 1:
            raise ValueError("Q must be at least 1.")

        if layer_num < 1:
            raise ValueError("layer_num must be at least 1.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X[:, None]

        n_nodes, T = X.shape

        num_wavelets = J + 2
        num_activations = 2
        num_paths = (num_wavelets * num_activations) ** layer_num

        B = np.zeros((T, num_paths, 1, Q), dtype=float)

        path_index = 0

        def build_layer(current_layer, current_signal):
            nonlocal path_index

            if current_layer == layer_num:
                for q in range(1, Q + 1):
                    moment = np.sum(np.abs(current_signal) ** q, axis=0)
                    B[:, path_index, 0, q - 1] = moment

                path_index += 1
                return

            x1 = self.compute_W_2_transform(
                A,
                current_signal,
                largest_scale=J + 1,
                low_pass_as_wavelet=True,
            )

            signal_width = current_signal.shape[1]
            S1 = x1.shape[1] // signal_width

            for j in range(S1):
                start = j * signal_width
                end = (j + 1) * signal_width

                wavelet_signal = x1[:, start:end]

                build_layer(current_layer + 1, self.relu(wavelet_signal))
                build_layer(current_layer + 1, self.reverse_relu(wavelet_signal))

        build_layer(0, X)

        return B


if __name__ == "__main__":
    gs2 = GraphScattering(label_path, adjacency_path, signal_path)
    A_real, X_real, labels_real = gs2.load_data()

    X_real = X_real[:, :, -1].T
    print(f"this is the shape of X_real after slicing/transposing: {X_real.shape}")

    B1 = gs2.calculate_scattering(A_real, X_real, J=4, Q=3, layer_num=1)
    print(f"this is the shape of B1/layer_1: {B1.shape}")

    B2 = gs2.calculate_scattering(A_real, X_real, J=4, Q=3, layer_num=2)
    print(f"this is the shape of B2/layer_2: {B2.shape}")