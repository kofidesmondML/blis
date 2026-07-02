import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from new_class import GraphScattering


def make_graph_scattering(device):
    try:
        gs = GraphScattering(device=device)
    except TypeError:
        gs = GraphScattering()

    return gs


def load_moment_features(
    data_dir,
    dataset,
    sub_dataset,
    wavelet_type,
    largest_scale,
    layer_list,
    moment_list,
):
    base_dir = os.path.join(
        data_dir,
        dataset,
        sub_dataset,
        "processed",
        "blis",
        wavelet_type,
        f"largest_scale_{largest_scale}",
    )

    features = []

    for layer in layer_list:
        for moment in moment_list:
            path = os.path.join(
                base_dir,
                f"layer_{layer}",
                f"moment_{moment}.npy",
            )

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing moment file: {path}")

            print(f"Loading: {path}")

            X = np.load(path)
            X = X.reshape(X.shape[0], -1)
            features.append(X)

    X_features = np.concatenate(features, axis=1)
    X_features = X_features.astype(np.float32)

    return X_features


def load_labels(data_dir, dataset, sub_dataset, task_type):
    label_path = os.path.join(
        data_dir,
        dataset,
        sub_dataset,
        task_type,
        "label.npy",
    )

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")

    y = np.load(label_path)

    classes = np.unique(y)
    class_to_id = {c: i for i, c in enumerate(classes)}

    y = np.array([class_to_id[c] for c in y], dtype=np.int64)

    return y


def split_like_blis(X_features, y, seed):
    all_idx = np.arange(len(X_features))

    train_idx, val_test_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=seed,
    )

    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=0.50,
        random_state=seed,
    )

    train_val_idx = np.concatenate([train_idx, val_idx], axis=0)

    X_train = X_features[train_val_idx]
    y_train = y[train_val_idx]

    X_test = X_features[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def torch_standardize(X_train, X_test, eps=1e-12):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)

    std = torch.where(
        std > eps,
        std,
        torch.ones_like(std),
    )

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def torch_pca(X_train, X_test, pca_variance):
    if pca_variance >= 1.0:
        return X_train, X_test, -1

    mean = X_train.mean(dim=0, keepdim=True)

    X_train_centered = X_train - mean
    X_test_centered = X_test - mean

    U, S, Vh = torch.linalg.svd(
        X_train_centered,
        full_matrices=False,
    )

    eigvals = S ** 2
    explained = eigvals / eigvals.sum()
    cumulative = torch.cumsum(explained, dim=0)

    num_components = int(torch.searchsorted(cumulative, pca_variance).item()) + 1

    V = Vh[:num_components].T

    X_train_pca = X_train_centered @ V
    X_test_pca = X_test_centered @ V

    print(f"PCA components: {num_components}")

    return X_train_pca, X_test_pca, num_components


class TorchLR(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class TorchLinearSVC(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class TorchMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def multiclass_hinge_loss(scores, y):
    row_idx = torch.arange(scores.shape[0], device=scores.device)

    correct_scores = scores[row_idx, y].view(-1, 1)

    margins = scores - correct_scores + 1.0
    margins[row_idx, y] = 0.0

    loss = torch.clamp(margins, min=0.0).mean()

    return loss


def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics


def train_torch_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs,
    batch_size,
    lr,
    weight_decay,
    device,
):
    input_dim = X_train.shape[1]
    num_classes = int(torch.max(y_train).item()) + 1

    if model_name == "LR":
        model = TorchLR(input_dim, num_classes).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

    elif model_name == "SVC":
        model = TorchLinearSVC(input_dim, num_classes).to(device)
        loss_fn = multiclass_hinge_loss

    elif model_name == "MLP":
        model = TorchMLP(input_dim, num_classes).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown torch model: {model_name}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    n_train = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train, device=device)

        model.train()
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            idx = perm[start:end]

            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()

            scores = model(xb)
            loss = loss_fn(scores, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch}, loss = {total_loss / n_train:.6f}")

    model.eval()

    with torch.no_grad():
        scores = model(X_test)
        y_pred = torch.argmax(scores, dim=1)

    y_true = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    return compute_metrics(y_true, y_pred)


def train_xgb_gpu(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier

    X_train_np = X_train.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()

    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
    )

    model.fit(X_train_np, y_train_np)

    y_pred = model.predict(X_test_np)

    return compute_metrics(y_test_np, y_pred)


def run_one_seed(
    X_features,
    y,
    model_name,
    seed,
    pca_variance,
    epochs,
    batch_size,
    lr,
    weight_decay,
    device,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    X_train_np, X_test_np, y_train_np, y_test_np = split_like_blis(
        X_features,
        y,
        seed,
    )

    X_train = torch.as_tensor(
        X_train_np,
        dtype=torch.float32,
        device=device,
    )

    X_test = torch.as_tensor(
        X_test_np,
        dtype=torch.float32,
        device=device,
    )

    y_train = torch.as_tensor(
        y_train_np,
        dtype=torch.long,
        device=device,
    )

    y_test = torch.as_tensor(
        y_test_np,
        dtype=torch.long,
        device=device,
    )

    X_train, X_test = torch_standardize(X_train, X_test)
    X_train, X_test, n_pca = torch_pca(X_train, X_test, pca_variance)

    if model_name == "XGB":
        metrics = train_xgb_gpu(
            X_train,
            y_train,
            X_test,
            y_test,
        )

    else:
        metrics = train_torch_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

    metrics["n_pca"] = n_pca

    return metrics


def summarize_results(rows, metric_names):
    df = pd.DataFrame(rows)

    summary_rows = []

    group_cols = [
        "dataset",
        "task",
        "wavelet_type",
        "largest_scale",
        "layers",
        "moments",
        "model",
    ]

    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))

        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)

        row["n_pca_mean"] = group["n_pca"].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="traffic")

    parser.add_argument(
        "--sub_datasets",
        nargs="+",
        type=str,
        default=["PEMS03", "PEMS04", "PEMS07", "PEMS08"],
    )

    parser.add_argument(
        "--task_types",
        nargs="+",
        type=str,
        default=["HOUR", "DAY", "WEEK"],
    )

    parser.add_argument("--wavelet_type", type=str, default="W2")
    parser.add_argument("--largest_scale", type=int, default=4)
    parser.add_argument("--highest_moment", type=int, default=3)

    parser.add_argument("--layer_list", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--moment_list", nargs="+", type=int, default=[1])

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["LR", "SVC", "MLP", "XGB"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 56],
    )

    parser.add_argument("--pca_variance", type=float, default=0.99)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--skip_moments", action="store_true")

    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    if args.sub_datasets == ["full"]:
        args.sub_datasets = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 90)
    print("BLIS W2 MOMENTS FROM new_class.py + TORCH CLASSIFICATION")
    print("=" * 90)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    gs = make_graph_scattering(device=device)

    all_rows = []

    for sub_dataset in args.sub_datasets:
        print("=" * 90)
        print(f"DATASET: {sub_dataset}")
        print("=" * 90)

        data_path = os.path.join(
            args.data_dir,
            args.dataset,
            sub_dataset,
        )

        adjacency_path = os.path.join(data_path, "adjacency_matrix.npy")
        signal_path = os.path.join(data_path, "graph_signals.npy")

        if not os.path.exists(adjacency_path):
            raise FileNotFoundError(f"Missing adjacency file: {adjacency_path}")

        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Missing graph signal file: {signal_path}")

        A = np.load(adjacency_path).astype(np.float32)
        X = np.load(signal_path).astype(np.float32)

        print(f"A shape: {A.shape}")
        print(f"X shape: {X.shape}")

        if not args.skip_moments:
            gs.save_scattering_moments(
                A=A,
                X=X,
                data_dir=args.data_dir,
                dataset=args.dataset,
                sub_dataset=sub_dataset,
                scattering_type="blis",
                wavelet_type=args.wavelet_type,
                largest_scale=args.largest_scale,
                highest_moment=args.highest_moment,
                layer_list=args.layer_list,
                force_recompute=args.force_recompute,
            )

        X_features = load_moment_features(
            data_dir=args.data_dir,
            dataset=args.dataset,
            sub_dataset=sub_dataset,
            wavelet_type=args.wavelet_type,
            largest_scale=args.largest_scale,
            layer_list=args.layer_list,
            moment_list=args.moment_list,
        )

        print(f"X_features shape: {X_features.shape}")

        for task_type in args.task_types:
            print("-" * 90)
            print(f"TASK: {task_type}")
            print("-" * 90)

            y = load_labels(
                data_dir=args.data_dir,
                dataset=args.dataset,
                sub_dataset=sub_dataset,
                task_type=task_type,
            )

            if len(y) != X_features.shape[0]:
                raise ValueError(
                    f"Label/features mismatch for {sub_dataset} {task_type}: "
                    f"len(y)={len(y)}, X_features.shape[0]={X_features.shape[0]}"
                )

            print(f"y shape: {y.shape}")
            print(f"classes: {np.unique(y)}")

            for model_name in args.models:
                print("-" * 90)
                print(f"MODEL: {model_name}")
                print("-" * 90)

                for seed in args.seeds:
                    print(f"Seed: {seed}")

                    metrics = run_one_seed(
                        X_features=X_features,
                        y=y,
                        model_name=model_name,
                        seed=seed,
                        pca_variance=args.pca_variance,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        device=device,
                    )

                    row = {
                        "dataset": sub_dataset,
                        "task": task_type,
                        "wavelet_type": args.wavelet_type,
                        "largest_scale": args.largest_scale,
                        "layers": " ".join(map(str, args.layer_list)),
                        "moments": " ".join(map(str, args.moment_list)),
                        "model": model_name,
                        "seed": seed,
                        "pca_variance": args.pca_variance,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "device": str(device),
                    }

                    row.update(metrics)

                    all_rows.append(row)

                    print(f"accuracy = {metrics['accuracy']:.6f}")
                    print(f"balanced_accuracy = {metrics['balanced_accuracy']:.6f}")
                    print(f"macro_f1 = {metrics['macro_f1']:.6f}")
                    print(f"weighted_f1 = {metrics['weighted_f1']:.6f}")
                    print(f"macro_precision = {metrics['macro_precision']:.6f}")
                    print(f"macro_recall = {metrics['macro_recall']:.6f}")
                    print(f"n_pca = {metrics['n_pca']}")

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]

    results_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(all_rows, metric_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub_dataset_name = "_".join(args.sub_datasets)
    task_name = "_".join(args.task_types)
    model_name = "_".join(args.models)
    layer_name = "_".join(map(str, args.layer_list))
    moment_name = "_".join(map(str, args.moment_list))

    base_name = (
        f"torch_blis_{sub_dataset_name}_{task_name}_{model_name}_"
        f"{args.wavelet_type}_L{layer_name}_M{moment_name}_{timestamp}"
    )

    per_seed_path = os.path.join(args.results_dir, f"{base_name}_per_seed.csv")
    summary_path = os.path.join(args.results_dir, f"{base_name}_summary.csv")

    results_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=" * 90)
    print(f"Saved per-seed results to: {per_seed_path}")
    print(f"Saved summary results to: {summary_path}")
    print("=" * 90)

    print(summary_df)


if __name__ == "__main__":
    main()