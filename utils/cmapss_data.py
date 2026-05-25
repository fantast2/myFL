"""
CMAPSS data loading and federated partitioning for turbofan RUL prediction.

Supports FD001-FD004 with sliding-window sequence construction,
engine-based non-IID client partitioning, and train/test normalization
that avoids data leakage (fit stats on train only).
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as data

from utils.util_logger import logger

# ---------------------------------------------------------------------------
# sensor columns to drop (near-constant across all CMAPSS subsets)
# These indices are 0-based within the 21-sensor block (cols 5-25 in raw file).
# Dropping them improves training stability.
_SENSOR_DROP_INDICES = [0, 1, 3, 4, 5, 7, 15, 17, 18]  # keeps 12 sensors

# All 21 sensor column names (for reference)
_SENSOR_NAMES = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
    "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17",
    "s18", "s19", "s20", "s21",
]


def _select_sensor_indices(raw_sensor_cols: int = 21) -> List[int]:
    """Return indices of sensor columns to keep (0-based within sensor block)."""
    return [i for i in range(raw_sensor_cols) if i not in _SENSOR_DROP_INDICES]


def _parse_cmapss_txt(filepath: str) -> np.ndarray:
    """Parse a space-separated CMAPSS txt file into a 2D numpy array."""
    return np.loadtxt(filepath, dtype=np.float32, ndmin=2)


def _build_rul_targets(cycles: np.ndarray, rul_cap: int = 130) -> np.ndarray:
    """
    Compute piecewise-linear RUL from cycle numbers for a single engine.

    RUL = max_cycle - current_cycle (capped at `rul_cap`).
    Returns array of same length as `cycles`.
    """
    max_cycle = cycles[-1]  # last cycle = failure point
    rul = max_cycle - cycles
    if rul_cap and rul_cap > 0:
        rul = np.minimum(rul, rul_cap)
    return rul.astype(np.float32)


def _load_cmapss_subset(
    data_dir: str,
    subset: str,
    rul_cap: int = 130,
    keep_sensors: Optional[List[int]] = None,
    keep_op_settings: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """
    Load a CMAPSS subset (e.g. fd001) and return per-engine train/test data.

    Returns:
        train_by_engine: {unit_id: array of shape (N, features)}
        test_by_engine:  {unit_id: array of shape (N, features)}
        test_rul:        array of shape (num_test_engines,) – true RUL per engine
    """
    if keep_sensors is None:
        keep_sensors = _select_sensor_indices()

    train_path = os.path.join(data_dir, f"train_{subset.upper()}.txt")
    test_path = os.path.join(data_dir, f"test_{subset.upper()}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset.upper()}.txt")

    if not all(os.path.isfile(p) for p in [train_path, test_path, rul_path]):
        raise FileNotFoundError(
            f"CMAPSS files not found. Expected:\n"
            f"  {train_path}\n  {test_path}\n  {rul_path}\n"
            f"Please download CMAPSS data and extract to: {data_dir}"
        )

    train_raw = _parse_cmapss_txt(train_path)
    test_raw = _parse_cmapss_txt(test_path)
    test_rul = _parse_cmapss_txt(rul_path).flatten()

    # Select feature columns from the raw data
    # cols: 0=unit, 1=time, 2-4=op_settings, 5-25=sensors
    if keep_op_settings:
        col_indices = [2, 3, 4] + [5 + s for s in keep_sensors]
    else:
        col_indices = [5 + s for s in keep_sensors]

    # --- training data ---
    train_by_engine = {}
    train_raw_columns = train_raw.shape[1]
    for row in train_raw:
        unit_id = int(row[0])
        features = row[col_indices].flatten() if col_indices[0] < train_raw_columns else np.array([])
        if unit_id not in train_by_engine:
            train_by_engine[unit_id] = []
        train_by_engine[unit_id].append(features)

    for uid in train_by_engine:
        arr = np.array(train_by_engine[uid], dtype=np.float32)
        cycles = np.arange(1, len(arr) + 1, dtype=np.float32)
        rul = _build_rul_targets(cycles, rul_cap)
        # Append RUL as the last feature column (will be split later)
        train_by_engine[uid] = np.hstack([arr, rul.reshape(-1, 1)])

    # --- test data ---
    test_by_engine = {}
    for row in test_raw:
        unit_id = int(row[0])
        features = row[col_indices].flatten() if col_indices[0] < train_raw_columns else np.array([])
        if unit_id not in test_by_engine:
            test_by_engine[unit_id] = []
        test_by_engine[unit_id].append(features)

    for uid in test_by_engine:
        arr = np.array(test_by_engine[uid], dtype=np.float32)
        test_by_engine[uid] = arr  # no RUL appended here; test labels come from test_rul file

    return train_by_engine, test_by_engine, test_rul


def _sliding_windows(
    engine_data: np.ndarray,
    window_size: int,
    stride: int = 1,
    pred_horizon: int = 1,
    has_labels: bool = True,
    label_idx: int = -1,
    base_rul: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a single engine's time series into sliding-window samples.

    Args:
        engine_data: (seq_len, features) array. If has_labels, last column is RUL.
        window_size: number of time steps per input window.
        stride: step between consecutive windows.
        pred_horizon: RUL prediction horizon (label taken from this many steps ahead).
        has_labels: whether engine_data includes label column.
        label_idx: column index of the label (default -1 = last).
        base_rul: if provided (test mode), the RUL at the *last* cycle of the
            original sequence. Per-window labels are computed as base_rul plus
            the number of cycles remaining from the label position to end of seq.

    Returns:
        X: (num_windows, window_size, num_features)
        y: (num_windows,)
    """
    orig_seq_len = engine_data.shape[0]
    if has_labels:
        features = engine_data[:, :label_idx]
        labels = engine_data[:, label_idx]
    else:
        features = engine_data
        labels = None

    num_features = features.shape[1]
    min_len = window_size + pred_horizon

    pad_len = 0
    if orig_seq_len < min_len:
        pad_len = min_len - orig_seq_len
        pad_features = np.repeat(features[:1], pad_len, axis=0)
        features = np.concatenate([pad_features, features], axis=0)
        if has_labels:
            pad_labels = np.full(pad_len, labels[0], dtype=labels.dtype)
            labels = np.concatenate([pad_labels, labels], axis=0)

    seq_len = features.shape[0]  # may include padding
    window_starts = list(range(0, seq_len - window_size - pred_horizon + 1, stride))

    X = np.zeros((len(window_starts), window_size, num_features), dtype=np.float32)
    y = np.zeros((len(window_starts),), dtype=np.float32)

    for i, start in enumerate(window_starts):
        end = start + window_size
        X[i] = features[start:end]
        if has_labels:
            y[i] = labels[end + pred_horizon - 1]
        elif base_rul is not None:
            label_pos = end + pred_horizon - 1
            orig_pos = label_pos - pad_len
            y[i] = base_rul + (orig_seq_len - 1 - orig_pos)
        else:
            y[i] = 0.0  # placeholder

    return X, y


class CmapssDataset(data.Dataset):
    """PyTorch Dataset for CMAPSS sliding-window samples."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()   # (N, window, features)
        self.y = torch.from_numpy(y).float()   # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CmapssNormalizer:
    """Fit on training data, then apply to any dataset (avoids leakage)."""

    def __init__(self, method: str = "standard"):
        if method not in ("standard", "minmax", "none"):
            raise ValueError(f"Unknown normalization method: {method}")
        self.method = method
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None

    def fit(self, X: np.ndarray) -> "CmapssNormalizer":
        """Fit normalizer on 3D array (N, window, features)."""
        # flatten windows to compute global feature-wise stats
        flat = X.reshape(-1, X.shape[-1])
        if self.method == "standard":
            self.mean = flat.mean(axis=0, keepdims=False)
            self.std = flat.std(axis=0, keepdims=False)
            self.std[self.std < 1e-8] = 1.0  # avoid div-by-zero for constant features
        elif self.method == "minmax":
            self.min_val = flat.min(axis=0, keepdims=False)
            self.max_val = flat.max(axis=0, keepdims=False)
            denom = self.max_val - self.min_val
            denom[denom < 1e-8] = 1.0
            self.max_val = self.min_val + denom  # adjust safely
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization to 3D array."""
        if self.method == "none":
            return X
        X = X.copy()
        if self.method == "standard":
            X = (X - self.mean) / self.std
        elif self.method == "minmax":
            X = (X - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def build_cmapss_client_data(
    data_dir: str,
    subset: str = "fd001",
    num_clients: int = 5,
    partition_type: str = "noniid",
    partition_beta: float = 0.25,
    window_size: int = 30,
    stride: int = 1,
    pred_horizon: int = 1,
    normalization_method: str = "standard",
    rul_cap: int = 130,
    batch_size: int = 64,
) -> Tuple[Dict[int, Tuple[data.DataLoader, data.DataLoader]], data.DataLoader]:
    """
    Build per-client train/test DataLoaders for CMAPSS federated learning.

    Partitioning is done by ENGINE ID to prevent data leakage:
      - All windows from the same engine stay together in one client.
      - IID partitioning: engines assigned randomly to clients.
      - Non-IID partitioning: engines assigned based on mean RUL distribution
        (Dirichlet), so clients see different degradation profiles.

    Returns:
        client_data_loader: {client_id: (train_loader, test_loader)}
        global_test_loader: DataLoader over all test engines (server eval)
    """
    # --- 1. Load raw engine data ---
    train_engines, test_engines, test_rul = _load_cmapss_subset(
        data_dir, subset, rul_cap=rul_cap
    )

    # Determine target scale factor: divide RUL by rul_cap so targets ∈ [0, ~1]
    # This stabilises training dramatically for MSE-based regression.
    _target_scale = float(rul_cap) if rul_cap and rul_cap > 0 else 1.0
    if _target_scale < 1.0:
        _target_scale = 1.0

    # --- 2. Build sliding windows per engine (training only) ---
    train_windows = {}   # uid -> (X, y)
    num_features = None

    for uid, eng_data in train_engines.items():
        X, y = _sliding_windows(eng_data, window_size, stride, pred_horizon, has_labels=True)
        y = y / _target_scale  # scale to [0, ~1]
        train_windows[uid] = (X, y)
        if num_features is None:
            num_features = X.shape[-1]

    # --- 3. Normalize: fit on training data only, then apply to train ---
    all_train_X = np.concatenate([v[0] for v in train_windows.values()], axis=0)
    normalizer = CmapssNormalizer(method=normalization_method)
    normalizer.fit(all_train_X)

    for uid in train_windows:
        X, y = train_windows[uid]
        train_windows[uid] = (normalizer.transform(X), y)

    # --- 4. Build engine-level test samples (last window per engine) ---
    # Both client and global evaluation use the same format: one sample per engine.
    # This avoids window-level vs engine-level metric mismatch.
    test_engine_ids = sorted(test_engines.keys())
    engine_test_X_list = []
    engine_test_y_list = []
    for uid in test_engine_ids:
        eng_data = test_engines[uid]
        engine_idx = test_engine_ids.index(uid)
        # Take only the last window_size cycles
        if len(eng_data) >= window_size:
            last_win = eng_data[-window_size:]
        else:
            pad = window_size - len(eng_data)
            last_win = np.concatenate(
                [np.repeat(eng_data[:1], pad, axis=0), eng_data], axis=0)
        engine_test_X_list.append(last_win)
        # True RUL at last recorded cycle, capped consistently with training
        true_rul = float(test_rul[engine_idx])
        if rul_cap and rul_cap > 0:
            true_rul = min(true_rul, float(rul_cap))
        engine_test_y_list.append(true_rul / _target_scale)

    engine_test_X = normalizer.transform(np.stack(engine_test_X_list, axis=0))
    engine_test_y = np.array(engine_test_y_list, dtype=np.float32)
    engine_test_ds = CmapssDataset(engine_test_X, engine_test_y)

    # --- 5. Partition engines across clients ---
    train_engine_ids = sorted(train_windows.keys())
    n_train_engines = len(train_engine_ids)

    if partition_type == "iid":
        logger.info("CMAPSS: IID partitioning (random engine assignment)")
        np.random.shuffle(train_engine_ids)
        engines_per_client = n_train_engines // num_clients
        remainder = n_train_engines % num_clients
        client_engines = {}
        offset = 0
        for cid in range(num_clients):
            count = engines_per_client + (1 if cid < remainder else 0)
            client_engines[cid] = train_engine_ids[offset : offset + count]
            offset += count

    elif partition_type == "noniid":
        logger.info("CMAPSS: Non-IID partitioning (Dirichlet on mean RUL)")
        # Compute mean RUL per engine
        engine_mean_rul = {}
        for uid in train_engine_ids:
            engine_mean_rul[uid] = train_windows[uid][1].mean()

        # Sort engines by mean RUL for Dirichlet partitioning (preserves order-based non-IID)
        sorted_engines = sorted(train_engine_ids, key=lambda u: engine_mean_rul[u])

        if partition_beta <= 0:
            partition_beta = 0.25

        # Assign each class region (sorted by RUL) with Dirichlet proportions
        # Split the sorted engine list into "classes" by RUL percentiles for non-IID
        num_pseudo_classes = num_clients
        class_size = max(1, n_train_engines // num_pseudo_classes)
        engine_classes = []
        for c in range(num_pseudo_classes):
            start = c * class_size
            if c == num_pseudo_classes - 1:
                end = n_train_engines
            else:
                end = start + class_size
            if start < n_train_engines:
                engine_classes.append(sorted_engines[start:end])

        # Dirichlet allocation of each "class" across clients
        client_engines = {cid: [] for cid in range(num_clients)}
        for eng_cls in engine_classes:
            proportions = np.random.dirichlet(np.repeat(partition_beta, num_clients))
            proportions = (proportions * len(eng_cls)).astype(int)
            # Adjust so sum matches class size
            diff = len(eng_cls) - proportions.sum()
            if diff > 0:
                proportions[-1] += diff
            proportions = np.minimum(proportions, len(eng_cls))
            # Cumulative split
            cum = np.cumsum(proportions)
            starts = np.concatenate([[0], cum[:-1]])
            for cid in range(num_clients):
                s, e = int(starts[cid]), min(int(cum[cid]), len(eng_cls))
                if s < e:
                    client_engines[cid].extend(eng_cls[s:e])

    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

    # --- 6. Build per-client DataLoaders ---
    # Per-client train: window-level (for training).
    # Per-client test: engine-level (one sample per engine, same granularity as global).
    #
    # Build a uid→index mapping (engine_test arrays are in sorted uid order).
    _sorted_test_uids = sorted(test_engines.keys())
    _uid_to_idx = {uid: i for i, uid in enumerate(_sorted_test_uids)}

    # Shuffle test engine assignment across clients for fairness.
    np.random.shuffle(test_engine_ids)

    client_data_loader = {}
    for cid in range(num_clients):
        # --- client train set (window-level, as before) ---
        train_X = []
        train_y = []
        for uid in client_engines[cid]:
            X, y = train_windows[uid]
            train_X.append(X)
            train_y.append(y)

        if train_X:
            train_X = np.concatenate(train_X, axis=0)
            train_y = np.concatenate(train_y, axis=0)
        else:
            train_X = np.zeros((0, window_size, num_features), dtype=np.float32)
            train_y = np.zeros((0,), dtype=np.float32)

        train_ds = CmapssDataset(train_X, train_y)
        train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        # --- client test set (engine-level: one prediction per assigned engine) ---
        test_share = len(test_engine_ids) // num_clients
        test_start = cid * test_share
        test_end = (cid + 1) * test_share if cid < num_clients - 1 else len(test_engine_ids)
        c_test_uids = test_engine_ids[test_start:test_end]
        c_test_indices = [_uid_to_idx[uid] for uid in c_test_uids]

        c_test_X = engine_test_X[c_test_indices]
        c_test_y = engine_test_y[c_test_indices]
        test_ds = CmapssDataset(c_test_X, c_test_y)
        test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        client_data_loader[cid] = (train_dl, test_dl)
        logger.info(
            f"Client {cid}: {len(train_ds)} train windows from {len(client_engines[cid])} engines, "
            f"{len(test_ds)} test engines"
        )

    # --- 7. Global test loader: same engine-level data (all test engines) ---
    # Global and per-client evaluation now use the SAME granularity:
    # one prediction per engine at the last recorded cycle.
    global_test_dl = data.DataLoader(
        engine_test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    logger.info(
        f"Global test (CMAPSS standard): {len(engine_test_ds)} engines, "
        f"one prediction per engine")
    logger.info(f"Target scale factor (rul_cap): {_target_scale:.1f}")

    # Store module-level attributes for downstream use
    build_cmapss_client_data._target_scale = _target_scale
    build_cmapss_client_data._client_num_engines = {
        cid: len(client_engines[cid]) for cid in range(num_clients)
    }

    return client_data_loader, global_test_dl


def get_cmapss_num_features(data_dir: str, subset: str = "fd001") -> int:
    """Quickly check how many sensor features will be used (without full loading)."""
    keep_sensors = _select_sensor_indices()
    return len(keep_sensors)
