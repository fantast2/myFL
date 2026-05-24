"""
Smoke tests for CMAPSS data loading, model, and FL integration.

Run: pytest tests/test_cmapss.py -v
"""

import os
import sys
import copy

# Must set LOG_FILE_NAME before importing project utils
os.environ.setdefault("LOG_FILE_NAME", os.path.join(os.path.dirname(__file__), "test_cmapss.log"))

import pytest
import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cmapss_data import (
    _parse_cmapss_txt,
    _load_cmapss_subset,
    _sliding_windows,
    _build_rul_targets,
    CmapssDataset,
    CmapssNormalizer,
    build_cmapss_client_data,
    get_cmapss_num_features,
)
from utils.models import CmapssCNN1D, CmapssLSTM, CmapssGRU, CmapssTransformer

# --- path helpers ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "CMAPSS")
TRAIN_FD001 = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FD001 = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FD001 = os.path.join(DATA_DIR, "RUL_FD001.txt")

requires_cmapss = pytest.mark.skipif(
    not (os.path.isfile(TRAIN_FD001) and os.path.isfile(TEST_FD001) and os.path.isfile(RUL_FD001)),
    reason="CMAPSS data files not found in data/CMAPSS/",
)


# =============================================================================
# Unit tests
# =============================================================================

class TestRulTargets:
    def test_basic(self):
        cycles = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        rul = _build_rul_targets(cycles, rul_cap=130)
        expected = np.array([4, 3, 2, 1, 0], dtype=np.float32)
        assert np.allclose(rul, expected)

    def test_with_cap(self):
        cycles = np.arange(1, 201, dtype=np.float32)  # max=200
        rul = _build_rul_targets(cycles, rul_cap=50)
        expected = np.minimum(200 - cycles, 50).astype(np.float32)
        assert np.allclose(rul, expected)


class TestSlidingWindows:
    def test_basic(self):
        # seq_len=10, features=3, window_size=4, stride=2
        data = np.random.randn(10, 4).astype(np.float32)  # last col = labels
        X, y = _sliding_windows(data, window_size=4, stride=2)
        # Starts: 0,2,4 -> 3 windows
        assert X.shape == (3, 4, 3)
        assert y.shape == (3,)

    def test_short_sequence(self):
        data = np.random.randn(5, 4).astype(np.float32)
        with pytest.raises(ValueError):
            _sliding_windows(data, window_size=10)


class TestCmapssNormalizer:
    def test_standard(self):
        X = np.random.randn(20, 5, 3).astype(np.float32)
        norm = CmapssNormalizer("standard")
        Xn = norm.fit_transform(X)
        assert Xn.shape == X.shape
        # after standard normalization, mean should be ~0 per feature
        flat = Xn.reshape(-1, 3)
        assert np.allclose(flat.mean(axis=0), 0, atol=1e-5)

    def test_minmax(self):
        X = np.random.randn(20, 5, 3).astype(np.float32) * 2 + 5
        norm = CmapssNormalizer("minmax")
        norm.fit(X)
        Xn = norm.transform(X)
        flat = Xn.reshape(-1, 3)
        assert flat.min() >= -1e-6
        assert flat.max() <= 1 + 1e-6

    def test_none(self):
        X = np.random.randn(20, 5, 3).astype(np.float32)
        norm = CmapssNormalizer("none")
        Xn = norm.fit_transform(X)
        assert np.allclose(X, Xn)

    def test_fit_train_only(self):
        """Verify transform uses train stats, not test stats."""
        train = np.ones((10, 3, 2), dtype=np.float32) * 5.0
        test = np.ones((3, 3, 2), dtype=np.float32) * 10.0
        norm = CmapssNormalizer("standard")
        norm.fit(train)
        Xt = norm.transform(test)
        # test data (10) normalized by train stats (mean=5, std=0→clipped to 1)
        # mean=5, std=1 -> (10-5)/1 = 5
        assert np.allclose(Xt, 5.0)


class TestCmapssDataset:
    def test_basic(self):
        X = np.random.randn(50, 10, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        ds = CmapssDataset(X, y)
        assert len(ds) == 50
        x0, y0 = ds[0]
        assert x0.shape == (10, 5)
        assert y0.shape == ()

    def test_dataloader(self):
        X = np.random.randn(50, 10, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        ds = CmapssDataset(X, y)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
        for batch_x, batch_y in loader:
            assert batch_x.shape[1:] == (10, 5)
            assert len(batch_y.shape) == 1
            break


# =============================================================================
# Integration tests (require CMAPSS data files on disk)
# =============================================================================

@requires_cmapss
class TestCmapssDataLoading:
    def test_parse_txt(self):
        data = _parse_cmapss_txt(TRAIN_FD001)
        assert data.ndim == 2
        assert data.shape[1] == 26  # 26 columns in raw CMAPSS

    def test_load_subset(self):
        train, test, test_rul = _load_cmapss_subset(DATA_DIR, "fd001", rul_cap=130)
        assert len(train) == 100  # 100 training engines
        assert len(test) == 100   # 100 test engines
        assert len(test_rul) == 100

    def test_num_features(self):
        nf = get_cmapss_num_features(DATA_DIR, "fd001")
        assert nf > 0
        assert nf <= 21

    def test_build_client_data(self):
        client_loaders, global_loader = build_cmapss_client_data(
            DATA_DIR, subset="fd001", num_clients=3, partition_type="iid",
            window_size=30, stride=2, batch_size=16, rul_cap=130,
        )
        assert len(client_loaders) == 3
        for cid, (train_dl, test_dl) in client_loaders.items():
            assert len(train_dl.dataset) > 0
            # Check a batch
            for Xb, yb in train_dl:
                assert Xb.ndim == 3  # (batch, window, features)
                assert yb.ndim == 1
                break
        # Global test loader
        for Xb, yb in global_loader:
            assert Xb.ndim == 3
            break

    def test_noniid_partition(self):
        client_loaders, _ = build_cmapss_client_data(
            DATA_DIR, subset="fd001", num_clients=5, partition_type="noniid",
            partition_beta=0.5, window_size=30, batch_size=16, rul_cap=130,
        )
        assert len(client_loaders) == 5
        sizes = [len(v[0].dataset) for v in client_loaders.values()]
        # Every client should get some data
        assert all(s > 0 for s in sizes)


@requires_cmapss
class TestCmapssModels:
    @pytest.fixture
    def batch(self):
        return torch.randn(8, 30, 12)  # batch=8, window=30, features=12

    def test_cnn1d_forward(self, batch):
        model = CmapssCNN1D(num_features=12, window_size=30)
        out = model(batch)
        assert out.shape == (8,)
        assert out.dtype == torch.float32

    def test_lstm_forward(self, batch):
        model = CmapssLSTM(num_features=12)
        out = model(batch)
        assert out.shape == (8,)

    def test_gru_forward(self, batch):
        model = CmapssGRU(num_features=12)
        out = model(batch)
        assert out.shape == (8,)

    def test_transformer_forward(self, batch):
        model = CmapssTransformer(num_features=12)
        out = model(batch)
        assert out.shape == (8,)

    def test_cnn1d_train_one_step(self, batch):
        model = CmapssCNN1D(num_features=12, window_size=30)
        target = torch.randn(8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        out = model(batch)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


@requires_cmapss
class TestCmapssFLIntegration:
    """Minimal FL round simulation with CMAPSS data."""

    def test_one_fl_round(self):
        from utils.cmapss_data import build_cmapss_client_data, get_cmapss_num_features
        from utils.util_model import set_cmapss_dims
        from utils.models import CmapssCNN1D
        from utils.util_fusion import fusion_fedavg

        nf = get_cmapss_num_features(DATA_DIR, "fd001")
        set_cmapss_dims(num_features=nf, window_size=30)

        client_loaders, global_loader = build_cmapss_client_data(
            DATA_DIR, subset="fd001", num_clients=3, partition_type="iid",
            window_size=30, stride=5, batch_size=16, rul_cap=130,
        )

        device = torch.device("cpu")
        server_model = CmapssCNN1D(num_features=nf, window_size=30).to(device)

        client_models = {}
        for cid in range(3):
            m = CmapssCNN1D(num_features=nf, window_size=30).to(device)
            m.load_state_dict(server_model.state_dict())
            client_models[cid] = m

        # Local training (1 epoch)
        criterion = nn.MSELoss()
        for cid in range(3):
            train_dl, _ = client_loaders[cid]
            m = client_models[cid]
            m.train()
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(m(Xb), yb)
                loss.backward()
                opt.step()
                break  # one batch only for smoke test

        # Aggregate
        data_sizes = {
            cid: len(client_loaders[cid][0].dataset) for cid in range(3)
        }
        aggregated = fusion_fedavg(client_models, data_sizes)
        server_model.load_state_dict(aggregated)

        # Evaluate
        server_model.eval()
        total_rmse = 0.0
        n_batches = 0
        with torch.no_grad():
            for Xb, yb in global_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = server_model(Xb)
                total_rmse += ((preds - yb) ** 2).mean().sqrt().item()
                n_batches += 1
        rmse = total_rmse / max(n_batches, 1)
        assert rmse > 0
        print(f"\n  CMAPSS FL round RMSE (1 batch/client, 3 clients): {rmse:.4f}")
