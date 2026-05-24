# Federated Learning Schemes

Federated learning simulation with defense mechanisms, supporting both image classification and CMAPSS turbofan RUL regression.

## Datasets

| Dataset  | Task              | Modality            |
|----------|-------------------|---------------------|
| mnist    | Image classification | Grayscale 28x28    |
| fmnist   | Image classification | Grayscale 28x28    |
| cifar10  | Image classification | Color 32x32         |
| svhn     | Image classification | Color 32x32         |
| cmapss   | RUL regression    | Multivariate time series |

## CMAPSS Data Setup

1. Download the CMAPSS dataset from NASA's prognostic data repository.
2. Extract the zip to `data/CMAPSS/`. The directory should contain:
   - `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
   - `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
   - `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
   - `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`
3. Alternatively, set `--cmapss_data_dir /path/to/CMAPSS/`.

The data is already present in `data/CMAPSS/` in this repository.

## Quick Start

### CMAPSS Federated RUL Prediction

```bash
# FD001, 5 clients, 50 rounds, MSE loss
python main.py --dataset cmapss --cmapss_subset fd001 \
    --num_clients 5 --fusion fedavg --training_round 50 \
    --local_epochs 3 --batch_size 64 --window_size 30 --loss_type mse

# FD001 with non-IID engine partitioning
python main.py --dataset cmapss --cmapss_subset fd001 \
    --num_clients 10 --fusion fedavg --training_round 100 \
    --local_epochs 5 --batch_size 128 --window_size 50 \
    --partition_type noniid --loss_type huber
```

### Image Classification

```bash
python main.py --dataset mnist --num_clients 100 --fusion fedavg \
    --training_round 10 --local_epochs 3 --batch_size 64
```

### With Attacks

```bash
python main.py --dataset cmapss --cmapss_subset fd001 \
    --num_clients 10 --fusion clipping_median --training_round 50 \
    --local_epochs 3 --batch_size 64 \
    --attacker_strategy gaussian --attack_start_round 20 --attacker_ratio 0.2
```

## Key CMAPSS Arguments

| Argument               | Default    | Description                              |
|------------------------|------------|------------------------------------------|
| `--dataset cmapss`     | —          | Enable CMAPSS mode                       |
| `--cmapss_subset`      | fd001      | fd001, fd002, fd003, or fd004            |
| `--window_size`        | 30         | Sliding window length                    |
| `--stride`             | 1          | Window stride                            |
| `--pred_horizon`       | 1          | RUL prediction horizon                   |
| `--normalization_method`| standard  | standard, minmax, or none                |
| `--rul_cap`            | 130        | Max RUL value (0 to disable)             |
| `--loss_type`          | mse        | mse, smooth_l1, or huber                 |
| `--cmapss_data_dir`    | auto       | Custom CMAPSS data directory             |

## Federated Aggregation Methods

`average`, `fedavg`, `krum`, `median`, `clipping_median`, `trimmed_mean`, `cos_defense`, `dual_defense`

All methods are compatible with both classification and regression tasks.

## Models

- **Image:** MNISTCNN, FashionMNISTCNN, ResNet18
- **CMAPSS:** CmapssCNN1D (default baseline), CmapssLSTM, CmapssGRU, CmapssTransformer

The default CMAPSS model is a lightweight 1D CNN. See `utils/models.py` for alternatives.

## Data Partitioning

- **IID:** Random assignment of engines (or samples, for images) to clients.
- **Non-IID:** Dirichlet distribution over classes (images) or RUL-based engine grouping (CMAPSS).
  - For CMAPSS, engines are sorted by mean RUL and partitioned with Dirichlet allocation to simulate heterogeneous degradation profiles across clients.
  - Engines are never split across clients (prevents data leakage).
