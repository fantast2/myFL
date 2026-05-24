from abc import ABC, abstractmethod
import time
import copy
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from utils.util_sys import get_available_device, intersection_of_lists
from utils.util_data import get_client_data_loader
from utils.util_data import get_global_test_data_loader
from utils.util_model import get_client_model
from utils.util_model import (
    ipm_attack_craft_model,
    scaling_attack,
    alie_attack,
    poison_attack,
)

from utils.util_model import get_server_model
from utils.util_fusion import (
    fusion_avg,
    fusion_clipping_median,
    fusion_cos_defense,
    fusion_fedavg,
    fusion_krum,
    fusion_median,
    fusion_trimmed_mean,
    fusion_dual_defense,
)
from utils.util_logger import logger


class SimulationFL(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.device = config.get("device", None)

        self.num_clients = config.get("num_clients", 5)
        self.dataset = config.get("dataset", "mnist")
        self.is_regression = self.dataset == "cmapss"
        self.fusion = config.get("fusion", "fedavg")
        self.partion_type = config.get("partition_type", "noniid")
        self.partion_dirichlet_beta = config.get("partition_dirichlet_beta", 0.25)
        self.dir_data = config.get("dir_data", "./data/")

        self.training_round = config.get("training_round", 10)
        self.local_epochs = config.get("local_epochs", 1)
        self.optimizer = config.get("optimizer", "sgd")
        self.learning_rate = config.get("learning_rate", 0.01)
        self.batch_size = config.get("batch_size", 64)
        self.regularization = config.get("regularization", 1e-5)

        self.attacker_ratio = config.get("attacker_ratio", 0.0)
        self.attacker_strategy = config.get("attacker_strategy", None)
        self.attacker_list = []
        self.attack_start_round = config.get("attack_start_round", -1)
        self.epsilon = config.get("epsilon", None)

        # CMAPSS / regression settings
        self.loss_type = config.get("loss_type", "mse")
        self.cmapss_subset = config.get("cmapss_subset", "fd001")
        self.window_size = config.get("window_size", 30)
        self.stride = config.get("stride", 1)
        self.pred_horizon = config.get("pred_horizon", 1)
        self.normalization_method = config.get("normalization_method", "standard")
        self.rul_cap = config.get("rul_cap", 130)
        self.cmapss_data_dir = config.get("cmapss_data_dir", None)

        self.metrics = {}
        self.tensorboard = config.get("tensorboard", None)

        # setup random seed
        self.seed = config.get("seed", 1001)

    def init_seed(self) -> None:
        if self.seed is not None and self.seed > 0:
            logger.info("setting up the seed as {}".format(self.seed))
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            random.seed(self.seed)
        else:
            logger.info("no seed is set")

    def init_data(self) -> None:
        if self.is_regression:
            self.client_data_loader = get_client_data_loader(
                self.dataset,
                self.dir_data,
                self.num_clients,
                self.partion_type,
                self.partion_dirichlet_beta,
                self.batch_size,
                cmapss_subset=self.cmapss_subset,
                window_size=self.window_size,
                stride=self.stride,
                pred_horizon=self.pred_horizon,
                normalization_method=self.normalization_method,
                rul_cap=self.rul_cap,
                cmapss_data_dir=self.cmapss_data_dir,
            )
        else:
            self.client_data_loader = get_client_data_loader(
                self.dataset,
                self.dir_data,
                self.num_clients,
                self.partion_type,
                self.partion_dirichlet_beta,
                self.batch_size,
            )
        self.server_test_data_loader = get_global_test_data_loader(
            self.dataset, self.dir_data, self.batch_size
        )

    def init_model(self) -> None:
        self.client_model = get_client_model(
            self.dataset, self.num_clients, self.device
        )
        self.server_model = get_server_model(self.dataset, self.device)

    def init_client_per_round(self) -> None:
        num_client_per_round = min(self.num_clients, 10)
        client_list_all = [i for i in range(self.num_clients)]
        round_client_list = []
        if num_client_per_round != self.num_clients:
            for _ in range(self.training_round):
                _client_list = random.sample(client_list_all, num_client_per_round)
                _client_list.sort()
                round_client_list.append(_client_list)
        else:
            for _ in range(self.training_round):
                round_client_list.append(client_list_all)
        self.round_client_list = round_client_list

    def init_attacker(self) -> None:
        if (
            self.attacker_strategy is not None
            and self.attacker_strategy != "none"
            and self.attacker_ratio > 0
        ):
            logger.info(
                "attacker env is set  with strategy: {} and ratio: {}".format(
                    self.attacker_strategy, self.attacker_ratio
                )
            )
            size_attackers = int(self.attacker_ratio * self.num_clients)
            self.attacker_list = random.sample(range(self.num_clients), size_attackers)
            logger.info("attacker list: {}".format(self.attacker_list))

    def init_device(self) -> None:
        self.device = get_available_device()

    def start(self):
        self.init_device()
        # self.init_seed()
        # self.init_attacker()
        self.init_client_per_round()
        self.init_data()
        self.init_model()

        logger.info("start the FL simulation")

        time_start = time.perf_counter()

        for _round_idx in range(self.training_round):
            logger.info(f"start training round {_round_idx}")
            self.metrics[_round_idx] = {"time": None, "parties": {}, "server": {}}

            # simulate query each client
            server_model_params = self.server_model.state_dict()
            round_client_list = self.round_client_list[_round_idx]
            round_client_models = {
                pid: self.client_model[pid] for pid in round_client_list
            }

            last_round_attackers = (
                intersection_of_lists(
                    self.round_client_list[_round_idx - 1], self.attacker_list
                )
                if _round_idx >= 1
                else intersection_of_lists(
                    self.round_client_list[_round_idx], self.attacker_list
                )
            )
            for _pid, _model in round_client_models.items():
                # part of hyper guard defense mechanism
                if (
                    _pid not in last_round_attackers
                    and _round_idx >= self.attack_start_round
                    and (
                        self.attacker_strategy.startswith("model_poisoning")
                        and self.attacker_strategy != "model_poisoning_ipm"
                    )
                    and self.fusion == "dual_defense"
                ):
                    fused_params = copy.deepcopy(server_model_params)
                    for param_key in server_model_params.keys():
                        fused_params[param_key] = torch.clamp(
                            fused_params[param_key], -0.2, 0.2
                        )
                    _model.load_state_dict(fused_params)
                else:
                    _model.load_state_dict(server_model_params)

            if (
                self.attacker_strategy is not None
                and self.attacker_strategy != "none"
                and self.attacker_ratio > 0
                and self.attack_start_round <= _round_idx - 1
            ):
                self.attacker_list = random.sample(
                    round_client_list, int(self.attacker_ratio * len(round_client_list))
                )
                logger.info(f"round {_round_idx} attackers: {self.attacker_list}")
            else:
                logger.info(f"no attack at the round {_round_idx}")

            # simulate local training in parallel
            model_dict = {}
            for _client_id in round_client_list:
                logger.info(f"start client {_client_id} training")
                model_client, eval_metrics = self.client_local_train(
                    _round_idx, _client_id, round_client_models[_client_id]
                )
                model_dict[_client_id] = model_client
                logger.info(f"end client {_client_id} training")

                # RECORD PARTY METRICS
                logger.info(f"client {_client_id} evaluation metrics: {eval_metrics}")
                self.metrics[_round_idx]["parties"][_client_id] = eval_metrics
                # self.tensorboard.add_scalar(
                #     "{}-{} - client {} Test Acc".format(
                #         self.dataset, self.fusion, _client_id
                #     ),
                #     eval_metrics["test_acc"],
                #     _round_idx,
                # )

            # simulate aggregation
            aggregated_params = self.aggregate_model(_round_idx, model_dict)
            self.server_model.load_state_dict(aggregated_params)

            # RECORD GLOBAL METRICS
            criterion = self._get_criterion()
            eval_result = self.model_evaluate(
                self.server_model, self.server_test_data_loader, criterion
            )
            self.metrics[_round_idx]["server"] = eval_result

            if self.is_regression:
                logger.info(
                    f"global side - test RMSE: {eval_result['rmse']:.4f}, "
                    f"MAE: {eval_result['mae']:.4f}, loss: {eval_result['loss']:.4f}"
                )
                prefix = f"{self.dataset}-{self.fusion}-{self.cmapss_subset}"
                self.tensorboard.add_scalar(
                    f"{prefix} - Server RMSE", eval_result["rmse"], _round_idx
                )
                self.tensorboard.add_scalar(
                    f"{prefix} - Server MAE", eval_result["mae"], _round_idx
                )
            else:
                logger.info(f"global side - test accuracy: {eval_result['acc']:.2f}%")
                prefix = f"{self.dataset}-{self.fusion}"
                self.tensorboard.add_scalar(
                    f"{prefix} - Server Test Acc", eval_result["acc"], _round_idx
                )
            self.tensorboard.flush()

            time_round_end = time.perf_counter()
            self.metrics[_round_idx]["time"] = time_round_end - time_start

        logger.info("end the FL simulation")
        logger.info("summarization - simulation metrics: {}".format(self.metrics))
        self.tensorboard.close()

    def client_local_train(
        self, round_idx: int, client_id: int, client_model: nn.Module
    ) -> None:

        logger.info(f"client {client_id} start local training ...")
        train_data_loader, test_data_loader = self.client_data_loader[client_id]

        model = client_model.to(self.device)
        model.train()
        criterion = self._get_criterion()
        optimizer = self.get_optimizer(model)

        for _epoch in range(self.local_epochs):
            train_loss_lst = []
            epoch_total = 0

            _size_total_data = len(train_data_loader.dataset)
            _size_batch = len(train_data_loader)

            for _batch_idx, (_data, _target) in enumerate(train_data_loader):
                data_batch = _data.to(self.device)
                target_batch = _target.to(self.device)

                optimizer.zero_grad()
                output = model(data_batch)
                loss = criterion(output, target_batch)
                loss.backward()
                optimizer.step()

                train_loss_lst.append(loss.item())
                epoch_total += target_batch.size(0)

            epoch_avg_loss = np.mean(train_loss_lst)

        # --- handle attacker model poisoning ---
        # attacker code paths call model_evaluate which now returns a dict
        if (
            self.attacker_strategy == "model_poisoning_ipm"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = ipm_attack_craft_model(
                self.server_model.to(self.device), model.to(self.device)
            )
            eval_test = self.model_evaluate(crafted_model, test_data_loader, criterion)
            eval_train = self.model_evaluate(crafted_model, train_data_loader, criterion)
            return crafted_model, {"train_loss": eval_train["loss"], **eval_test}
        elif (
            self.attacker_strategy == "model_poisoning_scaling"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = scaling_attack(model.to(self.device))
            eval_test = self.model_evaluate(crafted_model, test_data_loader, criterion)
            eval_train = self.model_evaluate(crafted_model, train_data_loader, criterion)
            return crafted_model, {"train_loss": eval_train["loss"], **eval_test}
        elif (
            self.attacker_strategy == "model_poisoning_alie"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = alie_attack(model.to(self.device))
            eval_test = self.model_evaluate(crafted_model, test_data_loader, criterion)
            eval_train = self.model_evaluate(crafted_model, train_data_loader, criterion)
            return crafted_model, {"train_loss": eval_train["loss"], **eval_test}
        elif (
            self.attacker_strategy != "none"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = poison_attack(self.attacker_strategy, model.to(self.device))
            eval_test = self.model_evaluate(crafted_model, test_data_loader, criterion)
            eval_train = self.model_evaluate(crafted_model, train_data_loader, criterion)
            return crafted_model, {"train_loss": eval_train["loss"], **eval_test}
        else:
            eval_test = self.model_evaluate(model, test_data_loader, criterion)
            eval_train = self.model_evaluate(model, train_data_loader, criterion)
            return model, {"train_loss": eval_train["loss"], **eval_test}

    def _batch_records_debug(
        self,
        epoch: int,
        batch_idx: int,
        size_total_data: int,
        size_data: int,
        size_batch: int,
        loss: Any,
    ) -> None:
        if batch_idx % 10 == 0:
            logger.debug(
                "train epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}".format(
                    epoch,
                    batch_idx * size_data,
                    size_total_data,
                    100.0 * batch_idx / size_batch,
                    loss.item(),
                )
            )

    def _get_criterion(self) -> nn.Module:
        """Return the appropriate loss function for the current dataset."""
        if self.is_regression:
            if self.loss_type == "smooth_l1":
                return nn.SmoothL1Loss().to(self.device)
            elif self.loss_type == "huber":
                return nn.HuberLoss().to(self.device)
            else:
                return nn.MSELoss().to(self.device)
        else:
            return nn.CrossEntropyLoss().to(self.device)

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
            )
        elif self.optimizer == "amsgrad":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
                amsgrad=True,
            )
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.regularization,
            )
        return optimizer

    def _get_target_scale(self) -> float:
        """Return the RUL scaling factor used during data processing (1.0 if none)."""
        if not self.is_regression:
            return 1.0
        try:
            from utils.cmapss_data import build_cmapss_client_data
            return getattr(build_cmapss_client_data, "_target_scale", 1.0)
        except Exception:
            return 1.0

    def model_evaluate(
        self,
        model: nn.Module,
        data_loader: data.DataLoader,
        criterion: nn.Module,
    ) -> dict:
        """
        Evaluate model. Returns a dict with metrics appropriate for the task:
          - Classification: {"loss": ..., "acc": ...}
          - Regression:     {"loss": ..., "rmse": ..., "mae": ...}
                             (RMSE/MAE are reported in original RUL scale)
        """
        model.eval()

        total_loss = 0.0
        total_samples = 0

        # regression accumulators (in scaled space)
        sum_sq_err = 0.0
        sum_abs_err = 0.0
        correct = 0

        with torch.no_grad():
            model.to(self.device)
            for _data, _targets in data_loader:
                data_batch = _data.to(self.device)
                targets_batch = _targets.to(self.device)

                outputs = model(data_batch)
                loss = criterion(outputs, targets_batch)
                total_loss += loss.item() * targets_batch.size(0)
                total_samples += targets_batch.size(0)

                if self.is_regression:
                    preds = outputs.view(-1)
                    tgts = targets_batch.view(-1)
                    sum_sq_err += ((preds - tgts) ** 2).sum().item()
                    sum_abs_err += (preds - tgts).abs().sum().item()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == targets_batch).sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if self.is_regression:
            scale = self._get_target_scale()
            rmse = ((sum_sq_err / total_samples) ** 0.5 * scale) if total_samples > 0 else 0.0
            mae = (sum_abs_err / total_samples * scale) if total_samples > 0 else 0.0
            return {"loss": avg_loss, "rmse": rmse, "mae": mae}
        else:
            accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0
            return {"loss": avg_loss, "acc": accuracy}

    # def aggregate_model(self, round_idx: int, model_updates: dict) -> Dict[str, Any]:
    #     logger.info("start model aggregation...fusion method: {}".format(self.fusion))
    #
    #     if self.fusion == "average":
    #         average_params = fusion_avg(model_updates)
    #         return average_params
    #     elif self.fusion == "fedavg":
    #         data_sizes = {
    #             p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
    #             for p_id in self.round_client_list[round_idx]
    #         }
    #         logger.debug("data sizes: {}".format(data_sizes))
    #         weighted_avg_params = fusion_fedavg(model_updates, data_sizes)
    #         return weighted_avg_params
    #     elif self.fusion == "krum":
    #         # max_expected_adversaries = int(self.attacker_ratio * self.num_clients)
    #         max_expected_adversaries = int(self.attacker_ratio * len(model_updates))
    #         krum_params = fusion_krum(
    #             model_updates, max_expected_adversaries, self.device
    #         )
    #         return krum_params
    #     elif self.fusion == "median":
    #         median_params = fusion_median(model_updates, device=self.device)
    #         return median_params
    #     elif self.fusion == "clipping_median":
    #         median_clipping_params = fusion_clipping_median(
    #             model_updates, clipping_threshold=0.1, device=self.device
    #         )
    #         return median_clipping_params
    #     elif self.fusion == "trimmed_mean":
    #         trimmed_mean_params = fusion_trimmed_mean(
    #             model_updates, trimmed_ratio=0.1, device=self.device
    #         )
    #         return trimmed_mean_params
    #     elif self.fusion == "cos_defense":
    #         weighted_params = fusion_cos_defense(self.server_model, model_updates)
    #         return weighted_params
    #     elif self.fusion == "dual_defense":
    #         logger.info("start hyper-guard fusion with epsilon {}".format(self.epsilon))
    #         lst_round_attackers = intersection_of_lists(
    #             list(model_updates.keys()), self.attacker_list
    #         )
    #         logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
    #         data_sizes = {
    #             p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
    #             for p_id in self.round_client_list[round_idx]
    #         }
    #         fused_params = fusion_dual_defense(
    #             self.server_model,
    #             model_updates,
    #             data_sizes,
    #             epsilon=self.epsilon,
    #         )
    #         return fused_params
    #     else:
    #         raise ValueError("Invalid fusion method")

    # def aggregate_model(self, round_idx: int, model_updates: dict) -> Dict[str, Any]:
    #     logger.info("start model aggregation...fusion method: {}".format(self.fusion))
    #
    #     if self.fusion == "average":
    #         average_params = fusion_avg(model_updates)
    #         return average_params
    #
    #     elif self.fusion == "fedavg":
    #         data_sizes = {
    #             p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
    #             for p_id in self.round_client_list[round_idx]
    #         }
    #         logger.debug("data sizes: {}".format(data_sizes))
    #         weighted_avg_params = fusion_fedavg(model_updates, data_sizes)
    #         return weighted_avg_params
    #
    #     elif self.fusion == "krum":
    #         max_expected_adversaries = int(self.attacker_ratio * len(model_updates))
    #         krum_params = fusion_krum(
    #             model_updates, max_expected_adversaries, self.device
    #         )
    #         return krum_params
    #
    #     elif self.fusion == "median":
    #         median_params = fusion_median(model_updates, device=self.device)
    #         return median_params
    #
    #     elif self.fusion == "clipping_median":
    #         median_clipping_params = fusion_clipping_median(
    #             model_updates, clipping_threshold=0.1, device=self.device
    #         )
    #         return median_clipping_params
    #
    #     elif self.fusion == "trimmed_mean":
    #         trimmed_mean_params = fusion_trimmed_mean(
    #             model_updates, trimmed_ratio=0.1, device=self.device
    #         )
    #         return trimmed_mean_params
    #
    #     elif self.fusion == "cos_defense":
    #         weighted_params = fusion_cos_defense(self.server_model, model_updates)
    #         return weighted_params
    #
    #     elif self.fusion == "dual_defense":
    #         logger.info("start secure robust fusion with epsilon {}".format(self.epsilon))
    #         data_sizes = {
    #             p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
    #             for p_id in self.round_client_list[round_idx]
    #         }
    #
    #         fused_params = fusion_dual_defense(
    #             self.server_model,
    #             model_updates,
    #             data_sizes,
    #             epsilon=self.epsilon,
    #         )
    #
    #         # Optional post-aggregation clipping to harden against extreme updates.
    #         if self.attacker_strategy is not None and self.attacker_strategy != "none":
    #             clipped = copy.deepcopy(fused_params)
    #             for key in clipped.keys():
    #                 clipped[key] = torch.clamp(clipped[key], -0.2, 0.2)
    #             fused_params = clipped
    #
    #         return fused_params
    #
    #     else:
    #         raise ValueError("Invalid fusion method")
    def aggregate_model(self, round_idx: int, model_updates: dict) -> Dict[str, Any]:
        logger.info("start model aggregation...fusion method: {}".format(self.fusion))

        if self.fusion == "average":
            return fusion_avg(model_updates)

        elif self.fusion == "fedavg":
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            return fusion_fedavg(model_updates, data_sizes)

        elif self.fusion == "krum":
            max_expected_adversaries = int(self.attacker_ratio * len(model_updates))
            return fusion_krum(model_updates, max_expected_adversaries, self.device)

        elif self.fusion == "median":
            return fusion_median(model_updates, device=self.device)

        elif self.fusion == "clipping_median":
            return fusion_clipping_median(model_updates, clipping_threshold=0.1, device=self.device)

        elif self.fusion == "trimmed_mean":
            return fusion_trimmed_mean(model_updates, trimmed_ratio=0.1, device=self.device)

        elif self.fusion == "cos_defense":
            return fusion_cos_defense(self.server_model, model_updates)

        elif self.fusion == "dual_defense":
            logger.info("start secure robust fusion with epsilon {}".format(self.epsilon))
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            return fusion_dual_defense(
                self.server_model,
                model_updates,
                data_sizes,
                epsilon=self.epsilon,
            )

        else:
            raise ValueError("Invalid fusion method")

