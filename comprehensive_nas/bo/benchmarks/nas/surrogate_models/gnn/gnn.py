import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch, DataLoader
from tqdm import tqdm

from .. import utils

# from surrogate_models.gnn.models.vsgae_enc import GNNpred, GNNpred_classifier
from ..surrogate_model import SurrogateModel
from .gnn_utils import NASBenchDataset, Patience
from .models.deep_multisets import DeepMultisets
from .models.diff_pool import DiffPool
from .models.gincnn import GIN


class GNNSurrogateModel(SurrogateModel):
    def __init__(self, gnn_type, data_root, log_dir, seed, model_config, data_config):
        super(GNNSurrogateModel, self).__init__(
            data_root=data_root,
            log_dir=log_dir,
            seed=seed,
            model_config=model_config,
            data_config=data_config,
        )

        self.dataset_statistic = json.load(
            open(
                os.path.join(os.path.dirname(__file__), "../dataset_statistic.json"), "r"
            )
        )
        self.device = torch.device("cpu")

        # Instantiate dataloader to extract one batch in order to know the number of node features
        test_queue = self.load_results_from_result_paths(
            [
                os.path.join(
                    os.path.dirname(__file__), "../test/results_fidelity_0/results_0.json"
                )
            ]
        )
        single_graph_batch = next(iter(test_queue))

        # Instantiate the GNN
        model = self.instantiate_gnn(
            gnn_type=gnn_type,
            num_node_features=single_graph_batch.num_node_features,
            model_config=model_config,
        )
        self.model = model.to(self.device)
        logging.info(
            "Num Parameters {}".format(
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            )
        )

    def instantiate_gnn(self, gnn_type, num_node_features, model_config):
        if gnn_type == "gnn_gin":
            model = GIN(
                dim_features=num_node_features, dim_target=1, model_config=model_config
            )
        elif gnn_type == "gnn_diff_pool":
            model = DiffPool(
                dim_features=num_node_features, dim_target=1, model_config=model_config
            )
        elif gnn_type == "gnn_deep_multisets":
            model = DeepMultisets(
                dim_features=num_node_features, dim_target=1, model_config=model_config
            )
        elif gnn_type == "gnn_vs_gae":
            model = GNNpred(
                dim_features=self.model_config["gnn_node_dimensions"],
                dim_target=1,
                model_config=model_config,
            )
        elif gnn_type == "gnn_vs_gae_classifier":
            model = GNNpred_classifier(
                dim_features=self.model_config["gnn_node_dimensions"],
                dim_target=1,
                model_config=model_config,
            )

        else:
            raise NotImplementedError("Unknown gnn_type.")
        return model

    def load_results_from_result_paths(self, result_paths):
        # Instantiate dataset
        dataset = NASBenchDataset(
            root=self.data_root,
            model_config=self.model_config,
            dataset_statistic=self.dataset_statistic,
            result_paths=result_paths,
            config_loader=self.config_loader,
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset, batch_size=self.model_config["batch_size"], pin_memory=True
        )
        return dataloader

    def train(self):
        if self.model_config["loss_function"] == "L1":
            criterion = torch.nn.L1Loss()
        elif self.model_config["loss_function"] == "L2":
            criterion = torch.nn.MSELoss()
        elif self.model_config["loss_function"] == "HUBER":
            criterion = torch.nn.SmoothL1Loss()

        else:
            raise NotImplementedError("Unknown loss function used.")

        # Create early stopper
        early_stopper = Patience(patience=20, use_loss=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_config["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.model_config["epochs"],
            eta_min=self.model_config["learning_rate_min"],
        )

        # Load training data
        train_queue = self.load_results_from_result_paths(self.train_paths)
        valid_queue = self.load_results_from_result_paths(self.val_paths)

        # Start train loop
        for epoch in tqdm(range(self.model_config["epochs"])):
            logging.info("Starting epoch {}".format(epoch))
            lr = scheduler.get_lr()[0]

            # training
            train_obj, train_results = self.train_epoch(
                train_queue, valid_queue, self.model, criterion, optimizer, lr, epoch
            )

            logging.info("train metrics: %s", train_results)
            scheduler.step()

            # validation
            valid_obj, valid_results = self.infer(
                train_queue, valid_queue, self.model, criterion, optimizer, lr, epoch
            )
            logging.info("validation metrics: %s", valid_results)

            # save the model
            self.save()

            # Early Stopping
            if early_stopper is not None and early_stopper.stop(
                epoch, val_loss=valid_obj, val_acc=valid_results["kendall_tau"]
            ):
                logging.info(
                    "Early Stopping at epoch {}, best is {}".format(
                        epoch, early_stopper.get_best_vl_metrics()
                    )
                )
                break

        return valid_results

    def normalize_data(self, val_accuracy):
        return ((val_accuracy / 100) - 0.1) / 0.9

    def unnormalize_data(self, normalized_accuracy):
        return ((normalized_accuracy * 0.9) + 0.1) * 100

    def create_bins(self, lower_bound, width, quantity):
        bins = []
        for low in range(lower_bound, lower_bound + quantity * width + 1, width):
            bins.append((low, low + width))
        return bins

    def find_bin(self, value, bins):

        for i in range(0, len(bins)):
            if bins[i][0] <= value < bins[i][1]:
                return i
        return -1

    def train_epoch(
        self, train_queue, valid_queue, model, criterion, optimizer, lr, epoch
    ):
        objs = utils.AvgrageMeter()

        # TRAINING
        preds = []
        targets = []

        model.train()

        for step, graph_batch in enumerate(train_queue):
            graph_batch = graph_batch.to(self.device)
            #             print(step)

            if self.model_config["model"] == "gnn_vs_gae_classifier":
                pred_bins, pred = self.model(graph_batch=graph_batch)
                criterion = torch.nn.BCELoss()
                criterion_2 = torch.nn.MSELoss()

                bins = self.create_bins(lower_bound=0, width=10, quantity=9)
                binned_weights = []
                for value in graph_batch.y.cpu().numpy():
                    bin_index = self.find_bin(value, bins)
                    binned_weights.append(bin_index)
                bins = torch.FloatTensor(binned_weights)
                make_one_hot = lambda index: torch.eye(self.model_config["no_bins"])[
                    index.view(-1).long()
                ]
                binns_one_hot = make_one_hot(bins).to(self.device)
                loss_1 = criterion(pred_bins, binns_one_hot)
                loss_2 = criterion_2(pred, self.normalize_data(graph_batch.y))
                alpha = self.model_config["classification_loss"]
                beta = self.model_config["regression_loss"]

                loss = alpha * loss_1 + beta * loss_2

            else:
                pred = self.model(graph_batch=graph_batch)
                loss = criterion(pred, self.normalize_data(graph_batch.y))

            preds.extend(self.unnormalize_data(pred.detach().cpu().numpy()))
            targets.extend(graph_batch.y.detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = graph_batch.num_graphs
            objs.update(loss.data.item(), n)

            if step % self.data_config["report_freq"] == 0:
                logging.info("train %03d %e", step, objs.avg)

        fig = utils.scatter_plot(
            np.array(preds),
            np.array(targets),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_train_{}.jpg".format(epoch)))
        plt.close()
        train_results = utils.evaluate_metrics(
            np.array(targets), np.array(preds), prediction_is_first_arg=False
        )

        return objs.avg, train_results

    def infer(self, train_queue, valid_queue, model, criterion, optimizer, lr, epoch):
        objs = utils.AvgrageMeter()

        # VALIDATION
        preds = []
        targets = []

        model.eval()
        for step, graph_batch in enumerate(valid_queue):
            graph_batch = graph_batch.to(self.device)

            if self.model_config["model"] == "gnn_vs_gae_classifier":
                pred_bins, pred = self.model(graph_batch=graph_batch)
                criterion = torch.nn.BCELoss()
                criterion_2 = torch.nn.MSELoss()

                bins = self.create_bins(lower_bound=0, width=10, quantity=9)
                binned_weights = []
                for value in graph_batch.y.cpu().numpy():
                    bin_index = self.find_bin(value, bins)
                    binned_weights.append(bin_index)
                bins = torch.FloatTensor(binned_weights)
                make_one_hot = lambda index: torch.eye(self.model_config["no_bins"])[
                    index.view(-1).long()
                ]
                binns_one_hot = make_one_hot(bins).to(self.device)

                loss_1 = criterion(pred_bins, binns_one_hot)
                loss_2 = criterion_2(pred, self.normalize_data(graph_batch.y))
                alpha = self.model_config["classification_loss"]
                beta = self.model_config["regression_loss"]

                loss = alpha * loss_1 + beta * loss_2

            else:
                pred = self.model(graph_batch=graph_batch)
                loss = criterion(pred, self.normalize_data(graph_batch.y))

            preds.extend(self.unnormalize_data(pred.detach().cpu().numpy()))
            targets.extend(graph_batch.y.detach().cpu().numpy())
            n = graph_batch.num_graphs
            objs.update(loss.data.item(), n)

            if step % self.data_config["report_freq"] == 0:
                logging.info("valid %03d %e ", step, objs.avg)

        fig = utils.scatter_plot(
            np.array(preds),
            np.array(targets),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_valid_{}.jpg".format(epoch)))
        plt.close()

        val_results = utils.evaluate_metrics(
            np.array(targets), np.array(preds), prediction_is_first_arg=False
        )

        return objs.avg, val_results

    def test(self):
        preds = []
        targets = []
        self.model.eval()

        test_queue = self.load_results_from_result_paths(self.test_paths)
        for step, graph_batch in enumerate(test_queue):
            graph_batch = graph_batch.to(self.device)

            if self.model_config["model"] == "gnn_vs_gae_classifier":
                pred_bins, pred = self.model(graph_batch=graph_batch)

            else:
                pred = self.model(graph_batch=graph_batch)

            preds.extend(self.unnormalize_data(pred.detach().cpu().numpy()))
            targets.extend(graph_batch.y.detach().cpu().numpy())

        fig = utils.scatter_plot(
            np.array(preds),
            np.array(targets),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_test.jpg"))
        plt.close()

        test_results = utils.evaluate_metrics(
            np.array(targets), np.array(preds), prediction_is_first_arg=False
        )
        logging.info("test metrics %s", test_results)

        return test_results

    def validate(self):
        preds = []
        targets = []
        self.model.eval()

        valid_queue = self.load_results_from_result_paths(self.val_paths)
        for step, graph_batch in enumerate(valid_queue):
            graph_batch = graph_batch.to(self.device)

            pred = self.model(graph_batch=graph_batch)
            preds.extend(self.unnormalize_data(pred.detach().cpu().numpy()))
            targets.extend(graph_batch.y.detach().cpu().numpy())

        fig = utils.scatter_plot(
            np.array(preds),
            np.array(targets),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_valid.jpg"))
        plt.close()

        val_results = utils.evaluate_metrics(
            np.array(targets), np.array(preds), prediction_is_first_arg=False
        )
        logging.info("validation metrics %s", val_results)

        return val_results

    def save(self):
        torch.save(
            self.model.state_dict(), os.path.join(self.log_dir, "surrogate_model.model")
        )

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def evaluate(self, result_paths):
        # Get evaluation data
        eval_queue = self.load_results_from_result_paths(result_paths)

        preds = []
        targets = []
        self.model.eval()
        for step, graph_batch in enumerate(eval_queue):
            graph_batch = graph_batch.to(self.device)

            pred = self.model(graph_batch=graph_batch)
            preds.extend(self.unnormalize_data(pred.detach().cpu().numpy()))
            targets.extend(graph_batch.y.detach().cpu().numpy())

        test_metrics = utils.evaluate_metrics(
            np.array(targets), np.array(preds), prediction_is_first_arg=False
        )
        return test_metrics, preds, targets

    def query(self, config_dict):
        # Get evaluation data
        config_space_instance = self.config_loader.query_config_dict(config_dict)
        dataset = NASBenchDataset(
            root="None",
            model_config=self.model_config,
            dataset_statistic=self.dataset_statistic,
            result_paths=None,
            config_loader=self.config_loader,
        )
        data_ptg = dataset.config_space_instance_to_pytorch_geometric_instance(
            config_space_instance
        )
        single_item_batch = Batch.from_data_list([data_ptg])

        self.model.eval()
        single_item_batch = single_item_batch.to(self.device)

        if self.model_config["model"] == "gnn_vs_gae_classifier":
            pred_bin, pred_normalized = self.model(graph_batch=single_item_batch)
        else:
            pred_normalized = self.model(graph_batch=single_item_batch)
        pred = self.unnormalize_data(pred_normalized.detach().cpu().numpy())

        return pred
