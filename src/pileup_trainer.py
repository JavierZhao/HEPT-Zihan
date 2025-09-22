import nni
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import sys
from io import TextIOBase
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from utils import set_seed, get_optimizer, log, get_lr_scheduler, get_loss
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


class Tee(TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class FilteredTee(TextIOBase):
    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream
        self._buffer = ""

    def write(self, s):
        # Console (keep tqdm)
        self.console.write(s)
        self.console.flush()

        # Buffer and write full lines to file, collapsing carriage returns
        self._buffer += s
        written = len(s)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if "\r" in line:
                line = line.split("\r")[-1]
            if not line:
                continue
            line_no_ansi = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
            self.file.write(line_no_ansi + "\n")
            self.file.flush()
        return written

    def flush(self):
        self.console.flush()
        self.file.flush()


def train_one_batch(model, optimizer, criterion, data, lr_s):
    model.train()
    embeddings = model(data)
    loss = criterion(embeddings[data.is_neu], data.y[data.is_neu].unsqueeze(-1).float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_s is not None and isinstance(lr_s, LambdaLR):
        lr_s.step()
    return loss.item(), embeddings.detach()


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s):
    model.eval()
    embeddings = model(data)
    loss = criterion(embeddings[data.is_neu], data.y[data.is_neu].unsqueeze(-1).float())
    return loss.item(), embeddings.detach()


def run_one_epoch(
    model, optimizer, criterion, data_loader, phase, epoch, device, metrics, lr_s
):
    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test " if phase == "test" else phase

    pbar = tqdm(data_loader, disable=__name__ != "__main__")
    for idx, data in enumerate(pbar):
        data = data.to(device)
        batch_loss, batch_embeddings = run_one_batch(
            model, optimizer, criterion, data, lr_s
        )
        batch_auc = update_metrics(metrics, data, batch_embeddings)
        metrics["loss"].update(batch_loss)

        desc = f"[Epoch {epoch}] {phase}, loss: {batch_loss:.4f}, auc: {batch_auc:.4f}"
        if idx == len(data_loader) - 1:
            metric_res = compute_metrics(metrics)
            loss, auc, f1, roc = (
                metric_res["loss"],
                metric_res["auc"],
                metric_res["f1"],
                metric_res["roc"],
            )
            desc = f"[Epoch {epoch}] {phase}, loss: {loss:.5f}, auc: {auc:.4f}, f1: {f1:.4f}, roc: {roc:.4f}"
            reset_metrics(metrics)
            # Force newline so FilteredTee captures final tqdm line
            print("\n", end="")
        pbar.set_description(desc)
    return metric_res


def reset_metrics(metrics):
    for metric in metrics.values():
        if isinstance(metric, MeanMetric):
            metric.reset()


def compute_metrics(metrics):
    return {
        f"{name}": metrics[f"{name}"].compute().item() for name in ["auc", "f1", "roc"]
    } | {"loss": metrics["loss"].compute().item()}


def update_metrics(metrics, data, embeddings):
    pred = (embeddings > 0.5).int()[data.is_neu].cpu()
    label = data.y[data.is_neu].cpu()
    embeddings = embeddings[data.is_neu].cpu()

    auc = average_precision_score(label, embeddings)
    roc = roc_auc_score(label, embeddings)
    f1 = f1_score(label, pred)

    metrics["auc"].update(auc)
    metrics["f1"].update(f1)
    metrics["roc"].update(roc)
    return auc


def run_one_seed(config, tune=False):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["num_threads"])

    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / dataset_name
    log(
        f"Device: {device}, Model: {model_name}, Dataset: {dataset_name}, Note: {config['note']}"
    )

    time = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand_num = np.random.randint(10, 100)
    base_logs_dir = (
        Path(config.get("out_dir"))
        if config.get("out_dir") is not None
        else (dataset_dir / "logs")
    )
    sort_type = config.get("model_kwargs", {}).get("sort_type", "none")
    sort_suffix = f"_{sort_type}" if sort_type != "none" else ""
    if sort_type == "morton":
        sort_bits = config.get("model_kwargs", {}).get("morton_bits", 10)
        sort_suffix += f"-mb{sort_bits}"
    log_dir = base_logs_dir / f"{time}_{model_name}_{sort_suffix}_{rand_num}"
    log(f"Log dir: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=False)
    # Logging tee
    log_file = (log_dir / "train.log").open("a")
    sys.stdout = FilteredTee(sys.stdout, log_file)
    sys.stderr = FilteredTee(sys.stderr, log_file)
    # Save config snapshot
    with (log_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    writer = SummaryWriter(log_dir) if config["log_tensorboard"] else None

    set_seed(config["seed"])
    dataset = get_dataset(
        dataset_name, dataset_dir, data_suffix=config.get("data_suffix", None)
    )
    loaders = get_data_loader(
        dataset, dataset.idx_split, batch_size=config["batch_size"]
    )

    model = get_model(model_name, config["model_kwargs"], dataset)
    # Log model architecture and parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters (trainable): {num_params}")
    log("Model architecture:\n" + str(model))
    if config.get("only_flops", False):
        raise RuntimeError
    if config.get("resume", False):
        log(f"Resume from {config['resume']}")
        model_path = dataset_dir / "logs" / (config["resume"] + "/best_model.pt")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

    opt = get_optimizer(
        model.parameters(), config["optimizer_name"], config["optimizer_kwargs"]
    )
    config["lr_scheduler_kwargs"]["num_training_steps"] = config["num_epochs"] * len(
        loaders["train"]
    )
    lr_s = get_lr_scheduler(
        opt, config["lr_scheduler_name"], config["lr_scheduler_kwargs"]
    )
    criterion = get_loss(config["loss_name"], config["loss_kwargs"])

    main_metric = config["main_metric"]
    metric_names = ["auc", "f1", "roc"]
    metrics = {f"{name}": MeanMetric() for name in metric_names}
    metrics["loss"] = MeanMetric()

    coef = 1 if config["mode"] == "max" else -1
    best_epoch, best_train = 0, {
        metric: -coef * float("inf") for metric in metrics.keys()
    }
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)

    # History and persistence
    history = {
        "epoch": [],
        "train_loss": [],
        "train_auc": [],
        "valid_loss": [],
        "valid_auc": [],
        "test_loss": [],
        "test_auc": [],
    }
    metrics_npz_path = log_dir / "metrics.npz"
    curves_png_path = log_dir / "curves.png"

    if writer is not None:
        layout = {
            "Gap": {
                "loss": ["Multiline", ["train/loss", "valid/loss", "test/loss"]],
                "auc": ["Multiline", ["train/auc", "valid/auc", "test/auc"]],
            }
        }
        writer.add_custom_scalars(layout)

    for epoch in range(config["num_epochs"]):
        if not config.get("only_eval", False):
            train_res = run_one_epoch(
                model,
                opt,
                criterion,
                loaders["train"],
                "train",
                epoch,
                device,
                metrics,
                lr_s,
            )
        valid_res = run_one_epoch(
            model,
            opt,
            criterion,
            loaders["valid"],
            "valid",
            epoch,
            device,
            metrics,
            lr_s,
        )
        test_res = run_one_epoch(
            model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s
        )

        # Update history and persist NPZ
        history["epoch"].append(epoch)
        history["train_loss"].append(
            train_res["loss"] if not config.get("only_eval", False) else float("nan")
        )
        history["train_auc"].append(
            train_res.get("auc", float("nan"))
            if not config.get("only_eval", False)
            else float("nan")
        )
        history["valid_loss"].append(valid_res["loss"])
        history["valid_auc"].append(valid_res["auc"])
        history["test_loss"].append(test_res["loss"])
        history["test_auc"].append(test_res["auc"])
        try:
            np.savez(
                metrics_npz_path,
                epoch=np.array(history["epoch"], dtype=np.int32),
                train_loss=np.array(history["train_loss"], dtype=np.float32),
                train_auc=np.array(history["train_auc"], dtype=np.float32),
                valid_loss=np.array(history["valid_loss"], dtype=np.float32),
                valid_auc=np.array(history["valid_auc"], dtype=np.float32),
                test_loss=np.array(history["test_loss"], dtype=np.float32),
                test_auc=np.array(history["test_auc"], dtype=np.float32),
            )
        except Exception as e:
            log(f"Saving metrics NPZ failed: {e}")

        if lr_s is not None and isinstance(lr_s, ReduceLROnPlateau):
            lr_s.step(valid_res[config["lr_scheduler_metric"]])

        if (valid_res[main_metric] * coef) > (best_valid[main_metric] * coef):
            best_epoch, best_train, best_valid, best_test = (
                epoch,
                train_res,
                valid_res,
                test_res,
            )
            torch.save(model.state_dict(), log_dir / "best_model.pt")
            with (log_dir / "best_metrics.txt").open("w") as f:
                f.write(f"best_epoch: {best_epoch}\n")
                for k, v in best_train.items():
                    f.write(f"train/{k}: {v}\n")
                for k, v in best_valid.items():
                    f.write(f"valid/{k}: {v}\n")
                for k, v in best_test.items():
                    f.write(f"test/{k}: {v}\n")

        print(
            f"[Epoch {epoch}] Best epoch: {best_epoch}, train: {best_train[main_metric]:.4f}, "
            f"valid: {best_valid[main_metric]:.4f}, test: {best_test[main_metric]:.4f}"
        )
        print("=" * 50), print("=" * 50)

        if writer is not None:
            writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
            for phase, res in zip(
                ["train", "valid", "test"], [train_res, valid_res, test_res]
            ):
                for k, v in res.items():
                    writer.add_scalar(f"{phase}/{k}", v, epoch)
            for phase, res in zip(
                ["train", "valid", "test"], [best_train, best_valid, best_test]
            ):
                for k, v in res.items():
                    writer.add_scalar(f"best_{phase}/{k}", v, epoch)

        # Plot curves every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == config["num_epochs"] - 1:
            try:
                plt.figure(figsize=(10, 4))
                # Loss
                plt.subplot(1, 2, 1)
                plt.plot(history["epoch"], history["train_loss"], label="train")
                plt.plot(history["epoch"], history["valid_loss"], label="valid")
                plt.plot(history["epoch"], history["test_loss"], label="test")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.title("Loss")
                plt.legend()
                # AUC
                plt.subplot(1, 2, 2)
                plt.plot(history["epoch"], history["train_auc"], label="train")
                plt.plot(history["epoch"], history["valid_auc"], label="valid")
                plt.plot(history["epoch"], history["test_auc"], label="test")
                plt.xlabel("epoch")
                plt.ylabel("auc")
                plt.title("AUC")
                plt.legend()
                plt.tight_layout()
                plt.savefig(curves_png_path)
                plt.close()
            except Exception as e:
                log(f"Plotting failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train a model for pileup.")
    parser.add_argument("-m", "--model", type=str, default="gcn")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/j-jepa-vol/HEPT-Zihan/logs/pileup",
        help="Directory to write logs and checkpoints.",
    )
    parser.add_argument(
        "--sort_type",
        type=str,
        choices=["none", "dr", "morton", "lz", "kt"],
        default="none",
        help="Optional sequence sorting within batch for dense attention models.",
    )
    parser.add_argument(
        "--morton_bits",
        type=int,
        default=10,
        help="Bit depth for Morton code when sort_type=morton.",
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default=None,
        help="Optional suffix for processed data file (e.g., 'sample').",
    )
    args = parser.parse_args()

    if args.model in ["gcn", "gatedgnn", "dgcnn", "gravnet"]:
        config_dir = Path(f"./configs/pileup/pileup_gnn_{args.model}.yaml")
    else:
        config_dir = Path(f"./configs/pileup/pileup_trans_{args.model}.yaml")
    config = yaml.safe_load(config_dir.open("r").read())
    config["out_dir"] = args.out_dir
    if args.sort_type is not None:
        config.setdefault("model_kwargs", {})["sort_type"] = args.sort_type
    if args.morton_bits is not None:
        config.setdefault("model_kwargs", {})["morton_bits"] = args.morton_bits
    if args.data_suffix is not None:
        config["data_suffix"] = args.data_suffix
    run_one_seed(config)


if __name__ == "__main__":
    main()
