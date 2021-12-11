import numpy as np
import torch
import torchmetrics


def general_num_params(m):
    # return number of differential parameters of input model
    return sum(
        np.prod(p.size()) for p in filter(lambda p: p.requires_grad, m.parameters())
    )


def reset_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def train(model, device, optimizer, criterion, loader, **train_args):
    model.train()
    grad_clip = train_args["grad_clip"] if "grad_clip" in train_args else None
    for data_blob in loader:
        data, target = (x.to(device) for x in data_blob)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


@torch.no_grad()
def evaluate(model, device, metric, loader):
    model.eval()
    metric.reset()
    for data_blob in loader:
        data, target = (x.to(device) for x in data_blob)
        output = model.forward(data)
        metric.update(output, target)
    return metric.compute().detach().cpu().item()


def run_training(
    model,
    train_criterion,
    evaluation_metric,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    test_loader,
    n_epochs,
    device,
    **train_args,
):
    model.to(device)
    evaluation_metric.to(device)
    best_valid_score = 0
    best_epoch = 0
    valid_scores = []
    test_scores = []
    for epoch in range(n_epochs):
        train(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=train_criterion,
            loader=train_loader,
            **train_args,
        )
        if valid_loader is not None:
            valid_score = evaluate(
                model=model,
                device=device,
                metric=evaluation_metric,
                loader=valid_loader,
            )
            valid_scores.append(valid_score)
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_epoch = epoch
        if test_loader is not None:
            test_score = evaluate(
                model=model,
                device=device,
                metric=evaluation_metric,
                loader=test_loader,
            )
            test_scores.append(test_score)

        scheduler.step()

    model.cpu()
    evaluation_metric.cpu()
    for optimizer_metrics in optimizer.state.values():
        for metric_name, metric in optimizer_metrics.items():
            if torch.is_tensor(metric):
                optimizer_metrics[metric_name] = metric.cpu()

    ret_val = {"best_epoch": best_epoch}
    if valid_loader is not None:
        ret_val["val_scores"] = valid_scores
        ret_val["best_val_score"] = best_valid_score
    if test_loader is not None:
        ret_val["test_scores"] = test_scores
        ret_val["best_test_score"] = test_scores[best_epoch]
    return ret_val


def training_pipeline(
    model: torch.nn.Module,
    train_criterion,
    evaluation_metric: torchmetrics.Metric,
    optimizer: torch.optim.Optimizer,
    scheduler,
    n_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader = None,
    test_loader: torch.utils.data.DataLoader = None,
    **train_args,
) -> dict:
    """General training pipeline.

    Args:
        model (torch.nn.Module): PyTorch model.
        train_criterion: Loss function.
        evaluation_metric (torchmetrics.Metric): Metric for outer loop. Needs to be maximized!
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler: Learning rate scheduler.
        n_epochs (int): Number of epochs.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        valid_loader (torch.utils.data.DataLoader, optional): Validation data loader. Defaults to None.
        test_loader (torch.utils.data.DataLoader, optional): Test data loader. Defaults to None.

    Returns:
        dict: Dictionary with results. If there is a valid_loader, there will be validation scores per epoch.
        Additionally, there will be an index indicating the epoch with highest validation score.
        Similarly, if there is a test_loader, there will be test scores per epoch. Note
        that there will be no best epoch index if no validation data is provided.
    """
    model.apply(reset_weights)
    results = run_training(
        model=model,
        train_criterion=train_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
        evaluation_metric=evaluation_metric,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        **train_args,
    )
    return results


if __name__ == "__main__":
    import argparse

    import torchvision.models as models

    # pylint: disable=C0412
    from neps.evaluation.utils import (
        get_evaluation_metric,
        get_loss,
        get_optimizer,
        get_scheduler,
        get_train_val_test_loaders,
    )

    # pylint: enable=C0412

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--dataset", help="Dataset to select.", required=True)
    parser.add_argument(
        "--data_path",
        default="",
        help="Path to folder with data or where data should be saved to if downloaded.",
    )
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    n_epochs = args.epochs
    batch_size = args.batch_size
    num_classes = args.num_classes
    seed = args.seed

    model = models.resnet18(num_classes=num_classes)
    train_criterion = get_loss("CrossEntropyLoss")
    evaluation_metric = get_evaluation_metric("Accuracy", top_k=1)
    optimizer = get_optimizer("SGD", model, lr=0.01, momentum=0.9, weight_decay=3e-4)
    scheduler = get_scheduler(
        scheduler="CosineAnnealingLR", optimizer=optimizer, T_max=n_epochs
    )
    train_loader, valid_loader, test_loader = get_train_val_test_loaders(
        dataset=args.dataset,
        data=args.data_path,
        batch_size=batch_size,
        seed=seed,
    )

    res = training_pipeline(
        model=model,
        train_criterion=train_criterion,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        n_epochs=n_epochs,
    )
    print(res)
