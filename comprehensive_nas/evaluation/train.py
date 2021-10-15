import numpy as np
import torch
import torchmetrics


def general_num_params(m):
    # return number of differential parameters of input model
    return sum(
        np.prod(p.size()) for p in filter(lambda p: p.requires_grad, m.parameters())
    )


def reset_weights(model):
    warn_non_resets = []
    diff_non_resets = []
    for module in model.modules():
        if type(module) != type(model):  # pylint: disable=C0123
            if "reset_parameters" in dir(module):
                module.reset_parameters()
            else:
                if "parameters" in dir(module):
                    n_params = general_num_params(module)
                    child_params = sum(general_num_params(m) for m in module.children())

                    if n_params != 0 and n_params != child_params:
                        diff_non_resets.append([type(module).__name__, n_params])
                else:
                    warn_non_resets.append(type(module).__name__)


def train(model, device, optimizer, criterion, loader, **train_args):
    model.train()
    grad_clip = train_args["grad_clip"] if "grad_clip" in train_args else None
    for data, target in loader:
        data, target = data.to(device), target.to(device)
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
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        metric.update(output, target)
    return metric.compute().cpu().item()


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
    best_test_score = 0
    best_epoch = 0
    for epoch in range(n_epochs):
        train(
            model=model,
            device=device,
            optimizer=optimizer,
            criterion=train_criterion,
            loader=train_loader,
            **train_args,
        )
        valid_score = evaluate(
            model=model,
            device=device,
            metric=evaluation_metric,
            loader=valid_loader,
        )
        test_score = evaluate(
            model=model,
            device=device,
            metric=evaluation_metric,
            loader=test_loader,
        )
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_test_score = test_score
            best_epoch = epoch
        scheduler.step()

    return {
        "best_epoch": best_epoch,
        "best_val_score": best_valid_score.item(),
        "best_test_score": best_test_score.item(),
    }


def training_pipeline(
    model: torch.nn.Module,
    train_criterion,
    evaluation_metric: torchmetrics.Metric,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    **train_args,
):
    reset_weights(model)
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
    return 1 - results["best_val_score"], 1 - results["best_test_score"]


if __name__ == "__main__":
    import argparse

    import torchvision.models as models

    from comprehensive_nas.evaluation.utils import (
        get_evaluation_metric,
        get_loss,
        get_optimizer,
        get_scheduler,
        get_train_val_test_loaders,
    )

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
