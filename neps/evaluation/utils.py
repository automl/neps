import os
from copy import deepcopy

import numpy as np
import torch
import torchmetrics
import torchvision.datasets as dset
from torchvision import transforms


class Cutout:
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img


class HelperDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform):
        self.x = x
        self.y = y
        self.transform = transform
        self.N = x.shape[0]

    def __getitem__(self, index):
        xi = self.x[index]
        yi = self.y[index]

        if self.transform is not None:
            xi = self.transform(xi)

        return xi, yi

    def __len__(self):
        return self.N


def _load_npy_data(dataset, path):
    def normalize(A):
        minimum = A.min((1, 2, 3))
        maximum = A.max((1, 2, 3))
        return ((A.transpose((1, 2, 3, 0)) - minimum) / (maximum - minimum)).transpose(
            (3, 0, 1, 2)
        )

    if not os.path.isdir(path):
        raise ValueError(f"Path {path} is not a valid directory")

    if dataset == "addNIST":
        train_x = np.load(os.path.join(path, "train_x.npy")).transpose((0, 2, 3, 1))
        train_y = np.load(os.path.join(path, "train_y.npy"))
        valid_x = np.load(os.path.join(path, "valid_x.npy")).transpose((0, 2, 3, 1))
        valid_y = np.load(os.path.join(path, "valid_y.npy"))
        test_x = np.load(os.path.join(path, "test_x.npy")).transpose((0, 2, 3, 1))
        test_y = np.load(os.path.join(path, "test_y.npy"))
        train_x = normalize(train_x)
        valid_x = normalize(valid_x)
        test_x = normalize(test_x)
        return train_x, train_y, valid_x, valid_y, test_x, test_y
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _get_transforms(dataset, args=None):
    if dataset == "fashionMNIST":
        FMNIST_MEAN = [0.2860]
        FMNIST_STD = [0.3202]
        train_transform = transforms.Compose(
            [
                # transforms.RandomCrop(28, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
            ]
        )
        if args is not None and "cutout" in args and args.cutout:
            train_transform.transforms.append(
                Cutout(args.cutout_length, args.cutout_prob)
            )
        valid_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(FMNIST_MEAN, FMNIST_STD)]
        )
    elif dataset == "cifar10":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        if args is not None and "cutout" in args and args.cutout:
            train_transform.transforms.append(
                Cutout(args.cutout_length, args.cutout_prob)
            )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif dataset == "cifar100":
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        if args is not None and "cutout" in args and args.cutout:
            train_transform.transforms.append(
                Cutout(args.cutout_length, args.cutout_prob)
            )

        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif dataset == "ImageNet16-120":
        IMAGENET16_MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
        IMAGENET16_STD = [x / 255 for x in [63.22, 61.26, 65.09]]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(16, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
            ]
        )
        if args.cutout:
            train_transform.transforms.append(
                Cutout(args.cutout_length, args.cutout_prob)
            )

        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
            ]
        )
    elif dataset == "svhn":
        SVHN_MEAN = [0.4377, 0.4438, 0.4728]
        SVHN_STD = [0.1980, 0.2010, 0.1970]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ]
        )
        if args is not None and "cutout" in args and args.cutout:
            train_transform.transforms.append(
                Cutout(args.cutout_length, args.cutout_prob)
            )

        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(SVHN_MEAN, SVHN_STD),
            ]
        )
    elif dataset == "addNIST":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_transform, valid_transform


def get_train_val_test_loaders(
    dataset: str,
    data,
    batch_size: int,
    train_portion: float = 0.8,
    num_workers: int = 8,
    eval_mode: bool = False,
):
    train_transform, valid_transform = _get_transforms(dataset)
    if dataset == "fashionMNIST":
        train_data = dset.FashionMNIST(
            root=data, download=True, transform=train_transform
        )
        test_data = dset.FashionMNIST(
            root=data, download=True, train=False, transform=valid_transform
        )
    elif dataset == "cifar10":
        train_data = dset.CIFAR10(
            root=data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "cifar100":
        train_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "svhn":
        train_data = dset.SVHN(
            root=data, split="train", download=True, transform=train_transform
        )
        test_data = dset.SVHN(
            root=data, split="test", download=True, transform=valid_transform
        )
    elif dataset == "ImageNet16-120":
        raise NotImplementedError
        # from naslib.utils.DownsampledImageNet import ImageNet16

        # data_folder = f"{data}/{dataset}"
        # train_data = ImageNet16(
        #     root=data_folder,
        #     train=True,
        #     transform=train_transform,
        #     use_num_of_class_only=120,
        # )
        # test_data = ImageNet16(
        #     root=data_folder,
        #     train=False,
        #     transform=valid_transform,
        #     use_num_of_class_only=120,
        # )
    elif dataset == "addNIST":
        train_x, train_y, valid_x, valid_y, test_x, test_y = _load_npy_data(dataset, data)
        train_dataset = HelperDataset(train_x, train_y, train_transform)
        valid_dataset = HelperDataset(valid_x, valid_y, valid_transform)
        test_dataset = HelperDataset(test_x, test_y, valid_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=int(batch_size),
            num_workers=num_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=int(batch_size),
            num_workers=num_workers,
        )
        return train_loader, valid_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if eval_mode:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=num_workers,
        )
        valid_loader = None
    else:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(train_portion * num_train))
        valid_data = deepcopy(train_data)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=False,
            num_workers=num_workers,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]
            ),
            pin_memory=False,
            num_workers=num_workers,
        )
        valid_loader.dataset.transform = valid_transform

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader


def get_optimizer(optimizer: str, model: torch.nn.Module, **optim_kwargs):
    if optimizer not in dir(torch.optim):
        raise ValueError(f"PyTorch has not implemented optimizer {optimizer} yet!")
    return getattr(torch.optim, optimizer)(model.parameters(), **optim_kwargs)


def get_scheduler(scheduler: str, optimizer, **scheduler_kwargs):
    if scheduler not in dir(torch.optim.lr_scheduler):
        raise ValueError(f"PyTorch has not implemented scheduler {scheduler} yet!")
    return getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **scheduler_kwargs)


def get_loss(loss_function: str, **loss_kwargs):
    if loss_function not in dir(torch.nn):
        raise ValueError(
            f"PyTorch has not implemented loss function {loss_function} yet!"
        )
    return getattr(torch.nn, loss_function)(**loss_kwargs)


def get_evaluation_metric(metric: str, **metrics_kwargs):
    if metric not in dir(torchmetrics):
        raise ValueError(f"Torchmetrics has not implemented metric {metric} yet!")
    return getattr(torchmetrics, metric)(**metrics_kwargs)
