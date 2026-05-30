"""
Split-CIFAR-100 continual-learning benchmark.

We split the 100 classes into `n_experiences` disjoint tasks (default 10 tasks
of 10 classes each), the standard Class-Incremental / Task-Incremental setup
(van de Ven & Tolias, 2019).

Swap-in note: ContinualAI/avalanche provides SplitCIFAR100 with richer plumbing;
this minimal loader keeps the experiment dependency-free and transparent.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset


def make_split_cifar100(
    root: str = "./data",
    n_experiences: int = 10,
    batch_size: int = 128,
    seed: int = 0,
    download: bool = True,
):
    """Returns (train_loaders, test_loaders, class_groups).

    Each task i exposes classes class_groups[i]. Labels are kept in the global
    0..99 space (use a task mask at eval time for task-incremental scoring).
    """
    from torchvision import datasets, transforms

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train = datasets.CIFAR100(root, train=True, download=download, transform=tf_train)
    test = datasets.CIFAR100(root, train=False, download=download, transform=tf_test)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(100, generator=g).tolist()
    per_task = 100 // n_experiences
    class_groups = [perm[i * per_task:(i + 1) * per_task] for i in range(n_experiences)]

    def indices_for(dataset, classes):
        cset = set(classes)
        targets = dataset.targets if hasattr(dataset, "targets") else [y for _, y in dataset]
        return [i for i, t in enumerate(targets) if int(t) in cset]

    train_loaders, test_loaders = [], []
    for classes in class_groups:
        tr = Subset(train, indices_for(train, classes))
        te = Subset(test, indices_for(test, classes))
        train_loaders.append(DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2))
        test_loaders.append(DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2))

    return train_loaders, test_loaders, class_groups
