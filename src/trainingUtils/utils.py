
import numpy as np

from collections import Counter
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import logging
import torch


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )

    train_dataset = Subset(dataset, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)

    return train_dataset, val_dataset


def __split_of_train_sequence(subset: Subset, train_split=1.0):
    if train_split == 1:
        return subset

    targets = np.array([subset.dataset.targets[i] for i in subset.indices])
    train_indices, _ = train_test_split(
        np.arange(targets.shape[0]),
        test_size=1 - train_split,
        stratify=targets
    )

    train_dataset = Subset(subset.dataset, indices=[subset.indices[i] for i in train_indices])

    return train_dataset


def __log_class_statistics(subset: Subset):
    train_classes = [subset.dataset.targets[i] for i in subset.indices]
    print(dict(Counter(train_classes)))




def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs).expand(1, -1, -1)

        loss = criterion(outputs[0], labels[0])
        loss.backward()
        # clip_grad_norm_ to prevent gradients from exploding.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        running_loss += loss.item()

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        scheduler.step(running_loss.item() / len(dataloader))
    print("Total files Train: ", str(i))
    return running_loss, pred_correct, pred_all, (pred_correct / pred_all)


def evaluate(model, dataloader, device, print_stats=False):
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs).expand(1, -1, -1)

        # Statistics
        pred = int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2)))
        #print("PRED:", str(pred), " - LABEL:", str(int(labels[0][0])))
        if pred == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1
        #print("POS EVOL: ", model.pos)
    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")
    print("Total files Validation: ", str(i+1))
    return pred_correct, pred_all, (pred_correct / pred_all)


def evaluate_top_k(model, dataloader, device, k=5):
    pred_correct, pred_all = 0, 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(inputs).expand(1, -1, -1)
        if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist():
            pred_correct += 1
        pred_all += 1
    return pred_correct, pred_all, (pred_correct / pred_all)
