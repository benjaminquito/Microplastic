from __future__ import annotations

import torch


def confusion_matrix(
    targets: list[int], predictions: list[int], num_classes: int
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target][prediction] += 1
    return matrix


def classification_metrics(
    targets: list[int], predictions: list[int], classes: list[str]
) -> dict[str, object]:
    matrix = confusion_matrix(targets, predictions, len(classes))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    correct = sum(matrix[index][index] for index in range(len(classes)))

    for index, name in enumerate(classes):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(classes))) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "accuracy": correct / len(targets) if targets else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": classes,
    }


def collect_predictions(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[list[int], list[int], float]:
    criterion = torch.nn.CrossEntropyLoss()
    targets: list[int] = []
    predictions: list[int] = []
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, batch_targets).item() * inputs.size(0)
            targets.extend(batch_targets.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    mean_loss = loss_sum / len(targets) if targets else 0.0
    return targets, predictions, mean_loss
