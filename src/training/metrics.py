import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K classes logits[k] (where 0 <= k < K) corresponds to the
                log-odds of class `k` being the correct one.
                Shape: (batch_size, num_classes)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size)
    Returns:
        accuracy: The accuracy of the predicted logits
                   (number of correct predictions / total number of examples)
    """
    prediction = torch.argmax(logits, dim=1)
    batch_accuracy = torch.sum(prediction == labels) / len(labels)
    batch_accuracy = batch_accuracy.cpu().item()

    return batch_accuracy


def compute_loss(
    model,
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    """
    loss = model.loss_criterion(model_output, target_labels)
    if is_normalize:
        loss /= len(model_output)

    return loss