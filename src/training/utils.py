import xml.etree.ElementTree as ET
import numpy as np
import torch
import cv2

def extractBoxes(filename):        
    # load and parse the file
    tree = ET.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = (ymin, ymax, xmin, xmax)
        boxes.append(coors)
        
    return boxes

def overlayBoundingBox(image, boxes):
    image = np.array(image)
    thickness = 2
    color = int(image.max())
    for box in boxes:
        start_point = (box[2], box[0])
        end_point = (box[3], box[1])
 
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

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