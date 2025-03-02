import os
from typing import Tuple
import numpy as np
from PIL import Image


def compute_mean_and_std(rootDir: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        rootDir: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """

    count = 0
    allImage = np.zeros((200, 200))
    trainImageDir = os.path.join(rootDir, "data", "train", "images")
    for className in os.listdir(trainImageDir):
        classDir = os.path.join(trainImageDir, className)
        for imageName in os.listdir(classDir):
            imageDir = os.path.join(classDir, imageName)
            image = Image.open(imageDir).convert(mode='L')
            image = np.array(image) / 255
            allImage += image
            count += 1

    allImage /= count
    allImage = allImage.ravel()
    mean = allImage.mean()
    var = allImage.var()
    std = np.sqrt(var)

    return mean, std
