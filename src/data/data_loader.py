import os
import torch
import numpy as np
from torchvision.transforms import Compose
from torch.utils import data
from typing import List, Tuple
from PIL import Image
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle
# from src.data_transforms import *


class ImageLoader(data.Dataset):
    def __init__(self, dataDir, split, transform):
        """
            args:
                dataDir: Directory of the dataset w.r.t. the root directory
                split: training or validation split
        """
        splitDir = os.path.join(dataDir, split)
        self.imDir = os.path.join(splitDir, "images")
        self.annotationDir = os.path.join(splitDir, "annotations")
        labelNames = os.listdir(self.imDir)
        labelNames.sort()
        labelIndices = list(range(len(labelNames)))
        self.labelInfo = dict(zip(labelNames, labelIndices))
        self.reverseLabelInfo = dict(zip(labelIndices, labelNames))
        self.dataset = self.load_imagepaths_with_labels()
        self.transform = transform

    def load_imagepaths_with_labels(self):
        imagePaths = []
        for defectType in os.listdir(self.imDir):
            typeDir = os.path.join(self.imDir, defectType)
            imageNames = os.listdir(typeDir)
            imageNames.sort(key=lambda x: x.split("_")[1])
            for imageName in imageNames:
                imageNumber = imageName.split(".")[0]
                imageDir = os.path.join(typeDir, imageName)
                annDir = os.path.join(self.annotationDir, f"{imageNumber}.xml")
                imagePaths.append((imageDir, annDir, self.labelInfo[defectType]))
        return imagePaths
    
    def load_img_from_path(self, path: str) -> Image:
        """
            Loads an image as grayscale (using Pillow).
            Note: do not normalize the image to [0,1]
            Args:
                paths: Tuple containing the path to image and annotation.
            Returns:
                image: grayscale image with values in [0,255] loaded using pillow
                    Note: Use 'L' flag while converting using Pillow's function.
        """
        image = Image.open(path).convert(mode='L')

        return image
    
    def __getitem__(self, index:int):
        imageDir, annDir, label = self.dataset[index]
        image = self.load_img_from_path(imageDir)
        image = self.transform(image)

        return image, label, imageDir, annDir # the directories are stored for later visualization
    
    def __len__(self):
        
        return len(self.dataset)