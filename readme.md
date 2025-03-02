# Surface Defect-detection and Localization using ResNet and Classical Computer Vision

To use this repository, please follow these steps:

- Clone this repository or download as a zip
- Make sure your system has Anaconda installed. Open a terminal to the root directory and enter the following command:
`conda env create -f environment.yml`
This will create a conda environment with the required libraries.
- After the required libraries have been installed, type `conda activate defect_detection` in your terminal to activate the newly created environment.

## Dataset

The dataset used for this project is the [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database). After downloading, please move the train and validation folders to `root/data`. The details of the dataset can be found on Kaggle.

## Training

The provided notebook can train, validate and visualize, given the dataset is in the correct directory. To train, please uncomment the training block in the notebook.

## Project Overview

In this project, we
1. Detect the type of defect.
2. Build a method of localizing the defects on the images. The localization method depends on the defect-type.

Step 1 consists of fine-tuning a ResNet-18 to detect defect types. The ResNet-18 works reasonable well as shown by the confusion matrix below.

<p align="center">
  <img src="confusion_matrix.png"/>
</p>

Step 2 in currently under progress. At the time of writing this document, methods for localizing defects have been developed for only 2 of the 6 defect types: patches and scratches.

Steps 1 and 2 for 15 random images from the validation set are shown below.

<p align="center">
  <img src="defects.png"/>
</p>

The boxes in images are generated using the annotations provided by the dataset - they were not determined as a part of this project. The defective regions are shown for the patches and scratches.

## Defect-localization Methods

### Crazing

Not yet implemented.

### Inclusion

Not yet implemented.

### Patches

The patches are determined using active contours. For simplicity, we use the Chan-Vese active contours from [scikit-image](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_chan_vese.html).

### Pitted Surface

Not yet implemented.

### Rolled-in Scale

Not yet implemented.

### Scratches

Given the high-contrast nature of the images, the scratches can be adequately localized by simply using an edge-detector. This work uses the Canny edge detector.