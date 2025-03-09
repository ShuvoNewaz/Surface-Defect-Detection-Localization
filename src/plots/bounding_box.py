import xml.etree.ElementTree as ET
import numpy as np
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