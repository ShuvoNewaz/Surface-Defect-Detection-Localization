import cv2


def findEdges(image):
    edges = cv2.Canny(image, 50, 150, None)
    image[edges != 0] = 255
    image[edges == 0] = 0
    
    return image