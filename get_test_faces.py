import cv2
from mtcnn import MTCNN
import os
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Dependencies
# OpenCV
# mtcnn
# tensorflow

# Isolate the faces and put in a separate folder
directory = 'test'
device = torch.device('cuda')
idx = 0
detector = MTCNN()
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        print(idx)
        image = cv2.imread(str(directory + '/' + str(idx) + '.jpg'))
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)
        
        
    # Isolate just the face from the image
    newsize = (180,180)
    try:
        face = detector.detect_faces(image)
        x, y, width, height = face[0]['box']
    
        # Crop the original image to a standard size
        face_image = image[y:y+height, x:x+width]
        face_image = cv2.resize(face_image, newsize)
    except:
        try:
            face_image = cv2.resize(image, newsize)
        except:
            face_image = face_im_prev
        
    cv2.imwrite(str('test_faces/' + str(idx) + '.jpg'), face_image)
    idx = idx + 1
    face_im_prev = face_image

        