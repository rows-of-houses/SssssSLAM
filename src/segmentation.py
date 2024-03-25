import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, transform, feature, color
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, warp
from skimage.color import rgb2gray
import imageio
import argparse
import imutils
from tqdm import tqdm
from IPython.display import Image
import torchvision.models.segmentation as segmentation
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import torch.nn.functional as F
import os

class FeatureExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        self.segmentation_model = deeplabv3_resnet101(pretrained=True).eval()
    def preprocess_image(self, image):
        """
        Preprocess the image for segmentation model.
        """
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    def segment_image(self, image):
        """
        Perform semantic segmentation on the image without preprocessing.
        """
        input_tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()
        input_tensor = input_tensor / 255

        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out'][0]
        output_predictions = output.argmax(0)

        mask = torch.nn.functional.interpolate(output_predictions.unsqueeze(0).unsqueeze(0).float(), 
                                               size=image.shape[:2], mode='nearest').byte().squeeze().cpu().numpy()
        stable_classes = [15]  # class ID for human
        stable_mask = np.isin(mask, stable_classes).astype(np.uint8)

        return stable_mask

    def extract_features(self, image, method='ORB'):
        if method.upper() == 'SIFT':
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
        elif method.upper() == 'ORB':
            keypoints, descriptors = self.orb.detectAndCompute(image, None)
        else:
            raise ValueError("Unsupported feature extraction method: {}".format(method))

        return keypoints, descriptors

    def visualize_features(self, image, keypoints):
        return cv2.drawKeypoints(image, keypoints, None)
    
    def extract_stable_features(self, image, method='ORB'):
        mask = self.segment_image(image) # human mask 
        inv_mask = np.ones(mask.shape) - mask 
        inv_mask = inv_mask.astype('uint8') # background without human
        stable_regions = cv2.bitwise_and(image, image, mask=inv_mask)
        keypoints_stable, descriptors_stable = self.extract_features(stable_regions, method)

        return keypoints_stable, descriptors_stable
    
    def process_image(self, image, method='ORB'):
        mask = self.segment_image(image)
        stable_regions = cv2.bitwise_and(image, image, mask=mask)
        keypoints_stable, descriptors_stable = self.extract_features(stable_regions, method)

        return keypoints_stable, descriptors_stable