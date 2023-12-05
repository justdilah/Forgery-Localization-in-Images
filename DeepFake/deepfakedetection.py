import cv2 as io
import glob
import os
import random
import scipy.signal as sp
import scipy.ndimage as ndi
import skimage
from skimage import color
from skimage.feature import CENSURE, blob_dog
from skimage.restoration import estimate_sigma
from skimage.util import img_as_float, random_noise

import TamperingFunctions as tf
import HelperFunctions as hp
import math
import numpy as np
import pandas as pd
import re


def extract_features(img):
    # gray_img = color.rgb2gray(img)
    gray_img = io.cvtColor(img, io.COLOR_BGR2GRAY)
    median_mat = io.medianBlur(gray_img, ksize=5)

    # median_mat = tf.getMedianFilteredMatrix_win3x3(mat)
    # median_mat = cv2.medianBlur(mat, 3)
    gray_img = np.subtract(gray_img, median_mat)
    # Variance Feature Extraction
    variance = np.var(gray_img)
    # Entropy Feature Extraction
    entropy = skimage.measure.shannon_entropy(img)
    # Wrapped Feature Extraction
    image_wrapped = np.angle(np.exp(1j * img))
    max_val = np.max(image_wrapped)
    min_val = np.min(image_wrapped)
    wrapped_range = max_val - min_val

    # Noise Feature Extraction
    astro = img_as_float(img)
    astro = astro[30:180, 150:300]
    sigma = 0.08

    if len(astro) != 0:
        noisy = random_noise(astro, var=sigma ** 2)
        sigma_est = np.mean(estimate_sigma(noisy))
        detector = CENSURE()
        detector.detect(gray_img)
        keypoints_detector = len(detector.keypoints)
        # Blob Dog Feature Extraction
        blobs_dog = len(blob_dog(gray_img, max_sigma=1, threshold=.1))
    else:
        sigma_est = 0
        keypoints_detector = 0
        blobs_dog = 0

    # Blur Feature Extraction
    print("TEST")
    blurred_images = [ndi.uniform_filter(img, size=k) for k in range(2, 32, 2)]
    print("TEST 1")
    img_stack = np.stack(blurred_images)
    print("TEST 2")
    # # Keypoints Feature Extraction
    # detector = CENSURE()
    # print("TEST 3")
    # detector.detect(gray_img)
    # print("TEST 4")
    # # Blob Dog Feature Extraction
    # blobs_dog = blob_dog(gray_img, max_sigma=1, threshold=.1)
    # print("TEST 5")
    # print("WHYYYYY")
    features = {
        'variance': variance,
        'entropy': entropy,
        'wrapped': wrapped_range,
        'noise': sigma_est,
        'blur': np.mean(img_stack),
        'keypoints': keypoints_detector,
        'blobs': blobs_dog
    }
    return features


def get_images_and_labels(folder, label):
    variance = []
    entropy = []
    wrapped = []
    noise = []
    blur = []
    keypoints = []
    blobs = []
    labels = []
    for filename in os.listdir(folder):
        if not filename.endswith(".jpg"):
            continue
        try:
            # print(subfolder_path)
            img = io.imread(os.path.join(folder, filename))
            features = extract_features(img)
            variance.append(features['variance'])
            entropy.append(features['entropy'])
            wrapped.append(features['wrapped'])
            noise.append(features['noise'])
            blur.append(features['blur'])
            keypoints.append(features['keypoints'])
            blobs.append(features['blobs'])
            labels.append(label)
        except Exception as e:
            print("Skipping file", filename)
            print("Error:", e)
    return variance, entropy, wrapped, noise, blur, keypoints, blobs, labels

# fake_variance, fake_entropy, fake_wrapped, fake_noise, fake_blur, fake_keypoints, fake_blobs, fake_labels = get_images_and_labels(
#     r"C:\Users\ASUS\PycharmProjects\Forgery-Localization-in-Images\FaceParts\Fake\RightEye", "fake")
# real_variance, real_entropy, real_wrapped, real_noise, real_blur, real_keypoints, real_blobs, real_labels = get_images_and_labels(
#     r"C:\Users\ASUS\PycharmProjects\Forgery-Localization-in-Images\FaceParts\Real\RightEye", "real")
# # # Creating data frame
# all_data = pd.DataFrame({
#     "Variance": fake_variance + real_variance,
#     "Entropy": fake_entropy + real_entropy,
#     "Wrapped": fake_wrapped + real_wrapped,
#     "Noise": fake_noise + real_noise,
#     "Blur": fake_blur + real_blur,
#     "Keypoints": fake_keypoints + real_keypoints,
#     "Blobs": fake_blobs + real_blobs,
#     "Label": fake_labels + real_labels
# })
# # Save the data to a csv file
# all_data.to_csv("all_data_features_RightEye_MFR.csv")
# print(all_data.head())
