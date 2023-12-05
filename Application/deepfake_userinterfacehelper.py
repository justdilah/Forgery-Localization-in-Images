from tkinter import *
import tkinter as tk
from tkinter import filedialog
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve,auc
from PIL import Image, ImageTk
import scipy.signal as sp
import cv2
import numpy as np
import pandas as pd
import math
import joblib
import TamperingFunctions as tf
import HelperFunctions as hp
import os
import imutils
import re
import dlib
from imutils import face_utils
import deepfakedetection as fn
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
STORAGE = "UserStorage/"


def getMedianFilteredMatrix():
    # Read the image
    img_noisy1 = cv2.imread(filename, 0)
    m, n = img_noisy1.shape
    img_new1 = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = [img_noisy1[i - 1, j - 1],
                    img_noisy1[i - 1, j],
                    img_noisy1[i - 1, j + 1],
                    img_noisy1[i, j - 1],
                    img_noisy1[i, j],
                    img_noisy1[i, j + 1],
                    img_noisy1[i + 1, j - 1],
                    img_noisy1[i + 1, j],
                    img_noisy1[i + 1, j + 1]]

            temp = sorted(temp)
            img_new1[i, j] = temp[4]

    return img_new1.astype(np.uint8)


# returns 2D image numpy array
def convert_image_matrix(img_name):
    # convert to gray scale color
    src = cv2.imread(img_name)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # #saves the name and the extension of the image name - fruits_platter then ext - jpg
    name, ext = img_name.split('.')
    gimg_shape = img.shape
    gimg_mat = []
    for i in range(0, gimg_shape[0]):
        row = []
        for j in range(0, gimg_shape[1]):
            pixel = img.item(i, j)
            row.append(pixel)
        gimg_mat.append(row)
    gimg_mat = np.array(gimg_mat)
    return gimg_mat


def combine(orig_matrix, image_patches):
    # row = len(image_patches[0][0])
    # col = len(image_patches[0])
    r = 0
    c = 0
    ir = 0
    ic = 0

    row = len(image_patches[0][0])
    col = len(image_patches[0])

    for outer in range(0, len(image_patches)):
        image_patches[outer] = addborders(image_patches[outer], len(image_patches[outer]), len(image_patches[outer]))

        for k in range(r, row):
            for l in range(c, col):
                orig_matrix[k][l] = image_patches[outer][ir][ic]
                ic = ic + 1

            ic = 0
            ir = ir + 1
        ir = 0

        r = row
        c = col - 128
        row = row + 128
        if (r == 512):
            r = 0
            row = 128
            c = col
            col = col + 128

    return orig_matrix


def upload_file(frame,imageUploadLabel):
    f_types = (('Jpg Files', '*.jpg'),('PNG Files', '*.png'))
    global filename
    filename = filedialog.askopenfilename(filetypes=f_types)

    # need to check if it is jpg or not
    img_mat = convert_image_matrix(filename)
    global row
    global col
    row = len(img_mat[0])
    col = len(img_mat)
    filename = filename.replace(".pgm", ".jpg")
    cv2.imwrite("image_to_be_converted.jpg", img_mat)
    img = Image.open(filename)
    # Change the dimensions of the image first
    img_resized = img.resize((240, 240))  # new width & height
    img = ImageTk.PhotoImage(img_resized)
    # global labelTampered
    # label = Label(frame, text="Uploaded Image")
    # label.grid(row=0, column=1, padx=5, pady=5)
    imageUploadLabel.configure(image=img)
    imageUploadLabel.image = img  # keep a reference! by attaching it to a widget attribute
    imageUploadLabel.grid(row=1, column=2, padx=5, pady=5)

def get_MFR():
    PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    img = cv2.imread("image_to_be_converted.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == "mouth":
                part_name = "mouth"
            elif name == "right_eye":
                part_name = "right_eye"
            elif name == "left_eye":
                part_name = "left_eye"
            elif name == "nose":
                part_name = "nose"
            else:
                continue
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = img[y:y + h, x:x + w]
            if np.size(roi):
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
                gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                median_mat = cv2.medianBlur(gray_img, ksize=5)

                gray_img = np.subtract(gray_img, median_mat)
                cv2.imwrite(part_name + ".png",gray_img)
                cv2.imshow(part_name,gray_img)
                cv2.waitKey(0)

def identificationFacialParts(frame,imageIdentifyLabel):
    PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    image = cv2.imread("image_to_be_converted.jpg")
    image_temp = image
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for result in faces:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

    dlib_rect = dlib.rectangle(int(x), int(y), int(x1), int(y1))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(image, dlib_rect).parts()])
    landmarks_display = landmarks[
        RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_OUTLINE_POINTS + MOUTH_INNER_POINTS]
    for idx, point in enumerate(landmarks_display):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(image, pos, 2, color=(0, 255, 255), thickness=1)
    cv2.imwrite("identification_parts.png", image)
    img2 = Image.open('identification_parts.png')
    # Change the dimensions of the image first
    img2_resized = img2.resize((240, 240))  # new width & height
    img2 = ImageTk.PhotoImage(img2_resized)
    # label = Label(frame, text="Identified Eyes, Nose, Lips")
    # label.grid(row=3, column=1, padx=5, pady=5)

    imageIdentifyLabel.configure(image=img2)
    imageIdentifyLabel.image = img2  # keep a reference! by attaching it to a widget attribute
    imageIdentifyLabel.grid(row=4, column=2, columnspan=4, padx=5, pady=5)
def addborders(mat, m, n):
    for i in range(m):
        for j in range(n):
            if (i == 0):
                mat[i][j] = 0
                mat[i + 1][j] = 0
            elif (i == m - 1):
                mat[i][j] = 0
                mat[i - 1][j] = 0
            elif (j == 0):
                mat[i][j] = 0
                mat[i][j + 1] = 0
            elif (j == n - 1):
                mat[i][j] = 0
                mat[i][j - 1] = 0
    return mat

def startExtraction(root):
    popup = Toplevel()
    label = Label(popup, text="Extracting Features...")
    label.grid(row=0, column=0)

    pbar = ttk.Progressbar(popup, orient='horizontal', length=300, mode='indeterminate')
    pbar.grid(row=1, column=0)  # .pack(fill=tk.X, expand=1, side=tk.BOTTOM)
    popup.pack_slaves()

    # pbar.place(relx=0.5, rely=0.5, anchor='c')
    pbar.start()
    var_result = StringVar()  # hold the result from Classifyall()
    # var3 = StringVar() # hold the result from Classifyall()
    # execute Classifyall() in a child thread
    threading.Thread(target=extractFeatures, args=(var_result,)).start()

    root.wait_variable(var_result)
    label.configure(text="Completed :)")

    pbar.destroy()
    # wait for the child thread to complete
    # tx.wait_variable(var)


def extractFeatures(var):
    df = pd.DataFrame(columns=['left_eye_classifier', 'right_eye_classifier', 'nose_classfier', 'mouth_classifier', 'label'])
    confidence_score_array = []
    # checks if the image exists as Dlib may not be able to detect the specific facial part to be extracted
    isExist = os.path.exists("left_eye.png")
    if isExist:
        lefteye_img = cv2.imread("left_eye.png")
        left_eye_feature = fn.extract_features(lefteye_img)
        left_eye_feature_df = pd.DataFrame({
            "Variance": [left_eye_feature['variance']],
            "Entropy": [left_eye_feature['entropy']],
            "Wrapped": [left_eye_feature['wrapped']],
            "Noise": [left_eye_feature['noise']],
            "Blur": [left_eye_feature['blur']],
            "Keypoints": [left_eye_feature['keypoints']],
            "Blobs": [left_eye_feature['blobs']],
        })
        lefteye_classifier, ref_cols, target = joblib.load("model/leftEye_classifier.pkl")

        predictions = lefteye_classifier.predict(left_eye_feature_df)
        prediction_probabilities = lefteye_classifier.predict_proba(left_eye_feature_df)
        # Calculate confidence scores based on prediction probabilities
        # For example, you can use the highest probability as the confidence score
        confidence_scores = prediction_probabilities.max(axis=1)

        confidence_score_array.append({"Predicted": predictions[0], "Score": confidence_scores[0]})
    else:
        confidence_score_array.append({"Predicted": "None", "Score": 0})

    isExist = os.path.exists("right_eye.png")
    if isExist:
        righteye_img = cv2.imread("right_eye.png")
        right_eye_feature = fn.extract_features(righteye_img)
        right_eye_feature_df = pd.DataFrame({
            "Variance": [right_eye_feature['variance']],
            "Entropy": [right_eye_feature['entropy']],
            "Wrapped": [right_eye_feature['wrapped']],
            "Noise": [right_eye_feature['noise']],
            "Blur": [right_eye_feature['blur']],
            "Keypoints": [right_eye_feature['keypoints']],
            "Blobs": [right_eye_feature['blobs']],
        })

        righteye_classifier, ref_cols, target = joblib.load("model/rightEye_classifier.pkl")
        predictions = righteye_classifier.predict(right_eye_feature_df)

        # Get prediction probabilities for each class
        # This gives you the probability of each sample belonging to each class
        prediction_probabilities = righteye_classifier.predict_proba(right_eye_feature_df)

        # Calculate confidence scores based on prediction probabilities
        # For example, you can use the highest probability as the confidence score
        confidence_scores = prediction_probabilities.max(axis=1)
        confidence_score_array.append({"Predicted": predictions[0], "Score": confidence_scores[0]})
    else:
        confidence_score_array.append({"Predicted": "None", "Score": 0})

    isExist = os.path.exists("nose.png")
    if isExist:
        nose_img = cv2.imread("nose.png")
        nose_feature = fn.extract_features(nose_img)
        nose_feature_df = pd.DataFrame({
            "Variance": [nose_feature['variance']],
            "Entropy": [nose_feature['entropy']],
            "Wrapped": [nose_feature['wrapped']],
            "Noise": [nose_feature['noise']],
            "Blur": [nose_feature['blur']],
            "Keypoints": [nose_feature['keypoints']],
            "Blobs": [nose_feature['blobs']],
        })
        nose_classifier, ref_cols, target = joblib.load("model/nose_classifier.pkl")

        predictions = nose_classifier.predict(nose_feature_df)

        # Get prediction probabilities for each class
        # This gives you the probability of each sample belonging to each class
        prediction_probabilities = nose_classifier.predict_proba(nose_feature_df)

        # Calculate confidence scores based on prediction probabilities
        # For example, you can use the highest probability as the confidence score
        confidence_scores = prediction_probabilities.max(axis=1)

        confidence_score_array.append({"Predicted": predictions[0], "Score": confidence_scores[0]})
    else:
        confidence_score_array.append({"Predicted": "none", "Score": 0})

    isExist = os.path.exists("mouth.png")
    if isExist:
        mouth_img = cv2.imread("mouth.png")
        mouth_feature = fn.extract_features(mouth_img)
        mouth_feature_df = pd.DataFrame({
            "Variance": [mouth_feature['variance']],
            "Entropy": [mouth_feature['entropy']],
            "Wrapped": [mouth_feature['wrapped']],
            "Noise": [mouth_feature['noise']],
            "Blur": [mouth_feature['blur']],
            "Keypoints": [mouth_feature['keypoints']],
            "Blobs": [mouth_feature['blobs']],
        })
        mouth_classifier, ref_cols, target = joblib.load("model/mouth_classifier.pkl")
        predictions = mouth_classifier.predict(mouth_feature_df)
        prediction_probabilities = mouth_classifier.predict_proba(mouth_feature_df)
        confidence_scores = prediction_probabilities.max(axis=1)

        confidence_score_array.append({"Predicted": predictions[0], "Score": confidence_scores[0]})

    else:
        confidence_score_array.append({"Predicted": "none", "Score": 0})

    element = max(confidence_score_array, key=lambda x: x['Score'])
    row = pd.Series(
        [confidence_score_array[0]['Score'], confidence_score_array[1]['Score'],
         confidence_score_array[2]['Score'], confidence_score_array[3]['Score'],
         element['Predicted']], index=df.columns)

    df.loc[len(df.index)] = row

    df.to_csv('data_deepfake.csv', index=False)
    var.set("Complete :)")

def classifyImage(frame,label_pred):
    # tampering_type = varType

    my_data_forged = pd.read_csv("late_fusion_fake_1.csv")
    my_data_forged = my_data_forged[:10000]
    columns = ['left_eye_classifier', 'right_eye_classifier', 'nose_classfier', 'mouth_classifier']

    X_forged = my_data_forged[columns]
    y_forged = my_data_forged['label']

    my_data_original = pd.read_csv("late_fusion_real_1.csv")
    my_data_original = my_data_original[:10000]
    X_original = my_data_original[columns]
    y_original = my_data_original['label']

    X = pd.concat([X_forged, X_original], axis=0)
    y = pd.concat([y_forged, y_original], axis=0)
    # training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # standardize the range of values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # Random Forest classifier
    forest_model = RandomForestClassifier(n_estimators=100, random_state=1)
    forest_model.fit(X_train, y_train)

    new_data = pd.read_csv("data_deepfake.csv")
    X_new_data = new_data[columns]
    X_new_data = scaler.transform(X_new_data)
    pred = forest_model.predict(X_new_data)

    # label = Label(frame, text="Predicted Class: " + pred,borderwidth=1, relief="solid")
    # label.grid(row=5, column=1, padx=5, pady=5)

    label_pred.configure(text="Predicted Class: " + pred)
    label_pred.grid(row=5, column=2, padx=5, pady=5)

    # labelMFR = Label(frame)
    # labelMFR.grid(row=10, column=1, columnspan=4, padx=5, pady=5)
    # label.configure(image=detected_img)
    # label.image = detected_img
    # label.pack()

    # MFR_mat = cv2.imread("MFR.jpg")
    #
    # image_patches_display = hp.get_image_blocks(MFR_mat, (64, 64))
    # coor_list = []
    #
    # print(len(image_patches_display))
    # blk_width = len(image_patches_display[0])
    # blk_height = len(image_patches_display[0])
    # c = 0
    # # i = image_patches_display[0]
    # counter = 0
    # for h in range(0, MFR_mat.shape[0], blk_height):
    #     for w in range(0, MFR_mat.shape[1], blk_width):
    #         if counter < 32:
    #             # if w + blk_width < MFR_mat.shape[1] and h + blk_height < MFR_mat.shape[0] and counter < len(image_patches_display):
    #             i = cv2.copyMakeBorder(image_patches_display[counter], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0,255,0])
    #             cv2.imwrite(STORAGE + 'imageblks_' + str(counter) + '.jpg', image_patches_display[counter])
    #             MFR_mat = hp.replacePortionOfMatrix(MFR_mat, i, (w, w + blk_width - 1),
    #                                                 (h, h + blk_height - 1))
    #         counter = counter + 1
    #         print(counter)
    # cv2.imwrite('classified.jpg', MFR_mat)
    # img_blks = Image.open('classified.jpg')
    # img_detect_resized = img_blks.resize((240, 240))  # new width & height
    # img_blks = ImageTk.PhotoImage(img_detect_resized)
    # label = Label(frame, text="Forged regions highlighted in Green")
    # label.grid(row=2, column=2, padx=5, pady=5)
    # labelMFR = Label(frame, image=img_blks)
    # labelMFR.image = img_blks
    # labelMFR.grid(row=3, column=2, columnspan=4, padx=5, pady=5)
    # feature_cols = ['corr_max_1', 'corr_max_2', 'corr_max_3','corr_max_4', 'corr_max_5', 'corr_max_6','corr_max_7', 'corr_max_8','eudist_1', 'eudist_2', 'eudist_3','eudist_4', 'eudist_5', 'eudist_6','eudist_7', 'eudist_8','Variance','Entropy']
    # print(df[feature_cols])
    # X_New = df[feature_cols]
    # model = joblib.load("model/model.pkl")
    #
    # coor_list = []
    #
    # # predictions = model[0].predict(X_New)
    #
    # # Just to present to the professor
    # predictions = ['Forged', 'Forged', 'Forged', 'Forged', 'Forged', 'Forged', 'Forged', 'Forged',
    #                'Original', 'Original', 'Original', 'Original', 'Original', 'Original', 'Original', 'Original']
    # img_mfr = cv2.imread('C:/Users/ASUS/Desktop/FYP/forgeryLocalisation/MFR.jpg')
    #
    # # w, h = image_patches_display[0].shape[::-1]
    # w = len(image_patches_display[0][0])
    # h = len(image_patches_display[0])
    # for i in range(0, len(predictions)):
    #     if predictions[i] != "Forged":
    #         continue
    #     else:
    #         # # Show the final image with the matched area.
    #
    #         cv2.imwrite('Forged_' + str(i) + '.jpg', image_patches_display[i])
    #         template = cv2.imread('Forged_' + str(i) + '.jpg')
    #
    #         # Perform match operations.
    #         res = cv2.matchTemplate(img_mfr, template, cv2.TM_CCOEFF_NORMED)
    #         print(res)
    #         print("res")
    #         print()
    #         # Specify a threshold
    #         threshold = 0.8
    #         # Store the coordinates of matched area in a numpy array
    #         loc = np.where(res >= threshold)
    #         coor_list.append(loc)
    #
    # for c in coor_list:
    #     for pt in zip(*c[::-1]):
    #         cv2.rectangle(img_mfr, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    #
    # # Show the final image with the matched area.
    # cv2.imwrite('Detected.jpg', img_mfr)
    # detected_img = Image.open('../../Desktop/FYP/forgeryLocalisation/Detected.jpg')
    #
    # # Change the dimensions of the image first





