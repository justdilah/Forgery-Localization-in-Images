import pandas as pd
import deepfakedetection as fn
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import joblib
import cv2
import os

# SOURCE = "Fake/"
# DESTINATION = "model/Faceparts_LF_fake/"
# files = os.listdir(SOURCE)
# files = [file.replace("\\", "/") for file in files]

df = pd.DataFrame(columns=['left_eye_classifier', 'right_eye_classifier', 'nose_classfier', 'mouth_classifier', 'label'])

PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

counter = 0
# print(len(files))
for index in range(0, 70002):
    confidence_score_array = []
    # img = cv2.imread(SOURCE + files[index])
    # print(files[index])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 1)
    #
    # for (i, rect) in enumerate(rects):
    #     shape = predictor(gray, rect)
    #     shape = face_utils.shape_to_np(shape)
    #
    #     # loop over the face parts individually
    #     for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    #         if name == "mouth":
    #             part_name = "mouth"
    #         elif name == "right_eye":
    #             part_name = "right_eye"
    #         elif name == "left_eye":
    #             part_name = "left_eye"
    #         elif name == "nose":
    #             part_name = "nose"
    #         else:
    #             continue
    #         (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    #         roi = img[y:y + h, x:x + w]
    #         if np.size(roi):
    #             roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
    #             cv2.imwrite(DESTINATION + part_name + "_fake_" + str(index) + ".jpg", roi)

    left_eye_dir = "FaceParts/Fake/LeftEye/"
    isExist = os.path.exists(left_eye_dir + "fake_" + str(index) + "_left_eye.jpg")
    if isExist:
        print(isExist)
        lefteye_img = cv2.imread(left_eye_dir + "fake_" + str(index) + "_left_eye.jpg")
        print(left_eye_dir + "fake_" + str(index) + "_left_eye.jpg")
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
        # probability_lefteye = lefteye_classifier.predict_proba(left_eye_feature_df)

        predictions = lefteye_classifier.predict(left_eye_feature_df)
        probability_lefteye = lefteye_classifier.predict_proba(left_eye_feature_df)
        # instance_prob = probability_lefteye[0]
        #
        # # Get the maximum probability
        # max_prob = max(instance_prob)
        # # Calculate the confidence score
        # confidence_score_lefteye = max_prob / sum(instance_prob)
        #
        # confidence_score_array.append(co
        # Get prediction probabilities for each class
        # This gives you the probability of each sample belonging to each class
        prediction_probabilities = lefteye_classifier.predict_proba(left_eye_feature_df)
        print(left_eye_feature_df)
        print(prediction_probabilities)

        # Calculate confidence scores based on prediction probabilities
        # For example, you can use the highest probability as the confidence score
        confidence_scores = prediction_probabilities.max(axis=1)
        print(left_eye_dir + "fake_" + str(index) + "_left_eye.jpg")
        print(confidence_scores)

        confidence_score_array.append({"Predicted": predictions[0], "Score": confidence_scores[0]})
    else:
        confidence_score_array.append({"Predicted": "None", "Score": 0})

    right_eye_dir = "FaceParts/Fake/RightEye/"
    isExist = os.path.exists(right_eye_dir + "fake_" + str(index) + "_right_eye.jpg")
    if isExist:
        righteye_img = cv2.imread(right_eye_dir + "fake_" + str(index) + "_right_eye.jpg")
        right_eye_feature = fn.extract_features(righteye_img)
        right_eye_feature_df = pd.DataFrame({
            "Variance": [left_eye_feature['variance']],
            "Entropy": [left_eye_feature['entropy']],
            "Wrapped": [left_eye_feature['wrapped']],
            "Noise": [left_eye_feature['noise']],
            "Blur": [left_eye_feature['blur']],
            "Keypoints": [left_eye_feature['keypoints']],
            "Blobs": [left_eye_feature['blobs']],
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

    nose_dir = "FaceParts/Fake/Nose/"
    isExist = os.path.exists(nose_dir + "fake_" + str(index) + "_nose.jpg")
    if isExist:
        nose_img = cv2.imread(nose_dir + "fake_" + str(index) + "_nose.jpg")
        nose_feature = fn.extract_features(nose_img)
        nose_feature_df = pd.DataFrame({
            "Variance": [left_eye_feature['variance']],
            "Entropy": [left_eye_feature['entropy']],
            "Wrapped": [left_eye_feature['wrapped']],
            "Noise": [left_eye_feature['noise']],
            "Blur": [left_eye_feature['blur']],
            "Keypoints": [left_eye_feature['keypoints']],
            "Blobs": [left_eye_feature['blobs']],
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

    mouth_dir = "FaceParts/Fake/Mouth/"
    isExist = os.path.exists(mouth_dir + "fake_" + str(index) + "_mouth.jpg")
    if isExist:
        mouth_img = cv2.imread(mouth_dir + "fake_" + str(index) + "_mouth.jpg")
        mouth_feature = fn.extract_features(mouth_img)
        mouth_feature_df = pd.DataFrame({
            "Variance": [left_eye_feature['variance']],
            "Entropy": [left_eye_feature['entropy']],
            "Wrapped": [left_eye_feature['wrapped']],
            "Noise": [left_eye_feature['noise']],
            "Blur": [left_eye_feature['blur']],
            "Keypoints": [left_eye_feature['keypoints']],
            "Blobs": [left_eye_feature['blobs']],
        })
        mouth_classifier, ref_cols, target = joblib.load("model/mouth_classifier.pkl")
        predictions = mouth_classifier.predict(mouth_feature_df)

        # Get prediction probabilities for each class
        # This gives you the probability of each sample belonging to each class
        prediction_probabilities = mouth_classifier.predict_proba(mouth_feature_df)

        # Calculate confidence scores based on prediction probabilities
        # For example, you can use the highest probability as the confidence score
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
    print("Complete " + str(counter) + ":)")

    counter += 1
    df.to_csv('late_fusion_fake.csv', index=False)
