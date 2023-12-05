import cv2
import glob
import os
import random
import scipy.signal as sp
import TamperingFunctions as tf
import HelperFunctions as hp
import math
import numpy as np
import pandas as pd
import re

SOURCE = 'Datasets/DB'
TEMP_STORAGE = 'Temp/'
READ_PATH = "CSV/128/"


# EXTRACT FEATURES FROM 128 BY 128 IMAGE BLOCKS
option = 9
if option == 0:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/mf/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/mf/Original/'
    tamper_type = "mf"
    read_csv = READ_PATH + "trainingdataset_128_mf_original_0_30.csv"

elif option == 1:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/avg/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/avg/Original/'
    tamper_type = "avg"
    read_csv = READ_PATH + "trainingdataset_128_avg_original_0_30.csv"

elif option == 2:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/gau/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/gau/Original/'
    tamper_type = "gau"
    read_csv = READ_PATH + "trainingdataset_128_gau_original_0_30.csv"

elif option == 3:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/wnr/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/wnr/Original/'
    tamper_type = "wnr"
    read_csv = READ_PATH + "trainingdataset_128_wnr_original_0_30.csv"

elif option == 4:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/ce/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/ce/Original/'
    tamper_type = "ce"
    read_csv = READ_PATH + "trainingdataset_128_ce_original_0_30.csv"

elif option == 5:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/res/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/res/Original/'
    tamper_type = "res"
    read_csv = READ_PATH + "trainingdataset_128_res_original_0_30.csv"

elif option == 6:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/awgn/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/awgn/Original/'
    tamper_type = "awgn"
    read_csv = READ_PATH + "trainingdataset_128_awgn_original_0_30.csv"

elif option == 7:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/um/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/um/Original/'
    tamper_type = "um"
    read_csv = READ_PATH + "trainingdataset_128_um_original_0_30.csv"


elif option == 8:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/jpg/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/jpg/Original/'
    tamper_type = "jpg"
    read_csv = READ_PATH + "trainingdataset_128_jpg_original_0_30.csv"

elif option == 9:
    FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/jp2/Forged/'
    ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/jp2/Original/'
    tamper_type = "jp2"
    read_csv = READ_PATH + "trainingdataset_128_jp2_original_0.csv"

# FORGED_FOLDER = 'Training Dataset/DB/128/JPG_30/wnr/Forged/'
# ORIGINAL_FOLDER = 'Training Dataset/DB/128/JPG_30/wnr/Original/'

# UCOMMENT THIS CODE IF THERE ARE SOME FILES THAT HAVE BEEN READ
# df_read_original = pd.read_csv(read_csv)
# read_img_original = df_read_original['image_name'].tolist()
# read_img_original = [file + ".jpg" for file in read_img_original]
# #
# print("Read Files")
# print(len(read_img_original))

files = os.listdir(ORIGINAL_FOLDER)
files = [file.replace("\\", "/") for file in files]
#
files = files[:1200]
# print("All Files")
# print(len(files))
# # #
# # # # UCOMMENT THIS CODE IF THERE ARE SOME FILES THAT HAVE BEEN READ
# for i in read_img_original:
#     if i in files:
#         files.remove(i)
# # #
# print("Filtered Files")
# print(len(files))

df = pd.DataFrame(
    columns=['image_name', 'image_patch_index', 'corr_max_1', 'corr_max_2', 'corr_max_3', 'corr_max_4',
             'corr_max_5',
             'corr_max_6', 'corr_max_7', 'corr_max_8', 'eudist_1', 'eudist_2', 'eudist_3', 'eudist_4', 'eudist_5',
             'eudist_6', 'eudist_7', 'eudist_8', 'Variance', 'Entropy', 'type'])

counter = 0
for file in files:
    name = file.replace(".jpg", "")
    name_img = name.split("_")[0]
    index = name.split("_")[-1]

    index = int(index)

    # Stores correlation values between neighbouring matrices and middle matrix
    n_blks_corr = []
    n_blks_eudist = []

    middle_mat = hp.convert_image_matrix(ORIGINAL_FOLDER + "/" + file)
    middle_median_mat = tf.getMedianFilteredMatrix_win3x3(middle_mat)
    MFR_middle_mat = np.subtract(middle_mat, middle_median_mat)

    converted_matrix_rb = MFR_middle_mat.astype('float32')
    converted_matrix_rb = cv2.cvtColor(converted_matrix_rb, cv2.COLOR_GRAY2BGR)

    hist_rb = cv2.calcHist([converted_matrix_rb], [0], None, [256], [0, 256])
    cv2.normalize(hist_rb, hist_rb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Calculate Entropy
    ent = 0
    for j in range(0, 256):
        if (hist_rb[j][0] != 0):
            ent -= hist_rb[j][0] * math.log(abs(hist_rb[j][0]))

    if index == 0:
        list_elements = ["4", "1", "5"]
    elif index == 12:
        list_elements = ["-4", "-3", "1"]
    elif index == 3:
        list_elements = ["-1", "3", "4"]
    elif index == 15:
        list_elements = ["-4", "-5", "-1"]
    elif index == 4 or index == 8:
        list_elements = ["-4", "-3", "1", "5", "4"]
    elif index == 7 or index == 11:
        list_elements = ["-4", "-5", "-1", "3", "4"]
    elif index == 1 or index == 2:
        list_elements = ["-1", "3", "4", "5", "1"]
    elif index == 13 or index == 14:
        list_elements = ["-1", "-3", "-4", "-5", "1"]
    else:
        list_elements = ["-5", "-1", "3", "-4", "4", "-3", "1", "5"]

    regex_string = "(" + name_img + "_).*(_"
    filtered_values = [regex_string + str(int(index) + int(x)) + ").jpg" for x in list_elements if
                       0 <= int(index) + int(x) <= 15]

    # regex_string = "/(000bc3906100ede4b1374cea075adedb_).*"
    original_files = glob.glob(FORGED_FOLDER + '/*' + name_img + '_*.jpg')
    original_files = [file.replace("\\", "/") for file in original_files]
    original_files = glob.glob(ORIGINAL_FOLDER + '/*' + name_img + '_*.jpg')
    original_files = [file.replace("\\", "/") for file in original_files]
    all_files = original_files + original_files
    neighbour_matrices = [hp.convert_image_matrix(x) for regex in filtered_values for x in all_files if
                          re.search(regex, x)]

    if 8 > len(neighbour_matrices):
        length = 16 - len(filtered_values)
        for i in range(0, length):
            zero_mat = np.zeros((128, 128))
            neighbour_matrices.append(zero_mat)

    for n_mat in neighbour_matrices:
        median_neighbour_mat = tf.getMedianFilteredMatrix_win3x3(n_mat)
        MFR_neighbour_mat = np.subtract(n_mat, median_neighbour_mat)
        corr = sp.correlate2d(MFR_middle_mat,
                              MFR_neighbour_mat,
                              mode='full')

        n_blks_corr.append(corr.max())

        # Histogram
        converted_matrix_nb = MFR_neighbour_mat.astype('float32')
        converted_matrix_nb = cv2.cvtColor(converted_matrix_nb, cv2.COLOR_GRAY2BGR)
        hist_nb = cv2.calcHist([converted_matrix_nb], [0], None, [256], [0, 256])
        cv2.normalize(hist_nb, hist_nb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Calculate eucliean distance
        eu_dist = cv2.norm(hist_rb, hist_nb, normType=cv2.NORM_L2)
        n_blks_eudist.append(eu_dist)

    row = pd.Series(
        [name, index, n_blks_corr[0], n_blks_corr[1], n_blks_corr[2], n_blks_corr[3], n_blks_corr[4],
         n_blks_corr[5],
         n_blks_corr[6], n_blks_corr[7], n_blks_eudist[0], n_blks_eudist[1], n_blks_eudist[2], n_blks_eudist[3],
         n_blks_eudist[4], n_blks_eudist[5], n_blks_eudist[6], n_blks_eudist[7], np.var(MFR_middle_mat), ent,
         "Original"], index=df.columns)

    # df = df.append(row, ignore_index=True)
    df.loc[len(df.index)] = row
    # df2 = pd.concat([row, df.loc[:]]).reset_index(drop=True)

    df.to_csv(READ_PATH + 'trainingdataset_128_'+ tamper_type + '_original_0_30.csv', index=False)

    counter += 1
    print("Complete :) " + str(counter) + " " + tamper_type)




