import cv2
import glob
import os
import random
import TamperingFunctions as tf
import HelperFunctions as hp

import numpy as np

# PRODUCE COMPLETELY FORGED BLOCKS FROM THE IMAGE
SOURCE = 'Datasets/DB'
TEMP_STORAGE = 'Temp/'

for option in range(0,10):
    # Completely_Forged
    files = os.listdir(SOURCE)
    files = [file.replace("\\", "/") for file in files]
    files = files[:500]
    print(len(files))

    # option = 0
    counter = 0
    for file in files:
        name = file.split(".")[0]
        # name = file.replace(".png", "")
        tampered_mat = hp.convert_image_matrix(SOURCE + "/" + file)
        if option == 0:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/mf/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/mf/Original/'
            tampered_method = "_forged_" + "mf_5_"
            tampered_mat = tf.getMedianFilteredMatrix_win5x5(tampered_mat)
        elif option == 1:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/avg/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/avg/Original/'
            tampered_method = "_forged_" + "avg_5_"
            tampered_mat = tf.getAverageFilteringMatrix_win5x5(tampered_mat)
        elif option == 2:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/gau/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/gau/Original/'
            tampered_method = "_forged_" + "gau_5_"
            tampered_mat = tf.getGaussianFilteredMatrix_win5x5(tampered_mat)
        elif option == 3:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/wnr/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/wnr/Original/'
            tampered_method = "_forged_" + "wnr_5_"
            tampered_mat = tf.getWienerFilteredMatrix_win5x5(tampered_mat)
            # mat = mat / 255
            tampered_mat = tampered_mat / 255
        elif option == 4:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/ce/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/ce/Original/'
            tampered_method = "_forged_" + "ce_0.5_"
            tampered_mat = tf.getContrastEnhancedMatrix(tampered_mat, 0.5)
        elif option == 5:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/res/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/res/Original/'
            tampered_method = "_forged_" + "res_1.5_"
            tampered_mat = tf.getRescaledMatrix(tampered_mat, 1.5)
        elif option == 6:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/awgn/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/awgn/Original/'
            tampered_method = "_forged_" + "awgn_0.001_"
            tampered_mat = tf.getGaussianNoiseMatrix(tampered_mat, 0.001) * 255
        elif option == 7:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/um/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/um/Original/'
            tampered_method = "_forged_" + "um_0.4_"
            tampered_mat = tf.getUnsharpMaskingMatrix(tampered_mat, 0.4) * 255

        elif option == 8:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/jpg/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/jpg/Original/'
            tampered_method = "_forged_" + "_jpg_90_"
            filename = name + "_jpg_90_" + ".jpg"
            tf.getJPEGCompressedImage(tampered_mat, 90, TEMP_STORAGE + filename)
            tampered_mat = hp.convert_image_matrix(TEMP_STORAGE + filename)
            print(tampered_mat.shape)

        elif option == 9:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/jp2/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/Completely_Forged/JPG_70/jp2/Original/'
            tampered_method = "_forged_" + "_jp2_1.5_"
            filename = name + "_jp2_1.5_" + ".jp2"
            tf.getJPEG2000Image(tampered_mat, 1.5, TEMP_STORAGE + filename)
            tampered_mat = hp.convert_image_matrix(TEMP_STORAGE + filename)

        jpeg_name = name + "_jpg_70" + ".jpg"
        tf.getJPEGCompressedImage(tampered_mat, 70, TEMP_STORAGE + jpeg_name)

        # convert the image to a matrix
        tampered_mat = hp.convert_image_matrix(TEMP_STORAGE + jpeg_name)

        # divide the image matrix into blocks with the specific blk dimensions
        imageblks = hp.get_image_blocks(tampered_mat, (64, 64))

        for index,blk in enumerate(imageblks):
            cv2.imwrite(FORGED_DESTINATION + name + tampered_method + "70_" + str(index) + ".jpg", blk)

        counter = counter + 1
        print(FORGED_DESTINATION)
        print("Complete :) " + str(counter))



