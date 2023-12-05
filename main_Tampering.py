import cv2
import glob
import os
import random
import TamperingFunctions as tf
import HelperFunctions as hp

import numpy as np

SOURCE = 'Datasets/DB'
# SOURCE_PB = 'Datasets/IEEE/Image Dataset'


TEMP_STORAGE = 'Temp/'

# PATH_DATASET_FORGED = 'Training Dataset/BOWS2/Variant_1/64/Forged/'
# PATH_DATASET_ORIGINAL = 'Training Dataset/BOWS2/Variant_1/64/Original/'
# PATH_CROPPED_512 = 'Datasets/IEEE/Image Dataset/Cropped_512/'
files = os.listdir(SOURCE)
# files = glob.glob(SOURCE + '/*')
files = [file.replace("\\", "/") for file in files]
files = files[:1]
# print(len(files))
# print(files[0])
for option in range(6,7):
    counter = 0
    for file in files:
        name = file.split(".")[0]
        # name = file.replace(".png", "")
        mat = hp.convert_image_matrix(SOURCE + "/" + "0a0ab9a903587d083e7051adfb17a779.png")
        left_side_mat = hp.get_left_side_of_the_image(mat)
    
        if option < 8:
            if option == 0:
                tampered_method = "_forged_" + "mf_5_"
                portion_mat = tf.getMedianFilteredMatrix_win5x5(left_side_mat)
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/mf/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/mf/Original/'
            elif option == 1:
                tampered_method = "_forged_" + "avg_5_"
                portion_mat = tf.getAverageFilteringMatrix_win5x5(left_side_mat)
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/avg/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/avg/Original/'
            elif option == 2:
                tampered_method = "_forged_" + "gau_5_"
                portion_mat = tf.getGaussianFilteredMatrix_win5x5(left_side_mat)
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/gau/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/gau/Original/'
            elif option == 3:
                tampered_method = "_forged_" + "wnr_5_"
                portion_mat = tf.getWienerFilteredMatrix_win5x5(left_side_mat)
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/wnr/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/wnr/Original/'
            elif option == 4:
                tampered_method = "_forged_" + "ce_0.5_"
                portion_mat = tf.getContrastEnhancedMatrix(left_side_mat, 0.5)
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/ce/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/ce/Original/'
            elif option == 5:
                tampered_method = "_forged_" + "res_1.5_"
                portion_mat = tf.getRescaledMatrix(left_side_mat, 1.5)
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/res/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/res/Original/'
            elif option == 6:
                tampered_method = "_forged_" + "awgn_0.001_"
                portion_mat = tf.getGaussianNoiseMatrix(left_side_mat, 0.001) * 255
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/awgn/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/awgn/Original/'
            elif option == 7:
                tampered_method = "_forged_" + "um_0.4_"
                portion_mat = tf.getUnsharpMaskingMatrix(left_side_mat, 0.4) * 255
                FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/um/Forged/'
                ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/um/Original/'
    
        elif option == 8:
            tampered_method = "_forged_" + "_jpg_90_"
            filename = name + "_jpg_90_" + ".jpg"
            tf.getJPEGCompressedImage(left_side_mat, 90, TEMP_STORAGE + filename)
            portion_mat = hp.convert_image_matrix(TEMP_STORAGE + filename)
            FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/jpg/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/jpg/Original/'
            print(portion_mat.shape)
    
        elif option == 9:
            tampered_method = "_forged_" + "_jp2_1.5_"
            filename = name + "_jp2_1.5_" + ".jp2"
            tf.getJPEG2000Image(left_side_mat, 1.5, TEMP_STORAGE + filename)
            portion_mat = hp.convert_image_matrix(TEMP_STORAGE + filename)
            FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_30/jp2/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_30/jp2/Original/'
            print(portion_mat.shape)
    
        mat = hp.combine(mat, portion_mat)
        jpeg_name = name + "_awgn_jpg_30" + ".jpg"
        tf.getJPEGCompressedImage(mat, 30, TEMP_STORAGE + jpeg_name)

        mat = hp.convert_image_matrix(TEMP_STORAGE + jpeg_name)
    
        # tamperedBlks, originalBlks = hp.separateOriginalAndTamperedBlocks(mat, (64, 64))
        #
        # for t in range(0, len(tamperedBlks)):
        #     cv2.imwrite(FORGED_DESTINATION + name + tampered_method + "30_" + str(t) + ".jpg", tamperedBlks[t])
        #
        # for o in range(0,len(originalBlks)):
        #     t = t + 1
        #     cv2.imwrite(ORIGINAL_DESTINATION + name + "_original_" + str(t) + ".jpg", originalBlks[o])


        counter+=1
        print("Complete " + str(counter) + ":)")
    


    # os.rename(SOURCE + file.replace('Datasets/IEEE+HR', ''), DESTINATION + file.replace('Datasets/IEEE+HR', ''))
    # name = name.replace(".png", "")
    # mat = hp.convert_image_matrix(file)
    # if mat.shape[0] > 512 and mat.shape[1] > 512:
    #     crop_mat = hp.centerCropMatrix(mat, (512, 512))
    #     # blks = hp.get_image_blocks(crop_mat, (512,512))
    #     # for i in range(0,len(blks)):
    #     cv2.imwrite(PATH_CROPPED_512 + name + ".png", crop_mat)
#
# SOURCE = 'Datasets/IEEE/\IEEE+HR'
# DESTINATION = 'Datasets/BOWS2/Trained'
# files = glob.glob(SOURCE)
# files = [file.replace("\\", "/") for file in files]


