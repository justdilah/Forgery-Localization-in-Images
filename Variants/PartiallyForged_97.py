import cv2
import glob
import os
import random
import TamperingFunctions as tf
import HelperFunctions as hp

import numpy as np

SOURCE = 'Datasets/DB'
TEMP_STORAGE = 'Temp/'
# # SOURCE_PB = 'Datasets/IEEE/Image Dataset'
# FORGED_DESTINATION = 'Training Dataset/DB/64/JPG_70/Forged/'
# ORIGINAL_DESTINATION = 'Training Dataset/DB/64/JPG_70/Original/'
#

for option in range(0,1):
    # PF_97 -> partially forged from image rows, columns 97 to 448

    files = os.listdir(SOURCE)
    # files = glob.glob(SOURCE + '/*')
    files = [file.replace("\\", "/") for file in files]
    # files = files[:1338]
    files = files[:1]
    print(len(files))
    # print(files[0])

    # option = 0
    counter = 0
    for file in files:
        name = file.split(".")[0]
        # name = file.replace(".png", "")
        mat = hp.convert_image_matrix(SOURCE + "/" + "00f2590c9968d4f36464ead50585e6eb.png")
        # mat = hp.convert_image_matrix(SOURCE + "/" + file)
        tampered_mat = hp.retrievePortionOfMatrix(mat,(97,448),(97,448))


        if option == 0:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/mf/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/mf/Original/'
            tampered_method = "_forged_" + "mf_5_"
            portion_mat = tf.getMedianFilteredMatrix_win5x5(tampered_mat)
        elif option == 1:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/avg/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/avg/Original/'
            tampered_method = "_forged_" + "avg_5_"
            portion_mat = tf.getAverageFilteringMatrix_win5x5(tampered_mat)
        elif option == 2:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/gau/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/gau/Original/'
            tampered_method = "_forged_" + "gau_5_"
            portion_mat = tf.getGaussianFilteredMatrix_win5x5(tampered_mat)
        elif option == 3:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/wnr/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/wnr/Original/'
            tampered_method = "_forged_" + "wnr_5_"
            portion_mat = tf.getWienerFilteredMatrix_win5x5(tampered_mat)
            mat = mat / 255
            portion_mat = portion_mat / 255
        elif option == 4:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/ce/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/ce/Original/'
            tampered_method = "_forged_" + "ce_0.5_"
            portion_mat = tf.getContrastEnhancedMatrix(tampered_mat, 0.5)
        elif option == 5:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/res/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/res/Original/'
            tampered_method = "_forged_" + "res_1.5_"
            portion_mat = tf.getRescaledMatrix(tampered_mat, 1.5)
        elif option == 6:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/awgn/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/awgn/Original/'
            tampered_method = "_forged_" + "awgn_0.001_"
            portion_mat = tf.getGaussianNoiseMatrix(tampered_mat, 0.001) * 255
        elif option == 7:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/um/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/um/Original/'
            tampered_method = "_forged_" + "um_0.4_"
            portion_mat = tf.getUnsharpMaskingMatrix(tampered_mat, 0.4) * 255

        elif option == 8:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/jpg/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/jpg/Original/'
            tampered_method = "_forged_" + "_jpg_90_"
            filename = name + "_jpg_90_" + ".jpg"
            tf.getJPEGCompressedImage(tampered_mat, 90, TEMP_STORAGE + filename)
            portion_mat = hp.convert_image_matrix(TEMP_STORAGE + filename)
            print(portion_mat.shape)

        elif option == 9:
            FORGED_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/jp2/Forged/'
            ORIGINAL_DESTINATION = 'Training Dataset/PB_DB/PF_97/JPG_70/jp2/Original/'
            tampered_method = "_forged_" + "_jp2_1.5_"
            filename = name + "_jp2_1.5_" + ".jp2"
            tf.getJPEG2000Image(tampered_mat, 1.5, TEMP_STORAGE + filename)
            portion_mat = hp.convert_image_matrix(TEMP_STORAGE + filename)
            print(portion_mat.shape)


        mat = hp.replacePortionOfMatrix(mat, portion_mat,(97,448),(97,448))
        jpeg_name = name + "_jpg_30_awgn" + ".jpg"
        tf.getJPEGCompressedImage(mat, 30, TEMP_STORAGE + jpeg_name)

        mat = hp.convert_image_matrix(TEMP_STORAGE + jpeg_name)

        imageblks = hp.get_image_blocks(mat, (64, 64))

        # for index,blk in enumerate(imageblks):
        #     if (index >= 0 and index <=7) or (index >= 56 and index <=63):
        #         cv2.imwrite(ORIGINAL_DESTINATION + name + "_original_" + str(index) + ".jpg", blk)
        #     elif (index == 8 or index == 16 or index == 24 or index == 32 or index == 40 or index == 48):
        #         cv2.imwrite(ORIGINAL_DESTINATION + name + "_original_" + str(index) + ".jpg", blk)
        #     elif (index == 15 or index == 23 or index == 31 or index == 39 or index == 47 or index == 55):
        #         cv2.imwrite(ORIGINAL_DESTINATION + name + "_original_" + str(index) + ".jpg", blk)
        #     elif (index == 9 or index == 17 or index == 25 or index == 33 or index == 41 or index == 49 or index == 55):
        #         cv2.imwrite(ORIGINAL_DESTINATION + name + "_original_" + str(index) + ".jpg", blk)
        #     elif (index >= 10 and index <= 14):
        #         cv2.imwrite(ORIGINAL_DESTINATION + name + "_original_" + str(index) + ".jpg", blk)
        #     else:
        #         cv2.imwrite(FORGED_DESTINATION + name + tampered_method + "70_" + str(index) + ".jpg", blk)
        #
        #
        # # if option == 10:
        # #     option = 0
        # counter = counter + 1
        # print(FORGED_DESTINATION)
        print("Complete :) " + str(counter))


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



