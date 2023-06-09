import cv2
import glob
import os
import TamperingFunctions as tf
import HelperFunctions as hp

SOURCE = 'Datasets/BOWS2/'
DESTINATION = 'Datasets/BOWS2/Trained'
PATH_DATASET_FORGED = 'Training Dataset/BOWS2/Completely_Original_&_Forged/32/Forged/'
PATH_DATASET_ORIGINAL = 'Training Dataset/BOWS2/Completely_Original_&_Forged/32/Original/'
files = glob.glob(SOURCE + '/*.pgm')
files = [file.replace("\\","/") for file in files]

# files = files[:1338]

# files = files[:19]

option = 0
for file in files:
    name = file.replace('Datasets/BOWS2/', '')
    name = name.replace(".pgm","")
    mat = hp.convert_image_matrix(file)
    tamperedBlks,originalBlks = hp.separateOriginalAndTamperedBlocks(mat,(32,32))

    filename = ""
    index = 0

    for i in range(0,len(tamperedBlks)):

        if option < 16:
            if option == 0:
                tamperedBlks[i] = tf.getMedianFilteredMatrix_win3x3(tamperedBlks[i])
                filename = name + "_forged_" + "mf_3_" + str(index) + ".png"
            elif option == 1:
                tamperedBlks[i] = tf.getMedianFilteredMatrix_win5x5(tamperedBlks[i])
                filename = name + "_forged_" + "mf_5_" + str(index) + ".png"
            elif option == 2:
                tamperedBlks[i] = tf.getAverageFilteringMatrix_win3x3(tamperedBlks[i])
                filename = name + "_forged_" + "avg_3_" + str(index) + ".png"
            elif option == 3:
                tamperedBlks[i] = tf.getAverageFilteringMatrix_win5x5(tamperedBlks[i])
                filename = name + "_forged_" + "avg_5_" + str(index) + ".png"
            elif option == 4:
                tamperedBlks[i] = tf.getGaussianFilteredMatrix_win3x3(tamperedBlks[i])
                filename = name + "_forged_" + "gau_3_" + str(index) + ".png"
            elif option == 5:
                tamperedBlks[i] = tf.getGaussianFilteredMatrix_win5x5(tamperedBlks[i])
                filename = name + "_forged_" + "gau_5_" + str(index) + ".png"
            elif option == 6:
                tamperedBlks[i] = tf.getWienerFilteredMatrix_win3x3(tamperedBlks[i])
                filename = name + "_forged_" + "wnr_3_" + str(index) + ".png"
            elif option == 7:
                tamperedBlks[i] = tf.getWienerFilteredMatrix_win5x5(tamperedBlks[i])
                filename = name + "_forged_" + "wnr_5_" + str(index) + ".png"
            elif option == 8:
                tamperedBlks[i] = tf.getContrastEnhancedMatrix(tamperedBlks[i],2)
                filename = name + "_forged_" + "ce_2_" + str(index) + ".png"
            elif option == 9:
                tamperedBlks[i] = tf.getContrastEnhancedMatrix(tamperedBlks[i], 0.5)
                filename = name + "_forged_" + "ce_0.5_" + str(index) + ".png"
            elif option == 10:
                tamperedBlks[i] = tf.getRescaledMatrix(tamperedBlks[i],1.2)
                filename = name + "_forged_" + "res_1.2_" + str(index) + ".png"
            elif option == 11:
                tamperedBlks[i] = tf.getRescaledMatrix(tamperedBlks[i], 1.5)
                filename = name + "_forged_" + "res_1.5_" + str(index) + ".png"
            elif option == 12:
                tamperedBlks[i] = tf.getGaussianNoiseMatrix(tamperedBlks[i],0.0005)*255
                filename = name + "_forged_" + "awgn_0.0005_" + str(index) + ".png"
            elif option == 13:
                tamperedBlks[i] = tf.getGaussianNoiseMatrix(tamperedBlks[i],0.001)*255
                filename = name + "_forged_" + "awgn_0.001_" + str(index) + ".png"
            elif option == 14:
                tamperedBlks[i] = tf.getUnsharpMaskingMatrix(tamperedBlks[i],0.2)*255
                filename = name + "_forged_" + "um_0.2_" + str(index) + ".png"
            elif option == 15:
                tamperedBlks[i] = tf.getUnsharpMaskingMatrix(tamperedBlks[i], 0.4)*255
                filename = name + "_forged_" + "um_0.4_" + str(index) + ".png"

            cv2.imwrite(PATH_DATASET_FORGED + filename, tamperedBlks[i])

        if option == 16:
            filename = name + "_forged_" + "jpg_80_" + str(index) + ".jpg"
            tf.getJPEGCompressedImage(tamperedBlks[i], 80, PATH_DATASET_FORGED + filename)
        elif option == 17:
            filename = name + "_forged_" + "jpg_90_" + str(index) + ".jpg"
            tf.getJPEGCompressedImage(tamperedBlks[i], 90, PATH_DATASET_FORGED + filename)
        elif option == 18:
            filename = name + "_forged_" + "jp2_2_" + str(index) + ".jp2"
            tf.getJPEG2000Image(tamperedBlks[i], 2, PATH_DATASET_FORGED + filename)
        elif option == 19:
            filename = name + "_forged_" + "jp2_1.5_" + str(index) + ".jp2"
            tf.getJPEG2000Image(tamperedBlks[i], 1.5, PATH_DATASET_FORGED + filename)

        index = index + 1

    index = index - 1

    for o in range(0,len(originalBlks)):
        index = index + 1
        cv2.imwrite(PATH_DATASET_ORIGINAL + name + "_original_" + str(index) + ".png", originalBlks[o])

    option = option + 1

    if option == 19:
        option = 0


    os.rename(SOURCE + file.replace('Datasets/BOWS2/',''), DESTINATION + file.replace('Datasets/BOWS2',''))

print("Completed :)")