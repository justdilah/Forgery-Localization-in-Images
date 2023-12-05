from tkinter import *
from tkinter import filedialog
import threading
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
import re
from tkinter import ttk

STORAGE = "UserStorage/"


def get_sub_matrices_custom_padding(orig_matrix, kernel_size):
    width = len(orig_matrix[0])
    height = len(orig_matrix)
    blksize = math.floor(int(kernel_size[1] / 3))
    giant_matrix = []
    for i in range(0, height - kernel_size[1] + 1, blksize):
        for j in range(0, width - kernel_size[0] + 1, blksize):
            giant_matrix.append(
                [
                    [orig_matrix[col][row] for row in range(j, j + kernel_size[0])]
                    for col in range(i, i + kernel_size[1])
                ]
            )
    img_sampling = np.array(giant_matrix)
    return img_sampling


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


def upload_file(frame, labelTampered):
    f_types = (('Jpg Files', '*.jpg'), ('PNG Files', '*.png'))
    global filename
    filename = filedialog.askopenfilename(filetypes=f_types)

    # need to check if it is jpg or not
    img_mat = convert_image_matrix(filename)
    global row
    global col
    row = len(img_mat[0])
    col = len(img_mat)
    filename = filename.replace(".pgm", ".jpg")
    cv2.imwrite(filename, img_mat)
    img = Image.open(filename)
    # Change the dimensions of the image first
    img_resized = img.resize((240, 240))  # new width & height
    img = ImageTk.PhotoImage(img_resized)
    labelTampered.configure(image=img)
    labelTampered.image = img  # keep a reference! by attaching it to a widget attribute
    labelTampered.grid(row=1, column=1, padx=5, pady=5)


def get_MFR(frame, labelMFR):
    forged_mat = cv2.imread(filename, 0)
    medianMat = tf.getMedianFilteredMatrix_win3x3(forged_mat)
    MFR_mat = np.subtract(forged_mat, medianMat)
    cv2.imwrite('MFR.png', MFR_mat)
    img2 = Image.open('MFR.png')
    # Change the dimensions of the image first
    img2_resized = img2.resize((240, 240))  # new width & height
    img2 = ImageTk.PhotoImage(img2_resized)

    labelMFR.configure(image=img2)
    labelMFR.image = img2

    labelMFR.grid(row=4, column=1, columnspan=4, padx=5, pady=5)


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


def getImageBlocks(var, frame, label, labelMFR, ):
    type = var.get()
    var = var.get().split(" ")[0]
    var = int(var)
    MFR_mat = cv2.imread("MFR.png")
    orig_mat = cv2.imread(filename)

    MFR_mat_padded = np.pad(MFR_mat, var, mode='constant')
    global image_patches_padded
    var_temp = var * 3
    image_patches_padded = get_sub_matrices_custom_padding(MFR_mat_padded, (var_temp, var_temp))

    global image_patches_display

    print(var)
    image_patches_display = hp.get_image_blocks(MFR_mat, (var, var))
    image_blocks = hp.get_image_blocks(orig_mat, (var, var))
    coor_list = []

    blk_width = len(image_patches_display[0])
    blk_height = len(image_patches_display[0])
    c = 0
    # i = image_patches_display[0]
    counter = 0
    for h in range(0, MFR_mat.shape[0], blk_height):
        for w in range(0, MFR_mat.shape[1], blk_width):
            # if w + blk_width < MFR_mat.shape[1] and h + blk_height < MFR_mat.shape[0] and counter < len(image_patches_display):
            i = cv2.copyMakeBorder(image_patches_display[counter], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 0])
            cv2.imwrite(STORAGE + 'imageblks_' + str(counter) + '.jpg', image_blocks[counter])
            MFR_mat = hp.replacePortionOfMatrix(MFR_mat, i, (w, w + blk_width - 1),
                                                (h, h + blk_height - 1))
            counter = counter + 1

    cv2.imwrite('imageblksDrawnOut.png', MFR_mat)
    img_blks = Image.open('imageblksDrawnOut.png')

    # Change the dimensions of the image first
    img_detect_resized = img_blks.resize((240, 240))  # new width & height
    img_blks = ImageTk.PhotoImage(img_detect_resized)
    label.configure(text="Divide into " + str(var) + "blocks")
    label.grid(row=0, column=2, padx=5, pady=5)

    labelMFR.configure(image=img_blks)
    labelMFR.image = img_blks
    labelMFR.grid(row=1, column=2, columnspan=4, padx=5, pady=5)


def startExtraction(var, root):
    # create a progress bar and start the animation
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
    threading.Thread(target=extractFeatures, args=(var,var_result,)).start()

    root.wait_variable(var_result)
    label.configure(text="Completed :)")
    # wait for the child thread to complete
    # tx.wait_variable(var)

    pbar.destroy()
    # popup.destroy()
    # tx.insert(END, var.get())


def extractFeatures(var, var_result):
    var = var.get().split(" ")[0]
    dimension = int(var)
    global df
    column_names = ['image_name', 'image_patch_index', 'corr_max_1', 'corr_max_2', 'corr_max_3', 'corr_max_4',
                    'corr_max_5', 'corr_max_6', 'corr_max_7', 'corr_max_8', 'eudist_1', 'eudist_2', 'eudist_3',
                    'eudist_4', 'eudist_5', 'eudist_6', 'eudist_7', 'eudist_8', 'Variance', 'Entropy']
    df = pd.DataFrame(columns=column_names)

    files = os.listdir(STORAGE)
    files = [file.replace("\\", "/") for file in files]

    counter = 0
    for file in files:
        name = file.replace(".jpg", "")
        name_img = name.split("_")[0]
        index = name.split("_")[1]
        # index = 0
        index = int(index)

        # Stores correlation values between neighbouring matrices and middle matrix
        n_blks_corr = []
        n_blks_eudist = []

        middle_mat = hp.convert_image_matrix(STORAGE + "/" + file)
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

        if dimension == 32:
            if index == 0:
                list_elements = ["16", "17", "1"]
            elif index == 240:
                list_elements = ["-16", "-15", "1"]
            elif index == 15:
                list_elements = ["-1", "15", "16"]
            elif index == 255:
                list_elements = ["-16", "-17", "-1"]
            elif index in [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224]:
                list_elements = ["-16", "-15", "1", "17", "16"]
            elif index in [31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239]:
                list_elements = ["-16", "-17", "-1", "15", "16"]
            elif index >= 1 and index <= 14:
                list_elements = ["-1", "15", "16", "17", "1"]
            elif index >= 241 and index <= 254:
                list_elements = ["-1", "-17", "-16", "-15", "1"]
            else:
                list_elements = ["-17", "-1", "15", "-16", "16", "-15", "1", "17"]
        if dimension == 64:
            if index == 0:
                list_elements = ["8", "9", "1"]
            elif index == 56:
                list_elements = ["-8", "-7", "1"]
            elif index == 7:
                list_elements = ["-1", "7", "8"]
            elif index == 63:
                list_elements = ["-9", "-8", "-1"]
            elif index == 8 or index == 16 or index == 24 or index == 32 or index == 40 or index == 48:
                list_elements = ["-8", "-7", "1", 8, "9"]
            elif index == 15 or index == 23 or index == 31 or index == 39 or index == 47 or index == 55:
                list_elements = ["-8", "-9", "-1", "7", "8"]
            elif index >= 1 and index <= 6:
                list_elements = ["-1", "7", "8", "9", "1"]
            elif index >= 57 and index <= 62:
                list_elements = ["-1", "-9", "-8", "-7", "1"]
            else:
                list_elements = ["-9", "-1", "7", "-8", "8", "-7", "1", "9"]

        if dimension == 128:
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

        filtered_values = ["imageblks_" + str(int(index) + int(x)) + ".jpg" for x in list_elements if
                           0 <= int(index) + int(x) <= 63]

        neighbour_matrices = [hp.convert_image_matrix(STORAGE + x) for regex in filtered_values for x in files if
                              re.search(regex, x)]

        if 8 > len(neighbour_matrices):
            length = 16 - len(filtered_values)
            for i in range(0, length):
                zero_mat = np.zeros((dimension, dimension))
                zero_mat = zero_mat.astype(int)
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

        row = pd.Series([name, index, n_blks_corr[0], n_blks_corr[1], n_blks_corr[2], n_blks_corr[3], n_blks_corr[4],
                         n_blks_corr[5], n_blks_corr[6], n_blks_corr[7], n_blks_eudist[0], n_blks_eudist[1],
                         n_blks_eudist[2], n_blks_eudist[3], n_blks_eudist[4], n_blks_eudist[5], n_blks_eudist[6],
                         n_blks_eudist[7], np.var(MFR_middle_mat), ent], index=df.columns)
        print(row)
        # df = df.append(row, ignore_index=True)
        df.loc[len(df.index)] = row
        # df2 = pd.concat([row, df.loc[:]]).reset_index(drop=True)

        df.to_csv('data.csv', index=False)

        counter += 1
        print("Complete :) " + str(counter))

    var_result.set("Finish :) ")


def classifyForgedBlocks(varType, frame, dimension,labelRegions):
    tampering_type = varType
    dimension = dimension.get().split(" ")[0]
    dimension = int(dimension)

    df = pd.read_csv("data.csv")
    # LOADED MODEL HAVE TO BE CHANGED
    # DUE TO TIME CONSTRAINTS, NEED TO CHANGE THE MODEL MANUALLY
    loaded_model= joblib.load("model/model_128_awgn_30.pkl")


    ref_cols = ['corr_max_1', 'corr_max_2', 'corr_max_3', 'corr_max_4', 'corr_max_5',
               'corr_max_6', 'corr_max_7', 'corr_max_8', 'eudist_1', 'eudist_2', 'eudist_3', 'eudist_4', 'eudist_5',
               'eudist_6', 'eudist_7', 'eudist_8', 'Variance', 'Entropy']

    img_mat = convert_image_matrix(filename)
    image_blks = hp.get_image_blocks(img_mat, (dimension, dimension))

    X_new = df[ref_cols]  # Features
    predictions = loaded_model.predict(X_new)
    print(predictions)
    forged_regions = []
    for index, row in enumerate(df["image_name"]):
        if predictions[index] == "Forged":
            row = row.split("_")[1]
            forged_regions.append(int(row))
            print(forged_regions)

    blk_width = len(image_blks[0])
    blk_height = len(image_blks[1])
    # c = 0
    # i = image_patches_display[0]
    counter = 0
    for h in range(0, img_mat.shape[0], blk_height):
        for w in range(0, img_mat.shape[1], blk_width):
            if counter in forged_regions:
                image_blks[counter][0, :] = 255
                image_blks[counter][1, :] = 255
                image_blks[counter][2, :] = 255
                image_blks[counter][3, :] = 255

                image_blks[counter][blk_width - 2, :] = 255
                image_blks[counter][blk_width - 3, :] = 255
                image_blks[counter][blk_width - 4, :] = 255
                image_blks[counter][blk_width - 5, :] = 255

                image_blks[counter][:, 0] = 255
                image_blks[counter][:, 1] = 255
                image_blks[counter][:, 2] = 255
                image_blks[counter][:, 3] = 255

                image_blks[counter][:, blk_width - 2] = 255
                image_blks[counter][:, blk_width - 3] = 255
                image_blks[counter][:, blk_width - 4] = 255
                image_blks[counter][:, blk_width - 5] = 255

                img_mat = hp.replacePortionOfMatrix(img_mat, image_blks[counter], (w, w + blk_width - 1),
                                                    (h, h + blk_height - 1))
                print(counter)

            counter = counter + 1

    cv2.imwrite('Identified.jpg', img_mat)
    img_mat = Image.open('Identified.jpg')

    img_detect_resized = img_mat.resize((240, 240))  # new width & height
    img_blks = ImageTk.PhotoImage(img_detect_resized)

    labelRegions.configure(image=img_blks)
    labelRegions.image = img_blks
    labelRegions.grid(row=3, column=2, columnspan=4, padx=5, pady=5)

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
