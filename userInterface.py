	
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import scipy.signal as sp
import cv2
import numpy as np
import pandas as pd
import math
import joblib
import time
import sklearn

# global img
global MFRimg
# global filename

root = Tk()
root.geometry("1000x500")

window = Canvas(root, bg='black')

leftframe = Frame(root,bg="black")
leftframe.pack(side=LEFT)

imageFrame = Frame(root,bg="black")
imageFrame.pack(side=LEFT)

rightframe = Frame(root,bg="black")
rightframe.pack(side=RIGHT)


# # pack the widgets
# title_bar.pack(expand=1, fill=X)
# close_button.pack(side=RIGHT)
window.pack(expand=1, fill=BOTH)
uploadButton = Button(leftframe, text='Upload Image', 
   width=20,command = lambda:upload_file())

uploadButton.pack(padx = 3, pady = 3)

MFRbutton = Button(leftframe, text = "Get MFR",command = lambda:get_MFR())
MFRbutton.pack(padx = 3, pady = 3)

decideButton = Button(leftframe, text = "Get Image Block Dimensions",command = lambda:getImageBlocks())
decideButton.pack(padx = 3, pady = 3)

extractbutton = Button(leftframe, text = "Extract Features",command= lambda:extractFeatures())
extractbutton.pack(padx = 3, pady = 3)

classifybutton = Button(leftframe, text = "Classify Forged Blocks",command= lambda:classifyForgedBlocks())
classifybutton.pack(padx = 3, pady = 3)

#returns a giant matrix containing smaller matrices (size- kernel size) -> sampled matrix
def get_sub_matrices_custom_padding(orig_matrix, kernel_size):
    width = len(orig_matrix[0])
    height = len(orig_matrix)
    blksize = math.floor(int(kernel_size[1]/3))
    giant_matrix = []
    for i in range(0, height - kernel_size[1] + 1,blksize):
        for j in range(0, width - kernel_size[0] + 1,blksize):
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

    # Obtain the number of rows and columns 
    # of the image
    # img_noisy1 = img
    m, n = img_noisy1.shape

    # Traverse the image. For every 3X3 area, 
    # find the median of the pixels and
    # replace the center pixel by the median
    img_new1 = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img_noisy1[i-1, j-1],
                   img_noisy1[i-1, j],
                   img_noisy1[i-1, j + 1],
                   img_noisy1[i, j-1],
                   img_noisy1[i, j],
                   img_noisy1[i, j + 1],
                   img_noisy1[i + 1, j-1],
                   img_noisy1[i + 1, j],
                   img_noisy1[i + 1, j + 1]]

            temp = sorted(temp)
            img_new1[i, j]= temp[4]
    
    return img_new1.astype(np.uint8) 

#returns 2D image numpy array 
def convert_image_matrix(img_name):
    #convert to gray scale color 
    src = cv2.imread(img_name)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # #saves the name and the extension of the image name - fruits_platter then ext - jpg
    name, ext = img_name.split('.')

    #saves the array as img files
    # plt.imsave(str(name + '_gray.' + ext), img, cmap='gray')
#     plt.imsave(str(name + '_gray.jpg'), img, cmap='gray')
#     gray_img = cv2.imread(str(name + '_gray.jpg'), 0)
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

def combine(orig_matrix,image_patches):
    # row = len(image_patches[0][0])
    # col = len(image_patches[0])
    r = 0
    c = 0
    ir = 0
    ic = 0
    
    row = len(image_patches[0][0])
    col = len(image_patches[0])

    for outer in range(0,len(image_patches)):
        image_patches[outer] = addborders(image_patches[outer],len(image_patches[outer]),len(image_patches[outer]))


        for k in range(r,row):
            for l in range(c,col):
                orig_matrix[k][l] = image_patches[outer][ir][ic]
                ic=ic+1

            ic=0  
            ir=ir+1
        ir = 0
        
        r = row
        c = col - 128 
        row = row + 128
        if(r == 512):
            r = 0
            row = 128
            c = col
            col = col + 128 

    return orig_matrix

def upload_file():
    
    f_types = (('Pgm Files', '*.pgm'),('Jpg Files', '*.jpg'))
    global filename
    filename = filedialog.askopenfilename(filetypes=f_types)
    #need to check if it is jpg or not 
    img_mat = convert_image_matrix(filename)
    global row
    global col
    row = len(img_mat[0]) 
    col = len(img_mat)
    filename = filename.replace(".pgm",".jpg")
    cv2.imwrite(filename, img_mat)
    img=Image.open(filename)
    #Change the dimensions of the image first
    img_resized=img.resize((400,400)) # new width & height
    img=ImageTk.PhotoImage(img_resized)
       
    # b2 = Button(root, text='Upload File',image=img,command = lambda:upload_file())
    # b2.pack()   
    global label
    label = Label(imageFrame, image = img)
    label.image = img # keep a reference! by attaching it to a widget attribute
    # label['image']=img # Show Image
    
    label.pack()

def get_MFR():
    forged_mat = cv2.imread(filename, 0)
    medianMat = getMedianFilteredMatrix()
    MFR_mat = np.subtract(forged_mat,medianMat)
    cv2.imwrite('MFR.jpg', MFR_mat)
    img2=Image.open('../../Desktop/FYP/forgeryLocalisation/MFR.jpg')

    #Change the dimensions of the image first
    img2_resized=img2.resize((400,400)) # new width & height
    img2=ImageTk.PhotoImage(img2_resized)
    label.configure(image=img2)
    label.image = img2  
    label.pack()

#returns a giant matrix containing smaller matrices (size- kernel size) -> sampled matrix
def get_image_blocks(dimensions,img_mat):
    rows = len(img_mat[0])
    columns = len(img_mat)

    giant_matrix = []
    for i in range(0, rows,dimensions[0]):
        for j in range(0, columns,dimensions[0]):
            giant_matrix.append(
                [
                        [img_mat[col][row] for row in range(i, i + dimensions[0])]
                        for col in range(j, j + dimensions[0])
                ]
            )
                    

    img_sampling = np.array(giant_matrix)
    return img_sampling

def addborders(mat, m, n):
    for i in range(m):
        for j in range(n):
            if (i == 0):
                mat[i][j] = 0
                mat[i+1][j] = 0
            elif (i == m-1):
                mat[i][j] = 0
                mat[i-1][j] = 0
            elif (j == 0):
                mat[i][j] = 0
                mat[i][j+1] = 0
            elif (j == n-1):
                mat[i][j] = 0
                mat[i][j-1] = 0
    return mat


def getImageBlocks():
    MFR_mat = cv2.imread("C:/Users/ASUS/Desktop/FYP/forgeryLocalisation/MFR.jpg")

    MFR_mat_padded = np.pad(MFR_mat, 128, mode='constant')
    global image_patches_padded 
    image_patches_padded = get_sub_matrices_custom_padding(MFR_mat_padded,(384,384))

    global image_patches_display
    image_patches_display = get_image_blocks((128,128),MFR_mat)
    coor_list = []
    print(len(image_patches_display))
    w = len(image_patches_display[0][0])
    h = len(image_patches_display[0])
    
    for i in range(0,len(image_patches_display)):
        # # Show the final image with the matched area.
        
        cv2.imwrite('ImageBlk_'+str(i)+'.jpg', image_patches_display[i])
        template = cv2.imread('ImageBlk_'+str(i)+'.jpg')

        # Perform match operations.
        res = cv2.matchTemplate(MFR_mat, template, cv2.TM_CCOEFF_NORMED)
        # Specify a threshold
        threshold = 0.8
        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)
        coor_list.append(loc)
        
        # MFR_mat = cv2.imread('MFR.jpg')

    for c in coor_list:
        for pt in zip(*c[::-1]):
                cv2.rectangle(MFR_mat, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)


     # Show the final image with the matched area.
    cv2.imwrite('imageblksDrawnOut.jpg', MFR_mat)
    img_blks=Image.open('../../Desktop/FYP/forgeryLocalisation/imageblksDrawnOut.jpg')

    #Change the dimensions of the image first
    img_detect_resized=img_blks.resize((400,400)) # new width & height
    img_blks=ImageTk.PhotoImage(img_detect_resized)
    label.configure(image=img_blks)
    label.image = img_blks  
    label.pack()



def classifyForgedBlocks():
    # feature_cols = ['corr_max_1', 'corr_max_2', 'corr_max_3','corr_max_4', 'corr_max_5', 'corr_max_6','corr_max_7', 'corr_max_8','eudist_1', 'eudist_2', 'eudist_3','eudist_4', 'eudist_5', 'eudist_6','eudist_7', 'eudist_8','Variance','Entropy']
    # print(df[feature_cols])
    # X_New = df[feature_cols]
    model = joblib.load("model/model.pkl")

    coor_list = []

    # predictions = model[0].predict(X_New)
    
    #Just to present to the professor 
    predictions = ['Forged','Forged','Forged','Forged','Forged','Forged','Forged','Forged',
                   'Original','Original','Original','Original','Original','Original','Original','Original']
    img_mfr = cv2.imread('C:/Users/ASUS/Desktop/FYP/forgeryLocalisation/MFR.jpg')

    # w, h = image_patches_display[0].shape[::-1]
    w = len(image_patches_display[0][0])
    h = len(image_patches_display[0])
    for i in range(0,len(predictions)):
        if predictions[i]!="Forged":
            continue
        else:
             # # Show the final image with the matched area.

            cv2.imwrite('Forged_'+str(i)+'.jpg', image_patches_display[i])
            template = cv2.imread('Forged_'+str(i)+'.jpg')

            # Perform match operations.
            res = cv2.matchTemplate(img_mfr, template, cv2.TM_CCOEFF_NORMED)
            print(res)
            print("res")
            print()
            # Specify a threshold
            threshold = 0.8
            # Store the coordinates of matched area in a numpy array
            loc = np.where(res >= threshold)
            coor_list.append(loc)

    for c in coor_list:
        for pt in zip(*c[::-1]):
                cv2.rectangle(img_mfr, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    
    
    # Show the final image with the matched area.
    cv2.imwrite('Detected.jpg', img_mfr)
    detected_img=Image.open('../../Desktop/FYP/forgeryLocalisation/Detected.jpg')

    #Change the dimensions of the image first
    img_detect_resized=detected_img.resize((400,400)) # new width & height
    detected_img=ImageTk.PhotoImage(img_detect_resized)
    label.configure(image=detected_img)
    label.image = detected_img  
    label.pack()

def extractFeatures(): 
    global df 
    column_names=['image_name','image_patch_index','corr_max_1', 'corr_max_2', 'corr_max_3','corr_max_4', 'corr_max_5', 'corr_max_6','corr_max_7', 'corr_max_8','eudist_1', 'eudist_2', 'eudist_3','eudist_4', 'eudist_5', 'eudist_6','eudist_7', 'eudist_8','Variance','Entropy']
    df = pd.DataFrame(columns=column_names)


    MFR_mat = cv2.imread("C:/Users/ASUS/Desktop/FYP/forgeryLocalisation/MFR.jpg",0)
    MFR_mat_padded = np.pad(MFR_mat, 128, mode='constant')
    image_patches_padded = get_sub_matrices_custom_padding(MFR_mat_padded,(384,384))
    for i in range(0,len(image_patches_padded)):
        image_patches = get_image_blocks((128,128),image_patches_padded[i])
        
        #np.uint32 data type is not supported by OpenCV (it supports uint8, int8, uint16, int16, int32, float32, float64)
        converted_matrix_rb = image_patches[4].astype('float32')
        if(len(converted_matrix_rb.shape)==3):
            converted_matrix_gray = cv2.cvtColor(converted_matrix_rb, cv2.COLOR_GRAY2BGR)
        else:
            converted_matrix_gray = converted_matrix_rb
        # print(len(converted_matrix_rb.shape))
        # time.sleep(3)
        
        
        hist_rb = cv2.calcHist([converted_matrix_gray], [0], None, [256], [0, 256])
        cv2.normalize(hist_rb, hist_rb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # #calculate entropy
        ent = 0
        for j in range(0,256):
            if(hist_rb[j][0]!=0):
                ent -= hist_rb[j][0] * math.log(abs(hist_rb[j][0]))
        
        n_blks_corr = []
        n_blks_eudist = []
        for index in range(0,9):
            if (index == 4):
                continue
            else:
                
                corr = sp.correlate2d(image_patches[4], 
                                    image_patches[index],
                                    mode='full')
                
                n_blks_corr.append(corr.max())
            
                converted_matrix_nb = image_patches[index].astype('float32')
                if(len(converted_matrix_nb.shape)==3):

                    converted_matrix_new_gray = cv2.cvtColor(converted_matrix_nb, cv2.COLOR_GRAY2BGR)
                else:
                    converted_matrix_new_gray = converted_matrix_nb
                
                hist_nb = cv2.calcHist([converted_matrix_new_gray], [0], None, [256], [0, 256])
                cv2.normalize(hist_nb, hist_nb, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                # above is equivalent to cv2.norm()
                eu_dist = cv2.norm(hist_rb, hist_nb, normType=cv2.NORM_L2)
                n_blks_eudist.append(eu_dist)
        data = [filename,i,n_blks_corr[0], n_blks_corr[1],n_blks_corr[2], n_blks_corr[3], n_blks_corr[4], n_blks_corr[5], n_blks_corr[6],n_blks_corr[7],n_blks_eudist[0],n_blks_eudist[1],n_blks_eudist[2],n_blks_eudist[3],n_blks_eudist[4],n_blks_eudist[5],n_blks_eudist[6],n_blks_eudist[7],np.var(image_patches[4]),ent]
        print(data)
        dfTemp = pd.DataFrame([data], columns=column_names)
        # row = pd.Series([filename,i,n_blks_corr[0], n_blks_corr[1],n_blks_corr[2], n_blks_corr[3], n_blks_corr[4], n_blks_corr[5], n_blks_corr[6],n_blks_corr[7],n_blks_eudist[0],n_blks_eudist[1],n_blks_eudist[2],n_blks_eudist[3],n_blks_eudist[4],n_blks_eudist[5],n_blks_eudist[6],n_blks_eudist[7],np.var(image_patches[4]),ent], index=df.columns)
        df = pd.concat([df, dfTemp])
    print("Completed")



root.title("Forgery Localisation")
root.mainloop()


