#Median Filtering
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
from PIL import Image
from matplotlib import cm

import glymur

def getMedianFilteredMatrix_win3x3(mat):
    # Read the image
    # Obtain the number of rows and columns of the image (m * n )
    mat_temp = mat
    m = len(mat_temp[0])
    n = len(mat_temp)

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and
    # replace the center pixel by the median
    mat_tampered = np.zeros([m, n])

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = [mat_temp[i - 1, j - 1],
                    mat_temp[i - 1, j],
                    mat_temp[i - 1, j + 1],
                    mat_temp[i, j - 1],
                    mat_temp[i, j],
                    mat_temp[i, j + 1],
                    mat_temp[i + 1, j - 1],
                    mat_temp[i + 1, j],
                    mat_temp[i + 1, j + 1]]

            temp = sorted(temp)
            mat_tampered[i, j] = temp[4]

    #convert matrix value type to int
    return mat_tampered.astype(np.uint8)


def getMedianFilteredMatrix_win5x5(mat):
    # Read the image
    mat_temp = mat

    # Obtain the number of rows and columns
    # of the image
    m, n = mat_temp.shape

    img_new1 = np.zeros([m, n])

    for i in range(2, m - 2):
        for j in range(2, n - 2):
            temp = [
                # mat_temp[i - 1, j - 1],
                #     mat_temp[i - 1, j],
                #     mat_temp[i - 1, j + 1],
                #     mat_temp[i - 2, j - 2],
                #     mat_temp[i - 2, j],
                #     mat_temp[i - 2, j + 2],

                    mat_temp[i - 2, j - 2],
                    mat_temp[i - 2, j - 1],
                    mat_temp[i - 2, j],
                    mat_temp[i - 2, j + 1],
                    mat_temp[i - 2, j + 2],

                    mat_temp[i - 1, j - 2],
                    mat_temp[i - 1, j - 1],
                    mat_temp[i - 1, j],
                    mat_temp[i - 1, j + 1],
                    mat_temp[i - 1, j + 2],

                    mat_temp[i, j - 2],
                    mat_temp[i, j - 1],
                    mat_temp[i, j],
                    mat_temp[i, j + 1],
                    mat_temp[i, j + 2],

                    mat_temp[i + 1, j - 2],
                    mat_temp[i + 1, j - 1],
                    mat_temp[i + 1, j],
                    mat_temp[i + 1, j + 1],
                    mat_temp[i + 1, j + 2],

                    mat_temp[i + 2, j - 2],
                    mat_temp[i + 2, j - 1],
                    mat_temp[i + 2, j],
                    mat_temp[i + 2, j + 1],
                    mat_temp[i + 2, j + 2],

                    # mat_temp[i - 2, j],
                    # mat_temp[i - 1, j],
                    # mat_temp[i, j],
                    # mat_temp[i + 1, j],
                    # mat_temp[i + 2, j],
                    #
                    # mat_temp[i + 1, j - 1],
                    # mat_temp[i + 1, j],
                    # mat_temp[i + 1, j + 1],
                    # mat_temp[i + 2, j - 2],
                    # mat_temp[i + 2, j],
                    # mat_temp[i + 2, j + 2]
            ]

            temp = sorted(temp)
            img_new1[i, j] = temp[12]

    return img_new1.astype(np.uint8)

#Gaussian Filtering
#The Gaussian kernel will have size 2*radius + 1 along each axis where radius = round(truncate * sigma)
def getGaussianFilteredMatrix_win3x3(mat):
    radius = int((3 - 1) / 2)
    filter_mat = gaussian_filter(mat.astype(np.uint8), sigma=1.1,radius=radius)
    return filter_mat

def getGaussianFilteredMatrix_win5x5(mat):
    radius = int((5 - 1) / 2)
    filter_mat = gaussian_filter(mat.astype(np.uint8), sigma=1.1,radius=radius)
    return filter_mat

#Wiener Filtering
def getWienerFilteredMatrix_win3x3(mat):
    filtered_mat = wiener(mat.astype(np.uint8), (3, 3))
    return filtered_mat

def getWienerFilteredMatrix_win5x5(mat):
    filtered_mat = wiener(mat.astype(np.uint8), (5, 5))
    return filtered_mat

#Average Filtering
def getAverageFilteringMatrix_win3x3(mat):
    filtered_mat = cv2.blur(mat.astype(np.uint8),(3, 3))
    return filtered_mat

def getAverageFilteringMatrix_win5x5(mat):
    filtered_mat = cv2.blur(mat.astype(np.uint8),(5, 5))
    return filtered_mat


#Contrast Enhancement
def getContrastEnhancedMatrix(mat,ce_factor):
    adjusted_mat = cv2.convertScaleAbs(mat.astype(np.uint8), alpha=ce_factor)
    return adjusted_mat


#JPEG compression
# Quality Factor = 80,90
def getJPEGCompressedImage(mat,quality_factor,new_path):
    # mat = mat/255
    pil_image = Image.fromarray(np.uint8(mat))
    # pil_image = Image.open(name)
    pil_image.save(new_path, 'JPEG', optimize=True, quality=quality_factor)

#Rescaling
def getRescaledMatrix(mat,scale):
    width = int(len(mat[0]) * scale)
    height = int(len(mat) * scale)

    dsize = (width, height)

    mat = mat.astype('float32')
    # Bilinear interpolation -> estimating unknown values using related known values
    # -> using known pixel values we estimate the pixel value at any particular location in the resized image
    # resize image (bilinear interpolation)
    resize_img = cv2.resize(mat, dsize)
    cropped_mat = resize_img[0:len(mat), 0:len(mat[0])]

    # cv2.imwrite("cropped.png", crop_img)
    return cropped_mat


#Additive White Gaussian Noise
def getGaussianNoiseMatrix(mat,factor):
    #normalise matrix
    mat = cv2.normalize(mat.astype(np.uint8), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mean=0

    #standard deviation -> how the data spreads out
    scale=np.sqrt(factor)
    x,y = mat.shape

    gaussian_noise = np.random.normal(loc=mean, scale=scale, size=(x, y))
    # cv2.imshow('gaussian_noise', gaussian_noise)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    filtered_mat = mat + gaussian_noise
    #
    # cv2.imshow('gaussian', filtered_mat)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return filtered_mat

#Unsharp masking
# sharpened = original + (original − blurred) × amount
def getUnsharpMaskingMatrix(mat,strengthAmount):
    radius = (3 - 1) / 2
    radius = int(radius)
    # normalise matrix
    mat = mat / 255
    gaussian_mat = gaussian_filter(mat, sigma=1.1, radius=radius)
    detail_mat = mat - gaussian_mat
    result = mat + (strengthAmount*detail_mat)
    return result

#JPEG 2000
#pip install Glymur
def getJPEG2000Image(mat,quality_factor,new_path):
    glymur.lib.openjpeg_library = r'C:/Users/ASUS/openjpeg-v2.5.0-windows-x64/bin/openjp2.dll'
    mat = mat.astype(np.uint8)
    # pil_image = Image.open(name)
    # print(name)

    # jp2_image = glymur.Jp2k('0_jpeg2000compressed.jp2')

    # Read the image data as a NumPy array
    # image_data = jp2_image[:]

    # Create a PIL Image object from the image data
    pil_image = Image.fromarray(mat)
    # x,y = mat.shape
    # Resize the image using PIL
    # resized_image = pil_image.resize((x,y))

    # Convert the resized image back to a NumPy array
    resized_data = np.array(pil_image)

    # Save the resized image using Glymur
    # output_jp2_image = 'path_to_output_image.jp2'
    glymur.Jp2k(new_path, data=resized_data,cratios=[quality_factor])

    # jp2_image = glymur.Jp2k('path_to_jp2_image.jp2')

    # Read the image data as a NumPy array
    # image_data = jp2_image[:]

    # Create a PIL Image object from the image data
    # pil_image = Image.fromarray(image_data)

    # Resize the image using PIL
    # resized_image = pil_image.resize((512, 512))

    # Convert the resized image back to a NumPy array
    # resized_data = np.array(resized_image)


    # pil_image = Image.open(name)


    # jp2_image = glymur.Jp2k('0_jpeg2000compressed.jp2')

    # Read the image data as a NumPy array
    # image_data = jp2_image[:]

    # Create a PIL Image object from the image data
    # pil_image = Image.fromarray(image_data)

    # # Resize the image using PIL
    # resized_image = pil_image.resize((512, 512))

    # Convert the resized image back to a NumPy array
    # data = np.array(pil_image)
    # jp2 = glymur.Jp2k('myfile.jp2', data=resized_data, cratios=[20, 10, 1])
    # jp2 = glymur.Jp2k(new_path + name, data=data,cratios=[quality_factor])
