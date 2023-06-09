#https://medium.com/analytics-vidhya/image-convolution-from-scratch-d99bf639c32a
#https://en.wikipedia.org/wiki/Kernel_(image_processing)
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import math

#notice that most of the pictures are square matrices 

#returns 2D image numpy array 
def convert_image_matrix(img_name):
    src = cv2.imread(img_name)

    #convert to gray scale color 
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print(img)
    #saves the name and the extension of the image name - fruits_platter then ext - jpg
    name, ext = img_name.split('.')


    #saves the array as img files
    # plt.imsave(str(name + '_gray.' + ext), img, cmap='gray')
    plt.imsave(img_name, img)
    gray_img = cv2.imread(str(name + '_gray.' + ext), 0)
    gimg_shape = gray_img.shape
    gimg_mat = []
    for i in range(0, gimg_shape[0]):
        row = []
        for j in range(0, gimg_shape[1]):
            pixel = gray_img.item(i, j)
            row.append(pixel)
        gimg_mat.append(row)
    gimg_mat = np.array(gimg_mat)
    return gimg_mat

#returns a giant matrix containing smaller matrices (size- kernel size) -> sampled matrix
def get_sub_matrices(orig_matrix, kernel_size):
    width = len(orig_matrix[0])
    height = len(orig_matrix)
    if kernel_size[0] == kernel_size[1]:
        if kernel_size[0] > 2:
            orig_matrix = np.pad(orig_matrix, kernel_size[0] - 2, mode='constant')
        else: pass
    else: pass
    
    giant_matrix = []
    for i in range(0, height - kernel_size[1] + 1):
        for j in range(0, width - kernel_size[0] + 1):
            giant_matrix.append(
                [
                    [orig_matrix[col][row] for row in range(j, j + kernel_size[0])]
                    for col in range(i, i + kernel_size[1])
                ]
            )
    img_sampling = np.array(giant_matrix)
    return img_sampling

#sampled matrix -> passed, kernel filter passed to perform convolution
def get_transformed_matrix(matrix_sampling, kernel_filter):
    transform_mat = []
    for each_mat in matrix_sampling:
        #multipy the each smaller matrix with kernel filter and sum the values up for 1 pixel
        transform_mat.append(
            np.sum(np.multiply(each_mat, kernel_filter))
        )
    #matrix_sampling.shape[0] - number of rows
    #matrix_sampling.shape - e.g. (3,8) -> 3 - rows and 8 is columns 
    reshape_val = int(math.sqrt(matrix_sampling.shape[0]))
    #returns an array containing the same data with a new shape 
    transform_mat = np.array(transform_mat).reshape(reshape_val, reshape_val)
    return transform_mat

def original_VS_convoluted(img_name, kernel_name, convoluted_matrix):
    name, ext = img_name.split('.')
    cv2.imwrite(str(name + '_' + kernel_name + '.' + ext), convoluted_matrix)
    orig = cv2.imread(str(name + '_gray.' + ext))
    conv = cv2.imread(str(name + '_' + kernel_name + '.' + ext))
    
    fig = plt.figure(figsize=(16, 25))
    ax1 = fig.add_subplot(2,2,1)
    ax1.axis("off")
    ax1.title.set_text('Original')
    ax1.imshow(orig)
    ax2 = fig.add_subplot(2,2,2)
    ax2.axis("off")
    ax2.title.set_text(str(kernel_name).title())
    ax2.imshow(conv)
    plt.show()
    return True

# print(convert_image_matrix('women.jpg'))

# (1/9)*np.array([[1,1,1],[1,1,1],[1,1,1]]) -> average filtering
# identity_kernel = (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]]) -> gaussian filtering
img_name = '0.pgm'
img_mat = convert_image_matrix(img_name)
# identity_kernel = identity_kernel = (1/9)*np.array([[1,1,1],[1,1,1],[1,1,1]])
# img_sampling = get_sub_matrices(img_mat, identity_kernel.shape)
# transform_mat = get_transformed_matrix(img_sampling, identity_kernel)
# original_VS_convoluted(img_name,'sharpen', transform_mat)