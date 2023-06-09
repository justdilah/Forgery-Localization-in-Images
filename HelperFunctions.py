import cv2
import numpy as np

#returns 2D image numpy array
def convert_image_matrix(img_name):
    #convert to gray scale color
    src = cv2.imread(img_name)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    #saves the array as img files
    mat_shape = img.shape
    mat = []
    for i in range(0, mat_shape[0]):
        row = []
        for j in range(0, mat_shape[1]):
            pixel = img.item(i, j)
            row.append(pixel)
        mat.append(row)
    mat = np.array(mat)
    return mat


# returns a giant matrix containing smaller matrices (size- kernel size) -> sampled matrix
def get_image_blocks(mat,dimensions):
    rows = len(mat[0])
    columns = len(mat)

    giant_matrix = []
    for i in range(0, rows, dimensions[0]):
        for j in range(0, columns, dimensions[0]):
            giant_matrix.append(
                [
                    [mat[col][row] for row in range(i, i + dimensions[0])]
                    for col in range(j, j + dimensions[0])
                ]
            )

    img_sampling = np.array(giant_matrix)
    return img_sampling

def separateOriginalAndTamperedBlocks(mat,dimensions):
    blks = get_image_blocks(mat,dimensions)
    blks_length = len(blks)
    midpoint = int(blks_length / 2)

    tamperedBlks = [blks[i] for i in range(0,midpoint)]
    originalBlks = [blks[i] for i in range(midpoint,blks_length)]

    return tamperedBlks,originalBlks


