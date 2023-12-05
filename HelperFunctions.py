import cv2
import numpy as np
import math

# returns 2D image numpy array
def convert_image_matrix(img_name):
    # convert to gray scale color
    src = cv2.imread(img_name)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # saves the array as img files
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
def get_image_blocks(mat, dimensions):
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


def separateOriginalAndTamperedBlocks(mat, dimensions):
    blks = get_image_blocks(mat, dimensions)
    blks_length = len(blks)
    midpoint = int(blks_length / 2)

    tamperedBlks = [blks[i] for i in range(0, midpoint)]
    originalBlks = [blks[i] for i in range(midpoint, blks_length)]

    return tamperedBlks, originalBlks


def retrievePortionOfMatrix(mat, range_row, range_col):
    start_row, end_row = range_row
    start_col, end_col = range_col

    gimg_mat = []
    for i in range(start_row, end_row + 1):
        row = []
        for j in range(start_col, end_col + 1):
            pixel = mat.item(i, j)
            row.append(pixel)
        gimg_mat.append(row)
    gimg_mat = np.array(gimg_mat)
    return gimg_mat


def replacePortionOfMatrix(orig_mat, portion_mat, range_row, range_col):
    start_row, end_row = range_row
    start_col, end_col = range_col
    portion_mat[np.isnan(portion_mat)] = 0

    row = 0
    column = 0
    for i in range(start_row, end_row + 1):
        for j in range(start_col, end_col + 1):
            orig_mat[i][j] = portion_mat[row][column]
            column = column + 1
        row = row + 1
        column = 0
    return orig_mat


def combine(orig_matrix, left_region_matrix):
    left_region_matrix = left_region_matrix.astype(np.uint8)
    row, column = left_region_matrix.shape
    # column = len(left_region_matrix)
    for i in range(0, row):
        for j in range(0, column):
            orig_matrix[i][j] = left_region_matrix[i][j]
    return orig_matrix


def centerCropMatrix(mat, crop_dimensions):
    width, height = mat.shape
    crop_width, crop_height = crop_dimensions

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_mat = mat[mid_x - cw2:mid_x + cw2, mid_y - ch2:mid_y + ch2]

    return crop_mat


def get_left_side_of_the_image(orig_matrix):
    noOfRows = len(orig_matrix[0])
    noOfColumns = len(orig_matrix)
    halfOfColumns = int(noOfColumns / 2)
    gimg_mat = []
    for i in range(0, noOfRows):
        row = []
        for j in range(0, halfOfColumns):
            pixel = orig_matrix.item(i, j)
            row.append(pixel)
        gimg_mat.append(row)
    gimg_mat = np.array(gimg_mat)
    return gimg_mat


def get_overlapped_image_blocks(orig_matrix, blksize):
    width = len(orig_matrix[0])
    height = len(orig_matrix)
    giant_matrix = []
    for i in range(0, height):
        for j in range(0, width):

            if j + blksize[0] < width and i + blksize[1] < height:
                giant_matrix.append(
                    [
                        [orig_matrix[col][row] for row in range(j, j + blksize[0])]
                        for col in range(i, i + blksize[1])
                    ]
                )
    img_sampling = np.array(giant_matrix)
    return img_sampling

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

