# Credit https://github.com/nima7920/image-blending

import numpy as np
import cv2
import scipy.sparse.linalg as sparse_la
from scipy import sparse

def get_laplacian(img):
    kernel = np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype='float32')
    result = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return result

def generate_matrix_b(source, target, mask):
    source_laplacian_flatten = get_laplacian(source).flatten('C')
    target_flatten = target.flatten('C')
    mask_flatten = mask.flatten('C')
    b = (mask_flatten) * source_laplacian_flatten + (1 - mask_flatten) * target_flatten
    return b

def generate_matrix_A(mask):
    data, cols, rows = [], [], []
    h, w = mask.shape[0], mask.shape[1]
    mask_flatten = mask.flatten('C')
    zeros = np.where(mask_flatten == 0)
    ones = np.where(mask_flatten == 1)
    # adding ones to data
    n = zeros[0].size
    data.extend(np.ones(n, dtype='float32').tolist())
    rows.extend(zeros[0].tolist())
    cols.extend(zeros[0].tolist())

    # adding 4s to data
    m = ones[0].size
    data.extend((np.ones(m, dtype='float32') * (4)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend(ones[0].tolist())

    # adding -1s
    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] - 1).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] + 1).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] - w).tolist())

    data.extend((np.ones(m, dtype='float32') * (-1)).tolist())
    rows.extend(ones[0].tolist())
    cols.extend((ones[0] + w).tolist())
    return data, cols, rows

def solve_sparse_linear_equation(data, cols, rows, b, h, w):
    sparse_matrix = sparse.csc_matrix((data, (rows, cols)), shape=(h * w, h * w), dtype='float32')
    f = sparse_la.spsolve(sparse_matrix, b)
    f = np.reshape(f, (h, w)).astype('float32')
    return f

def blend_images(source, target, mask):
    h, w = source.shape[0], source.shape[1]
    
    data, cols, rows = generate_matrix_A(mask)
    
    b = generate_matrix_b(source, target, mask)
    
    blended = solve_sparse_linear_equation(data, cols, rows, b, h, w)
    
    return blended
