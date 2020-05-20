import numpy as np

def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    hout = int((input_matrix_shape[2] + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1)
    wout = int((input_matrix_shape[3] + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1)

    return [input_matrix_shape[0], out_channels, hout, wout]

print(np.array_equal(
    calc_out_shape(input_matrix_shape=[2, 3, 10, 10],
                   out_channels=10,
                   kernel_size=3,
                   stride=1,
                   padding=0),
    [2, 10, 8, 8]))

# ... и ещё несколько подобных кейсов