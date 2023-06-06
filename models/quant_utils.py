from typing import List, Tuple

import numpy as np
from numpy import float64, int64, ndarray


def unquantize_matrix(
    matrix: ndarray, bit_width: int = 8, r_max: float64 = 0.5
) -> ndarray:
    matrix = matrix.astype(int)
    og_sign = np.sign(matrix)
    uns_matrix = np.multiply(matrix, og_sign)
    uns_result = np.multiply(
        uns_matrix, np.divide(r_max, (pow(2, bit_width - 1) - 1.0))
    )
    result = og_sign * uns_result
    return result.astype(np.float32)


def quantize_matrix_stochastic(
    matrix: ndarray, bit_width: int = 8, r_max: float64 = 0.5
) -> Tuple[ndarray, ndarray]:
    og_sign = np.sign(matrix)
    uns_matrix = np.multiply(matrix, og_sign)
    uns_result = np.multiply(
        uns_matrix, np.divide((pow(2, bit_width - 1) - 1.0), r_max)
    )
    result = np.multiply(og_sign, uns_result)
    return result, og_sign


def quantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        x, _ = quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
        result.append(x)

    return np.array(result)


def clip_with_threshold(grads, thresholds):
    return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]


def get_alpha_gaus(values: ndarray, values_size: int64, num_bits: int) -> float64:
    """
    Calculating optimal alpha(clipping value) in Gaussian case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    """

    # Dictionary that stores optimal clipping values for N(0, 1)
    alpha_gaus = {
        2: 1.71063516,
        3: 2.02612148,
        4: 2.39851063,
        5: 2.76873681,
        6: 3.12262004,
        7: 3.45733738,
        8: 3.77355322,
        9: 4.07294252,
        10: 4.35732563,
        11: 4.62841243,
        12: 4.88765043,
        13: 5.1363822,
        14: 5.37557768,
        15: 5.60671468,
        16: 5.82964388,
        17: 6.04501354,
        18: 6.25385785,
        19: 6.45657762,
        20: 6.66251328,
        21: 6.86053901,
        22: 7.04555454,
        23: 7.26136857,
        24: 7.32861916,
        25: 7.56127906,
        26: 7.93151212,
        27: 7.79833847,
        28: 7.79833847,
        29: 7.9253003,
        30: 8.37438905,
        31: 8.37438899,
        32: 8.37438896,
    }
    # That's how ACIQ paper calculate sigma, based on the range (efficient but not accurate)
    gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
    # sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / (
        (2 * np.log(values_size)) ** 0.5
    )
    return alpha_gaus[num_bits] * sigma


def calculate_clip_threshold_aciq_g(
    grads: ndarray, grads_sizes: List[int64], bit_width: int = 8
) -> List[float64]:
    res = []
    for idx in range(len(grads)):
        res.append(get_alpha_gaus(grads[idx], grads_sizes[idx], bit_width))
    # return [aciq.get_alpha_gaus(x, bit_width) for x in grads]
    return res
