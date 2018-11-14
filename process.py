import pandas as pd
import numpy as np
import os


def stretch(x):
    percent = np.percentile(x, [2, 98], axis=[0, 1])
    print(percent)
    x = np.clip(x, percent[0, :], percent[1, :])
    x = (x - percent[0, :]) / (percent[1, :] - percent[0, :]) * 255
    x = np.round(x).astype(np.uint8)
    return x


if __name__ == '__main__':
    pass
