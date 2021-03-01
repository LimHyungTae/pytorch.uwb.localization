import torch
import math
import numpy as np

EPSILON = 0.000000000000001
def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return np.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.rmse = 0.0
        self.mean = 0.0
        self.median = 0.0
        self.var = 0.0
        self.error_max = 0.0
        self.abs_diff = None

    def update(self, rmse, mean, median, var, error_max):
        self.rmse = rmse
        self.mean = mean
        self.median = median
        self.var = var
        self.error_max = error_max

    def evaluate(self, output, target):
        diff = output - target
        self.abs_diff = np.abs(diff)
        self.rmse = math.sqrt(np.mean(np.power(diff, 2)))
        self.var = np.var(self.abs_diff)
        self.mean = np.mean(self.abs_diff)
        self.median = np.median(self.abs_diff)
        self.error_max = np.amax(self.abs_diff)

class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.is_initial = True
        self.abs_diff = None

    def reset(self):
        self.count = 0.0
        self.sum_rmse = 0.0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        if self.is_initial:
            self.abs_diff = result.abs_diff
            self.is_initial = False
        else:
            self.abs_diff = np.concatenate((self.abs_diff, result.abs_diff), axis=0)
        self.sum_rmse += n*result.rmse

    def average(self):
        avg = Result()
        var = np.var(self.abs_diff)
        mean = np.mean(self.abs_diff)
        median = np.median(self.abs_diff)
        error_max = np.amax(self.abs_diff)
        avg.update(self.sum_rmse / self.count, mean, median, var, error_max)
        return avg