from statistics import mean, stdev
from math import sqrt
from scipy.stats.distributions import t
import numpy as np


class TTest:

    def __init__(self,
                 alpha=0.05):

        self._validate_alpha(alpha)

    def _validate_alpha(self, alpha):

        if isinstance(alpha, float):
            self.alpha = alpha
        else:
            raise TypeError(
                f'alpha should be a float, but it is a {type(alpha)} instead'
            )

    def compute_t(self, sample_1, sample_2):

        mean_1, mean_2 = mean(sample_1), mean(sample_2)
        std_1, std_2 = stdev(sample_1), stdev(sample_2)
        n1, n2 = len(sample_1), len(sample_2)
        std_error_1, std_error_2 = std_1/sqrt(n1), std_2/sqrt(n2)
        std_error_dif = sqrt(std_error_1**2.0 + std_error_2**2.0)
        t_stat = (mean_1 - mean_2) / std_error_dif

        degrees_freedom = n1 + n2 - 2
        critical_value = t.ppf(1.0 - self.alpha, degrees_freedom)

        # calculate the p-value
        p_value = (1 - t.cdf(abs(t_stat), degrees_freedom)) * 2

        return t_stat, p_value, degrees_freedom, critical_value
