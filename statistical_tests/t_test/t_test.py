from random import sample
from statistics import mean, stdev
from math import sqrt
from scipy.stats.distributions import t
import numpy as np


class TTest:

    """
    A class to apply different kinds 
    of t-tests

    ...

    Attributes
    ----------
    alpha : float
        significance to be used in the test
        to calculate critical value

    Methods
    -------
    one_sample_t_test(sample, mean_value):
        Apply one sample t-test
    independent_t_test(sample_1, sample_2):
        Apply one sample t-test
    """

    def __init__(self,
                 alpha=0.05):

        self._validate_alpha(alpha)

    def _validate_alpha(self, alpha):

        """
        Validate the alpha parameter

        Parameters
        ----------
        alpha : float
        The alpha level to be used to calculate t statistics

        Raises
        ----------
        TypeError:
            When alpha is not a float
        """

        if isinstance(alpha, float):
            self.alpha = alpha
        else:
            raise TypeError(
                f'alpha should be a float, but it is a {type(alpha)} instead'
            )

    def one_sample_t_test(self, sample, mean_value):

        """
        Apply one sample t-test

        Parameters
        ----------
        sample : array_like
            The sample that will be tested
        mean:value: float
            The mean value that will be used to 
            compare with the sample mean

        Returns
        ----------
        t_stat: Float
            the t statistics calculated
        p_value: Float
            the p-value calculated
        degrees_freedom: Integer
            the degrees of freedom of the sample
        critical_value:
            the critical value estimated based on alpha
        """

        mean_1 = mean(sample)
        std_1 = stdev(sample)
        n1 = len(sample)

        t_stat = (mean_1 - mean_value) / (std_1/sqrt(1))
        degrees_freedom = n1 - 1
        critical_value = t.ppf(1.0 - self.alpha, degrees_freedom)

        # calculate the p-value
        p_value = (1 - t.cdf(abs(t_stat), degrees_freedom)) * 2

        return t_stat, p_value, degrees_freedom, critical_value

    def independent_t_test(self, sample_1, sample_2):

        """
        Apply independent t-test

        Parameters
        ----------
        sample_1 : array_like
            The first sample that will be tested
        sample_2 : array_like
            The second sample that will be tested

        Returns
        ----------
        t_stat: Float
            the t statistics calculated
        p_value: Float
            the p-value calculated
        degrees_freedom: Integer
            the degrees of freedom of the sample
        critical_value:
            the critical value estimated based on alpha
        """

        mean_1, mean_2 = mean(sample_1), mean(sample_2)
        std_1, std_2 = stdev(sample_1), stdev(sample_2)
        n1, n2 = len(sample_1), len(sample_2)
        std_error_dif = sqrt((std_1**2)/n1 + (std_2**2)/n2)
        t_stat = (mean_1 - mean_2) / std_error_dif

        degrees_freedom = n1 + n2 - 2
        critical_value = t.ppf(1.0 - self.alpha, degrees_freedom)

        # calculate the p-value
        p_value = (1 - t.cdf(abs(t_stat), degrees_freedom)) * 2

        return t_stat, p_value, degrees_freedom, critical_value
