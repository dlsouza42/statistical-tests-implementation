from t_test import TTest
import numpy as np
from statistics import mean

# Creating two random samples from normal distribution

sample_1 = np.random.normal(1, 0.2, 200)
sample_2 = np.random.normal(1.05, 0.2, 200)
mean_value = 1.5

# One sample t-test

t_stat, p_value, degrees_freedom, critical_value = TTest(
    alpha=0.05).one_sample_t_test(sample_1, mean_value)

mean_diff = mean(sample_1) - mean_value

print(
    f'The mean difference between the sample and the mean is {mean_diff} '
    f'and the p-value is {p_value}'
)

# Independent t-test

t_stat, p_value, degrees_freedom, critical_value = TTest(
    alpha=0.05).independent_t_test(sample_1, sample_2)

mean_diff = mean(sample_1) - mean(sample_2)

print(
    f'The mean difference between the samples is {mean_diff} '
    f'and the p-value is {p_value}'
)
