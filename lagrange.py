import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

"""
Repeat the experiment with zero-mean Gaussian 
noise ε added to y. Vary the standard deviation 
for ε and report your findings.
"""
INTERVAL_LOW = 0
INTERVAL_HIGH = 3.14
N = 100
x = np.random.uniform(INTERVAL_LOW, INTERVAL_HIGH, N)
y = np.sin(x)
sorted_idxs = np.argsort(x)
sorted_x = x[sorted_idxs]
sorted_y = y[sorted_idxs]

"""
Build a model f using Lagrange Interpolation
"""
num_points_for_interpolation = 17
indices = np.floor(np.linspace(0, N-1, num_points_for_interpolation)).astype(int)
lagrange_model = lagrange(sorted_x[indices], sorted_y[indices])
print("Found coefficients: ")
print(Polynomial(lagrange_model.coef[::-1]).coef)

"""
Plot the results so far
"""
plt.scatter(x, y, label="points")
plt.plot(sorted_x[indices], Polynomial(lagrange_model.coef[::-1])(sorted_x[indices]), label="interpolation")
plt.legend()
plt.show()

"""
Generate a test set using the same distribution as your 
train set. Compute and report the resulting model's train and
test error. What do you observe? 
"""
train_pred_y = Polynomial(lagrange_model.coef[::-1])(x)
train_loss = mean_squared_error(y, train_pred_y)
test_x = np.random.uniform(INTERVAL_LOW, INTERVAL_HIGH, N)
test_y = np.sin(test_x)
test_pred_y = Polynomial(lagrange_model.coef[::-1])(test_x)
test_loss = mean_squared_error(test_y, test_pred_y)
print(f"Training loss: {train_loss}, \n \
    Testing loss: {test_loss}")

# I used MSE to calculate the losses.
# Both the training loss and the testing loss are incredibly
# small. The training loss ranged from 
# 1.8118153970989106e-10  -  3.5180321022014014e-06
# while the testing loss ranged from
# 2.1188568447939583e-10  -  4.038013723745428e-06
# over 5 trials. Sometimes the testing loss 
# was below the training loss which was weird.