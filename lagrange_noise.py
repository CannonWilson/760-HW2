import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy.random import normal
import numpy as np

"""
Repeat the experiment with zero-mean Gaussian noise ε added to x. Vary the
standard deviation for ε and report your findings.
"""
MIN_ST_DEV = 0.05
MAX_ST_DEV = 0.25
NUM_EXPERIMENTS = 5
st_devs = np.linspace(MIN_ST_DEV, MAX_ST_DEV, NUM_EXPERIMENTS)
INTERVAL_LOW = 0
INTERVAL_HIGH = 3.14
N = 100
losses = [{"train": 0, "test": 0} for _ in st_devs]

for idx, sigma in enumerate(st_devs):
    
    x = np.random.uniform(INTERVAL_LOW, INTERVAL_HIGH, N)
    y = np.sin(x)
    x = x + normal(loc=0, scale=sigma, size=N) # update x to have noise
    sorted_idxs = np.argsort(x)
    sorted_x = x[sorted_idxs]
    sorted_y = y[sorted_idxs]

    num_points_for_interpolation = 17
    indices = np.floor(np.linspace(0, N-1, num_points_for_interpolation)).astype(int)
    lagrange_model = lagrange(sorted_x[indices], sorted_y[indices])

    train_pred_y = Polynomial(lagrange_model.coef[::-1])(x)
    train_loss = mean_squared_error(y, train_pred_y)
    test_x = np.random.uniform(INTERVAL_LOW, INTERVAL_HIGH, N)
    test_y = np.sin(test_x)
    test_pred_y = Polynomial(lagrange_model.coef[::-1])(test_x)
    test_loss = mean_squared_error(test_y, test_pred_y)
    losses[idx]["train"] = train_loss
    losses[idx]["test"] = test_loss

"""
Plot the results
"""
plt.scatter(x, y, label="points")
plt.plot(sorted_x[indices], Polynomial(lagrange_model.coef[::-1])(sorted_x[indices]), label="interpolation")
plt.legend()
plt.show()

"""

labels = st_devs
train_losses = [loss_dic["train"] for loss_dic in losses]
test_losses = [loss_dic["test"] for loss_dic in losses]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_loss, width, label='train')
rects2 = ax.bar(x + width/2, test_losses, width, label='test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss (MSE)')
ax.set_xlabel('Standard deviation of noise Gaussian')
ax.set_title('Loss for Lagrange Interpolation with Varying Noise')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
"""