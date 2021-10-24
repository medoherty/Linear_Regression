# Import the packages and dataset to be used in the program.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

# The dataset is loaded and the dimension of the data is increased by 1 by using numpy.newaxis, so that the array is a
# 2D array (for an x and a y axis).

d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]

# The data is split into training and testing sets, whereby the last 20 samples are used as testing data and the rest as
# training data for both the x (explanatory variables) and y (dependent variables) values.

dx_train = d_X[:-20]
dy_train = d.target[:-20]
dx_test = d_X[-20:]
dy_test = d.target[-20:]


# The Ordinary Least Squares method is used in an algorithm, whereby the gradient and the y-intercept of the line of
# best fit is calculated and the results are saved in the variables m (gradient) and b (y-intercept of the line of best
# fit) and returned. The extra dimension of 1 that was added to make the data 2-dimensional is removed by applying the
# squeeze() function to the x array when it is passed as an argument to the gradient_and_y_intercept method.

def gradient_and_y_intercept():
    m = (np.mean(dx_train.squeeze()) * np.mean(dy_train) - np.mean(dx_train.squeeze() * dy_train)) \
        / ((np.mean(dx_train).squeeze()) ** 2 - np.mean(dx_train.squeeze() ** 2))

    b = np.mean(dy_train) - m * np.mean(dx_train)

    return m, b


# The above method is called and the results stored in the m and b variables.

m, b = gradient_and_y_intercept()

# The Mean Square Error is calculated by calculating the error in the test set and taking the mean of those values
# (the average squared difference between the estimated values and the actual value).

mse = np.mean((((m * dx_test) + b) - dy_test) ** 2)

# The score of the variance (the dispersion of errors in the dataset) is calculated by taking 1 minus the residual sum
# of squares divided by the total sum of squares.

score = (1 - ((dy_test - ((m * dx_test) + b)) ** 2).sum() / ((dy_test - dy_test.mean()) ** 2).sum())

# The coefficient (gradient of the line of best fit) calculated previously, which is a float, is converted to a string
# and printed to the screen.

mString = str(m)
print("Coefficient: " + mString)

# The Mean Square Error (the average square of the errors) calculated previously, which is a float, is converted to a
# string and printed to the screen.

mseString = str(mse)
print("Mean Squared Error: " + mseString)

# The score of the variance (the dispersion of errors in the dataset) calculated previously, which is a float, is
# converted to a string and printed to the screen.

scoreString = str(score)
print("Variance score: " + scoreString)

# A figure is created as well as sub plots in 3 rows in 1 column in the figure. The first sub plot is a scatter plot of
# the training data in red with the label "Training Data". The second sub plot is a scatter plot of the test data in
# green with the label "Testing Data". The last sub plot is of the line of best fit in blue. The legends are then
# appended to each sub plot. A window is then opened and the figure is displayed.

fig = plt.figure()

ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

ax1.scatter(dx_train, dy_train, c='r', label="Training Data")
ax2.scatter(dx_test, dy_test, c='g', label="Testing Data")
ax3.plot(dx_test, ((m * dx_test) + b), c='b', label="Line of Best Fit")

legend1 = ax1.legend()
legend2 = ax2.legend()
legend3 = ax3.legend()

plt.show()
