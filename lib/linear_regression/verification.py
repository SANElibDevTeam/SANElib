import numpy as np
from sklearn.linear_model import LinearRegression
import time
from util import timer
import sanelib

# Set the number of columns to process
number_of_columns = 2

# Used for sklearn verification
test_data_path = "/X/linreg_X.csv"

# Used for sanelib verification
table_name = "linreg_X"


def run_sanelib_verification():
    timer.start()

    X = []
    for i in range(number_of_columns):
        X.append('x' + str(i + 1))
    y = ['y']

    lr = sanelib.linear_regression
    lr.estimate(table_name, X, y)

    timer.end()


def run_sklearn_verification():
    start_time = time.time()
    data_import = np.genfromtxt(test_data_path, delimiter=',', dtype=None, encoding=None)

    x_range = []
    for i in range(number_of_columns):
        x_range.append(i)

    X = data_import[1:, x_range].astype(float)
    y = data_import[1:, [number_of_columns]].astype(float)

    reg = LinearRegression().fit(X, y)

    print(f"Runtime: {time.time() - start_time} [s]")


if __name__ == "__main__":
    run_sanelib_verification()
    run_sklearn_verification()
