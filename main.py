import sanelib
from lib.linear_regression.example import run_bmi_example
from lib.mdh.example import run
from util import timer
from util.database_functions.mysql import multiply_matrices

# Run MDH example
# run(sanelib.mdh)

# Run LinearRegression example
# run_bmi_example()

lr = sanelib.linear_regression
# lr.set_log_level("DEBUG")

timer.start()
x_columns = []
for i in range(16):
    x_columns.append("x" + str(i + 1))
y_column = ["y"]
# lr.estimate("example_bmi", ["Age", "Height_Inches"], ["BMI"])
# lr.estimate("linreg_100000x16", x_columns, y_column)
lr.estimate("linreg_100000x2", ["x1", "x2"], ["y"])
print(lr.get_coefficients())

# x_columns = ['Height_Inches', 'Weight_Pounds', 'Age']
# y_column = ['BMI']
# lr.estimate("example_bmi", x_columns, y_column)

timer.end()