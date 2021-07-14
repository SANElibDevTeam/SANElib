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
x_columns = []
for i in range(8):
    x_columns.append("x" + str(i + 1))
y_column = ["y"]
timer.start()
# lr.estimate("example_bmi", ["Age", "Height_Inches"], ["BMI"])
# lr.estimate2("linreg_100000x8", x_columns, y_column)
# print(lr.get_coefficients())
# lr.estimate("linreg_1000000x2", ["x1", "x2"], ["y"])


x_columns = ['Height_Inches']
y_column = ['BMI']
lr.estimate("example_bmi", x_columns, y_column)
print(lr.get_coefficients())

timer.end()