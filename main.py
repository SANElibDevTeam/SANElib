import sanelib
from lib.mdh import example as mdh_example
from lib.linear_regression import example as lr_example
from util import timer

# Starting time
# timer.start()

# Run MDH example
# mdh = sanelib.mdh
# mdh_example.run(mdh)

# Run LinearRegression example
# lr_example.run_bmi_example()

lr = sanelib.linear_regression
# lr.set_log_level("INFO")
x_columns = ['Height_Inches', 'Weight_Pounds']
y_column = ['BMI']
# lr.estimate("bmi_short", x_columns, y_column)
# print(lr.get_coefficients())
lr.estimate("bmi_short", x_columns, y_column).predict().score()
print(lr.get_coefficients())
print(lr.get_score())


# End time
# timer.end()


