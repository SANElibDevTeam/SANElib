import sanelib
from lib.mdh import example as mdh_example
from util import timer

# Starting time
timer.start()

# Run library
# mdh = sanelib.mdh
# mdh_example.run(mdh)
lr = sanelib.linear_regression
x_columns = ['Height_Inches', 'Weight_Pounds']
y_column = ['BMI']
# lr.estimate("bmi_short", x_columns, y_column).predict().score()
# lr.create_model("test_onehotencoding", ['ohe'], ['y'], "test")
# lr.estimate()
# print(lr.get_coefficients())
lr.load_model("m0")
lr.estimate()
print(lr.get_coefficients())
# print(lr.get_active_model_description())
# lr.drop_model()

# End time
timer.end()


