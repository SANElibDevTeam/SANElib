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
lr.drop_model()
print(lr.get_model_list())

# End time
timer.end()


