import sanelib
from lib.mdh import example as mdh_example
from util import timer

# Starting time
timer.start()

# Run library
# mdh = sanelib.mdh
# mdh_example.run(mdh)

# x_columns = ['Height_Inches', 'Weight_Pounds']
# y_column = ['BMI']

lr = sanelib.linear_regression
lr.create_model("test_onehotencoding", ['ohe'], ['y'])
lr.estimate().predict("test_ohe_prediction", ["ohe"]).score()
print(lr.get_prediction_array())
lr.drop_model()

# End time
timer.end()


