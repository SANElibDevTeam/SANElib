import sanelib
from lib.mdh import example as mdh_example
from util import timer

# Starting time
timer.start()

# Run library
# mdh = sanelib.mdh
# mdh_example.run(mdh)

lr = sanelib.linear_regression
# x_columns = ['Height_Inches', 'Weight_Pounds']
# y_column = ['BMI']
# lr.estimate("bmi_short", x_columns, y_column).predict().score()
# print(lr.get_coefficients())
# print(lr.get_score())

# lr.create_model("bmi_short", x_columns, y_column)
# lr.load_model("m1")
# print(lr.get_active_model_description())
# lr.drop_model('m1')
# lr.estimate().predict().score()
# print(lr.get_coefficients())
# print(lr.get_score())
# print(lr.get_model_list())

lr.create_model("test_onehotencoding", ['ohe'], ['y'])
# lr.load_model()
lr.estimate().predict()
# lr.estimate().predict("test_ohe_prediction", ["ohe"])
print(lr.get_prediction_array())
lr.drop_model()
# print(lr.get_active_model_description())


# End time
timer.end()


