import sanelib
from lib.dtc.example import run_covertype_example as dtc_covertype_example
from lib.dtc.example import run_iris_example as dtc_iris_example
from lib.kmeans.example import run_iris_example as kmeans_iris_example
from lib.linear_regression.example import run_bmi_example
from lib.mdh.example import run
from util import timer
from util.database_functions.mysql import multiply_matrices

# Run MDH example
# mdh = sanelib.mdh
# run(mdh)


# Run LinearRegression example
# run_bmi_example()

lr = sanelib.linear_regression
lr.set_log_level("DEBUG")
# x_columns = []
# for i in range(128):
#     x_columns.append("x" + str(i + 1))
# y_column = ["y"]
timer.start()
lr.estimate("example_bmi", ["Age", "Height_Inches"], ["BMI"])
timer.end()

# Run KMeans example
# kmeans_iris_example()

# Run DecisionTreeClassifier example
# dtc_iris_example()
# dtc_covertype_example()
