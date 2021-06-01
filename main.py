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

# Run KMeans example
# kmeans_iris_example()

# Run DecisionTreeClassifier example
# dtc_iris_example()
# dtc_covertype_example()
# timer.start()
# dtc = sanelib.dtc
# dtc.initialize(target='Cover_Type', dataset='covertype')
# dtc.train_test_split(ratio=0.5, encode=True)
# dtc.estimate()
# # dtc.read_dtc_tree("C:/Users/nedeo/DTCcomplete/mssql_tree_5050split.pk1")
# dtc.save_dtc_tree("C:/Users/nedeo/DTCcomplete/mysql_tree_5050split.pk1")
# pred = dtc.predict()
# acc = len(pred.query("prediction == Cover_Type")) / len(pred)
# print("Accuracy is {}".format(acc))
# timer.end()