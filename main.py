import sanelib
from util.database_functions.mysql import multiply_matrices
from lib.linear_regression.example import run_bmi_example
from lib.mdh.example import run
from lib.kmeans.example import run_covtype_example
from lib.dtc.DecisionTreeClassifier import DecisionTreeClassifier
from lib.dtc.example import run_iris_example as dtc_iris_example
from lib.dtc.example import run_covertype_example as dtc_covertype_example
from util import timer

# Run MDH example
# mdh = sanelib.mdh
# run(mdh)

# Run LinearRegression example
# run_bmi_example()

# Run KMeans example
# run_covtype_example()

# Run DecisionTreeClassifier example
# dtc_iris()
#dtc_covertype_example()
dtc = sanelib.dtc.DecisionTreeClassifier(db=sanelib.db_connection, dataset="covertype")
dtc.train_test_split(encode=True)