import sanelib
from util.database_functions.mysql import multiply_matrices
from lib.linear_regression.example import run_bmi_example
from lib.mdh.example import run
from lib.kmeans.example import run_covtype_example
from lib.dtc.DecisionTreeClassifier import DecisionTreeClassifier
from util import timer

# Run MDH example
# mdh = sanelib.mdh
# run(mdh)

# Run LinearRegression example
# run_bmi_example()

# Run KMeans example
# run_covtype_example()

# Run DecisionTreeClassifier example
# dtc = DecisionTreeClassifier(db=sanelib.db_connection, dataset='covtype', max_samples=2)
# dtc.train_test_split(ratio=0.02, seed=1, encode=False)
# dtc.estimate()