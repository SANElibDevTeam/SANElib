import sanelib
from util.database_functions.mysql import multiply_matrices
from lib.linear_regression.example import run_bmi_example
from lib.mdh.example import run
from lib.kmeans.example import run_covtype_example
from util import timer

# Run MDH example
# mdh = sanelib.mdh
# run(mdh)

# Run LinearRegression example
# run_bmi_example()

# Run KMeans example
run_covtype_example()