import sanelib
from lib.mdh import example as mdh_example
from lib.linear_regression import example as lr_example
from util import timer
from util.database_functions import multiply_matrices

# Starting time
# timer.start()

# Run MDH example
# mdh = sanelib.mdh
# mdh_example.run(mdh)

# Run LinearRegression example
# lr_example.run_bmi_example()
database = sanelib.db
multiply_matrices(database, "ma", "mb", "mc")


# End time
# timer.end()


