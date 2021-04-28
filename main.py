import sanelib
from util.database_functions.mysql import multiply_matrices

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


