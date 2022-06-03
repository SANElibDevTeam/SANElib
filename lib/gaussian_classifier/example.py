import numpy as np
import sanelib
import util.timer as timer


def run_example(table_name="sample_cover_type_0_001"):
    timer.start()
    gc = sanelib.gc
    gc.set_log_level("INFO")

    # Univariate Gaussian
    print("Univariate Gaussian")
    x_columns = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology']
    y_column = ['Cover_Type']
    gc.estimate(table_name, x_columns, y_column, multivariate=False)
    gc.predict()
    gc.score()
    timer.end()

    timer.start()
    gc = sanelib.gc
    gc.set_log_level("INFO")

    # Multivariate Gaussian
    print("Multivariate Gaussian")
    x_columns = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology']
    y_column = ['Cover_Type']
    gc.estimate(table_name, x_columns, y_column)
    gc.predict()
    gc.score()
    timer.end()
    gc.clean_up()


if __name__ == '__main__':
    run_example()