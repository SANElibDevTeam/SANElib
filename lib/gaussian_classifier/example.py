import sanelib
import util.timer as timer


def run_example(table_name="covtypall", prediction_table_name="prediction_covtyp"):
    timer.start()
    gc = sanelib.gc
    gc.set_log_level("DEBUG")

    # Univariate Gaussian
    print("Univariate Gaussian")
    x_columns = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology']
    y_column = ['Cover_Type']
    gc.estimate(table_name, x_columns, y_column,multivariate=False)
    gc.predict()
    gc.score()
    print(gc.get_accuracy())
    timer.end()

    timer.start()
    gc = sanelib.gc
    gc.set_log_level("DEBUG")

    # Multivariate Gaussian
    print("Multivariate Gaussian")
    x_columns = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology']
    y_column = ['Cover_Type']
    gc.estimate(table_name, x_columns, y_column)
    gc.predict()
    gc.score()
    print(gc.get_accuracy())
    timer.end()


if __name__ == '__main__':
    run_example()
