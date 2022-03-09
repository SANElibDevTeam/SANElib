import sanelib


def run_example(table_name="covtyp", prediction_table_name="prediction_covtyp"):
    gc = sanelib.gc
    gc.set_log_level("DEBUG")

    # Simple linear regression
    print("SIMPLE Gaussian")
    x_columns = ['Elevation']
    y_column = ['Cover_Type']
    gc.estimate(table_name, x_columns, y_column)

if __name__ == '__main__':
    run_example()