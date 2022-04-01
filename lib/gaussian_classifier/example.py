import sanelib
import util.timer as timer

def run_example(table_name="covtyp", prediction_table_name="prediction_covtyp"):
    timer.start()
    gc = sanelib.gc
    gc.set_log_level("DEBUG")

    # Simple linear regression
    print("SIMPLE Gaussian")
    x_columns = ['Elevation','Horizontal_Distance_To_Fire_Points']
    y_column = ['Cover_Type']
    gc.estimate(table_name, x_columns, y_column)
    timer.end()
if __name__ == '__main__':
    run_example()