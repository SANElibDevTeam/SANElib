import numpy as np
import sanelib
import util.timer as timer


def run_example(table_name="sample_cover_type_0_01", prediction_table_name="prediction_covtyp"):
    for i in range(5):
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
        print(gc.get_accuracy())
        y_pred_float = gc.get_prediction_array()
        y_float = gc.get_target_array()
        timer.end()
        y_float = y_float[y_float != np.array(None)]
        y_pred_float = y_pred_float[y_pred_float != np.array(None)]
        y = y_float.astype(int)
        y_pred = y_pred_float.astype(int)
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_{i}_u_0_01.csv", y, delimiter=",")
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_pred_{i}_u_0_01.csv", y_pred,
                   delimiter=",")

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
        print(gc.get_accuracy())
        y_pred_float = gc.get_prediction_array()
        y_float = gc.get_target_array()
        timer.end()
        gc.clean_up()
        y_float = y_float[y_float != np.array(None)]
        y_pred_float = y_pred_float[y_pred_float != np.array(None)]
        y = y_float.astype(int)
        y_pred = y_pred_float.astype(int)
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_{i}_m_0_01.csv", y, delimiter=",")
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_pred_{i}_m_0_01.csv", y_pred, delimiter=",")


if __name__ == '__main__':
    run_example()
