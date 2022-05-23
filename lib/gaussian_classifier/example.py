import numpy as np
import sanelib
import util.timer as timer




def run_example_covertype(table_name="sample_cover_type_0_01", prediction_table_name="prediction_covtyp"):
    for i in range(5):
        timer.start()
        gc = sanelib.gc
        gc.set_log_level("NONE")

        # Univariate Gaussian
        print("Univariate Gaussian")
        x_columns = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology']
        y_column = ['Cover_Type']
        gc.estimate(table_name, x_columns, y_column, multivariate=False)
        gc.predict()
        gc.score()
        print(gc.get_accuracy())
        timer.end()
        y_pred_float = gc.get_prediction_array()
        y_float = gc.get_target_array()
        print(f"This was round {i}")
        gc.clean_up(multivariate=False)
        y_float = y_float[y_float != np.array(None)]
        y_pred_float = y_pred_float[y_pred_float != np.array(None)]
        y = y_float.astype(int)
        y_pred = y_pred_float.astype(int)
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_{i}_uni_0_0_1.csv", y, delimiter=",")
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_{i}_pred_uni_0_0_1.csv", y_pred,
                   delimiter=",")

        timer.start()
        gc = sanelib.gc
        gc.set_log_level("NONE")

        # Multivariate Gaussian
        print("Multivariate Gaussian")
        x_columns = ['Elevation', 'Horizontal_Distance_To_Fire_Points', 'Horizontal_Distance_To_Hydrology']
        y_column = ['Cover_Type']
        gc.estimate(table_name, x_columns, y_column)
        gc.predict()
        gc.score()
        print(gc.get_accuracy())
        timer.end()
        y_pred_float = gc.get_prediction_array()
        y_float = gc.get_target_array()
        print(f"This was round {i}")
        gc.clean_up()
        y_float = y_float[y_float != np.array(None)]
        y_pred_float = y_pred_float[y_pred_float != np.array(None)]
        y = y_float.astype(int)
        y_pred = y_pred_float.astype(int)
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_{i}_multi_0_0_1.csv", y, delimiter=",")
        np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_{i}_pred_multi_0_0_1.csv", y_pred, delimiter=",")
def run_example_customers(table_name="customers_classification_cleaned"):
    timer.start()
    gc = sanelib.gc
    gc.set_log_level("NONE")

    # Univariate Gaussian
    print("Univariate Gaussian")
    x_columns = ['Age','Job', 'Duration','Sex_cat','Housing_cat','Saving_accounts_cat','Purpose_cat','Checking_account_cat']
    y_column = ['ACB_class_cat']
    gc.estimate(table_name, x_columns, y_column, multivariate=False)
    gc.predict()
    gc.score()
    print(gc.get_accuracy())
    y_pred = gc.get_prediction_array()
    y = gc.get_target_array()
    timer.end()
    np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_customers_u.csv", y,
               delimiter=",")
    np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_pred_customers_u.csv", y_pred,
               delimiter=",")

    timer.start()
    gc = sanelib.gc
    gc.set_log_level("NONE")

    # Multivariate Gaussian
    print("Multivariate Gaussian")
    x_columns = ['Age','Job', 'Duration','Sex_cat','Housing_cat','Saving_accounts_cat','Purpose_cat','Checking_account_cat']
    y_column = ['ACB_class_cat']
    gc.estimate(table_name, x_columns, y_column, multivariate=True)
    gc.predict()
    gc.score()
    print(gc.get_accuracy())
    y_pred = gc.get_prediction_array()
    y = gc.get_target_array()
    timer.end()
    np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_customers_m.csv", y,
               delimiter=",")
    np.savetxt(f"C:/Users/annam/OneDrive/Documents/HSLU/MSC/FS22/Masterthesis/datasets/y_pred_customers_m.csv", y_pred,
               delimiter=",")

    gc.clean_up()

if __name__ == '__main__':
    run_example_covertype()
    # run_example_customers()
