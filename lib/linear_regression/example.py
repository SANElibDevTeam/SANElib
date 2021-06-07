import sanelib


def run_bmi_example(table_name="example_bmi", prediction_table_name="prediction_bmi"):
    lr = sanelib.linear_regression
    lr.set_log_level("DEBUG")

    # Simple linear regression
    print("SIMPLE LINEAR REGRESSION")
    x_columns = ['Height_Inches']
    y_column = ['BMI']
    lr.estimate(table_name, x_columns, y_column)
    print("COEFFICIENTS:")
    print(lr.get_coefficients())

    # Multiple linear regression with prediction & score
    print("\nMULTIPLE LINEAR REGRESSION WITH PREDICTION & SCORE")
    x_columns = ['Height_Inches', 'Weight_Pounds']
    y_column = ['BMI']
    lr.estimate(table_name, x_columns, y_column).predict().score()
    print("COEFFICIENTS:")
    print(lr.get_coefficients())
    print("PREDICTION ARRAY:")
    print(lr.get_prediction_array())
    print("SCORE:")
    print(lr.get_score())

    # Model creation and handling
    print("\nMODEL CREATION &HANDLING")
    x_columns = ['Sex', 'Age', 'Height_Inches', 'Weight_Pounds']
    y_column = ['BMI']
    lr.create_model(table_name, x_columns, y_column, "example_model")
    print("MODEL DESCRIPTION:")
    print(lr.get_active_model_description())
    lr.estimate()
    print("COEFFICIENTS:")
    print(lr.get_coefficients())
    print("MODEL LIST:")
    print(lr.get_model_list())
    lr.drop_model("m1")

    # Model Loading & non default prediction
    print("\nMODEL LOADING & NON DEFAULT PREDICTION")
    lr.load_model("m0")
    print("MODEL DESCRIPTION:")
    print(lr.get_active_model_description())
    prediction_columns = ['Height_Inches', 'Weight_Pounds']
    lr.predict(prediction_table_name, prediction_columns)
    print("PREDICTION ARRAY:")
    print(lr.get_prediction_array())
