from util.database_connection import Database
from lib.linear_regression.model import Model
import numpy as np
from lib.linear_regression import sql_templates


class LinearRegression:
    def __init__(self, db):
        self.db_connection = Database(db)
        self.database = db["database"]
        self.model = None

    def save_model(self):
        pass

    def load_model(self):
        pass

    def drop_model(self, model_id=None):
        pass

    def get_model_list(self):
        pass

    def get_model_description(self, model_id=None):
        pass

    def estimate(self, table, x_columns, y_column):
        self.model = Model(table, x_columns, y_column)
        self.__add_ones_column(table)
        self.__init_calculation_table("linreg_" + self.model.id + "_calculation", self.model.input_size)
        self.__init_result_table("linreg_" + self.model.id + "_result")
        self.__calculate_equations("linreg_" + self.model.id + "_calculation", "bmi_short", self.model.input_size)
        equations = self.__get_equations("linreg_" + self.model.id + "_calculation")
        xtx = equations[:, 1:self.model.input_size + 1]
        xty = equations[:, self.model.input_size + 1]
        theta = np.linalg.solve(xtx, xty)
        for x in theta:
            sql_statement = sql_templates.tmpl['save_theta'].render(table="linreg_" + self.model.id + "_result",
                                                                    value=x)
            self.db_connection.execute(sql_statement)
        return self

    def score(self):
        self.__init_score_table("linreg_" + self.model.id + "_score")
        sql_statement = sql_templates.tmpl['calculate_save_score'].render(table_id='linreg_' + self.model.id,
                                                                          input_table=self.model.input_table,
                                                                          y=self.model.y_column[0])
        self.db_connection.execute(sql_statement)

    def predict(self, table=None, x_columns=None):
        self.__init_prediction_table("linreg_" + self.model.id + "_prediction")
        if table is not None and x_columns is not None:
            self.model.prediction_table = table
            self.model.prediction_columns = x_columns

        input_table = self.model.prediction_table
        coefficients = self.get_coefficients()
        prediction_statement = str(coefficients[0][0])
        for i in range(self.model.input_size - 1):
            prediction_statement = prediction_statement + " + " + self.model.prediction_columns[i] + "*" + \
                                   str(coefficients[i + 1][0])
        sql_statement = sql_templates.tmpl['predict'].render(table="linreg_" + self.model.id + "_prediction",
                                                             input_table=input_table,
                                                             prediction_statement=prediction_statement)
        self.db_connection.execute(sql_statement)
        return self

    def predict_array(self, data):
        return self

    def get_prediction_array(self):
        sql_statement = sql_templates.tmpl['select_x_from'].render(x='y_prediction', database=self.database,
                                                                   table='linreg_' + self.model.id + '_prediction')
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def get_coefficients(self):
        sql_statement = sql_templates.tmpl['select_x_from'].render(x='theta', database=self.database,
                                                                   table='linreg_' + self.model.id + '_result')
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def get_score(self):
        sql_statement = sql_templates.tmpl['select_x_from'].render(x='score', database=self.database,
                                                                   table='linreg_' + self.model.id + '_score')
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)[0][0]

    def __get_equations(self, table):
        sql_statement = sql_templates.tmpl['get_all_from'].render(database=self.database, table=table)
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def __get_column_names(self, table):
        sql_statement = sql_templates.tmpl['table_columns'].render(database=self.database, table=table)
        data = self.db_connection.execute_query(sql_statement)
        return np.asarray(data)

    def __init_calculation_table(self, table, input_size):
        sql_statement = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute(sql_statement)
        x = []
        for i in range(input_size):
            x.append('x' + str(i))
        sql_statement = sql_templates.tmpl['init_calculation_table'].render(database=self.database, table=table,
                                                                            x_columns=x)
        self.db_connection.execute(sql_statement)

    def __init_result_table(self, table):
        sql_statement = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute(sql_statement)
        sql_statement = sql_templates.tmpl['init_result_table'].render(database=self.database, table=table)
        self.db_connection.execute(sql_statement)

    def __init_prediction_table(self, table):
        sql_statement = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute(sql_statement)
        sql_statement = sql_templates.tmpl['init_prediction_table'].render(database=self.database, table=table)
        self.db_connection.execute(sql_statement)

    def __init_score_table(self, table):
        sql_statement = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute(sql_statement)
        sql_statement = sql_templates.tmpl['init_score_table'].render(database=self.database, table=table)
        self.db_connection.execute(sql_statement)

    def __add_ones_column(self, table):
        if 'linreg_ones' not in self.__get_column_names(table):
            sql_statement = sql_templates.tmpl['add_ones_column'].render(table=table,
                                                                         column='linreg_ones')
            self.db_connection.execute(sql_statement)

    def __calculate_equations(self, table, table_input, input_size):
        columns = ['linreg_ones', 'Height_Inches', 'Weight_Pounds', 'BMI']
        for i in range(input_size):
            sum_statements = []
            for j in range(input_size + 1):
                if j < input_size:
                    sum_statements.append("sum(" + columns[i] + "*" + columns[j] + ") FROM " + table_input + "),")
                else:
                    sum_statements.append("sum(" + columns[i] + "*" + columns[j] + ") FROM " + table_input + ")")
            sql_statement = sql_templates.tmpl['calculate_equations'].render(table=table, table_input=table_input,
                                                                             sum_statements=sum_statements)
            self.db_connection.execute(sql_statement)
