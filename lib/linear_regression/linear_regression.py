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

    def drop_model(self, model_id):
        pass

    def get_model_list(self):
        pass

    def estimate(self, table, x_columns, y_column):
        self.model = Model(table, x_columns, y_column)
        self.__add_ones_column(table)
        self.__init_calculation_table("linreg_"+self.model.id+"_calculation", self.model.input_size)
        self.__init_result_table("linreg_"+self.model.id+"_result")
        self.__calculate_equations("linreg_"+self.model.id+"_calculation", "bmi_short", self.model.input_size)
        equations = self.__get_equations("linreg_"+self.model.id+"_calculation")
        xtx = equations[:, 1:self.model.input_size + 1]
        xty = equations[:, self.model.input_size + 1]
        theta = np.linalg.solve(xtx, xty)
        for x in theta:
            query_string = sql_templates.tmpl['save_theta'].render(table="linreg_"+self.model.id+"_result", value=x)
            self.db_connection.execute_query_without_result(query_string)
        return self

    def score(self):
        return self

    def predict(self, table=None, x_columns=None):
        if table==None or x_columns==None:
            table = self.model.prediction_table
            self.__init_prediction_table("linreg_"+self.model.id+"_prediction")


        return self

    def predict_array(self, data):
        return self

    def get_coefficients(self):
        query_string = sql_templates.tmpl['select_x_from'].render(x='theta', database=self.database, table='linreg_'+self.model.id+'_result')
        data = self.db_connection.execute_query(query_string)
        return np.asarray(data)

    def __get_equations(self, table):
        query_string = sql_templates.tmpl['get_all_from'].render(database=self.database, table=table)
        data = self.db_connection.execute_query(query_string)
        return np.asarray(data)

    def __get_column_names(self, table):
        query_string = sql_templates.tmpl['table_columns'].render(database=self.database, table=table)
        data = self.db_connection.execute_query(query_string)
        return np.asarray(data)

    def __init_prediction_table(self, table):
        query_string = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute_query_without_result(query_string)
        query_string = sql_templates.tmpl['init_prediction_table'].render(database=self.database, table=table)
        self.db_connection.execute_query_without_result(query_string)

    def __init_calculation_table(self, table, input_size):
        query_string = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute_query_without_result(query_string)
        x = []
        for i in range(input_size):
            x.append('x' + str(i))
        query_string = sql_templates.tmpl['init_calculation_table'].render(database=self.database, table=table, x_columns=x)
        self.db_connection.execute_query_without_result(query_string)

    def __init_result_table(self, table):
        query_string = sql_templates.tmpl['drop_table'].render(table=table)
        self.db_connection.execute_query_without_result(query_string)
        query_string = sql_templates.tmpl['init_result_table'].render(database=self.database, table=table)
        self.db_connection.execute_query_without_result(query_string)

    def __add_ones_column(self, table):
        if 'linreg_'+self.model.id+'_ones' not in self.__get_column_names(table):
            query_string = sql_templates.tmpl['add_ones_column'].render(table=table, column='linreg_'+self.model.id+'_ones')
            self.db_connection.execute_query_without_result(query_string)

    def __calculate_equations(self, table, table_input, input_size):
        columns = ['linreg_'+self.model.id+'_ones', 'Height_Inches', 'Weight_Pounds', 'BMI']

        for i in range(input_size):
            sum_statements = []
            for j in range(input_size+1):
                if j < input_size:
                    sum_statements.append("sum(" + columns[i] + "*" + columns[j] + ") FROM " + table_input + "),")
                else:
                    sum_statements.append("sum(" + columns[i] + "*" + columns[j] + ") FROM " + table_input+")")

            query_string = sql_templates.tmpl['calculate_equations'].render(table=table, table_input=table_input, sum_statements=sum_statements)
            self.db_connection.execute_query_without_result(query_string)
