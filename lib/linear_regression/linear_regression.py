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

    def estimate(self, table, x_columns, y_column):
        self.model = Model(table, x_columns, y_column)
        # self.__add_ones_column(self.database, table)
        # self.__init_calculation_table(self.database, "linreg_"+self.model.id+"_calculation", self.model.input_size)
        # self.__init_result_table(self.database, "linreg_"+self.model.id+"_result")
        # self.__calculate_equations("linreg_"+self.model.id+"_calculation", "bmi_short", self.model.input_size)
        equations = self.__get_equations(self.database, "linreg_"+self.model.id+"_calculation")
        xtx = equations[:, 1:self.model.input_size + 1]
        xty = equations[:, self.model.input_size + 1]
        theta = np.linalg.solve(xtx, xty)
        print(theta)

    def score(self):
        pass

    def predict(self, table=None, x_columns=None):
        pass

    def predict_array(self, data=None):
        pass

    def get_coefficients(self, database, table, input_size):
        pass
    # TODO query result table

    def __drop_model_0(self):
        pass

    def __get_available_model_ids(self):
        pass

    def __get_equations(self, database, table):
        query_string = sql_templates.tmpl['get_all_from'].render(database=database, table=table)
        data = self.db_connection.execute_query(query_string)
        return np.asarray(data)

    def __get_column_names(self, database, table):
        query_string = sql_templates.tmpl['table_columns'].render(database=database, table=table)
        data = self.db_connection.execute_query(query_string)
        return np.asarray(data)

    def __init_calculation_table(self, database, table, input_size):
        x = []
        for i in range(input_size):
            x.append('x' + str(i))
        query_string = sql_templates.tmpl['init_calculation_table'].render(database=database, table=table, x_columns=x)
        self.db_connection.execute_query_without_result(query_string)

    def __init_result_table(self, database, table):
        query_string = sql_templates.tmpl['init_result_table'].render(database=database, table=table)
        self.db_connection.execute_query_without_result(query_string)

    def __add_ones_column(self, database, table):
        if 'linreg_'+self.model.id+'_ones' not in self.__get_column_names(database, table):
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
