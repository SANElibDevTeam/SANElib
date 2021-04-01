from util.database_connection import Database
import numpy as np
from lib.linear_regression import sql_templates


class LinearRegression:
    def __init__(self, db):
        self.db_connection = Database(db)
        self.models = None
        self.active_model = None
        self.x_columns = None
        self.y_column = None

    def estimate(self):
        # self.__add_ones_column("test_2", "bmi_short")
        # self.__init_calculation_table("test_2", "linreg_m1_calculation", 3)
        # self.__init_result_table("test_2", "linreg_m1_result")
        # self.__calculate_equations("linreg_m1_calculation", "bmi_short", 3)
        print(self.get_coefficients())


    def accuracy(self):
        pass

    def predict(self):
        pass

    def get_coefficients(self):
        equations = self.__get_equations("test_2", "linreg_m1_calculation")
        xtx = equations[:, 1:4]
        xty = equations[:, 4]
        theta = np.linalg.solve(xtx, xty)
        return theta

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
        for i in range(input_size + 1):
            x.append('x' + str(i))
        query_string = sql_templates.tmpl['init_calculation_table'].render(database=database, table=table, x_columns=x)
        self.db_connection.execute_query_without_result(query_string)

    def __init_result_table(self, database, table):
        query_string = sql_templates.tmpl['init_result_table'].render(database=database, table=table)
        self.db_connection.execute_query_without_result(query_string)

    def __add_ones_column(self, database, table):
        if 'linreg_m1_ones' not in self.__get_column_names(database, table):
            query_string = sql_templates.tmpl['add_ones_column'].render(table=table, column='linreg_m1_ones')
            self.db_connection.execute_query_without_result(query_string)

    def __calculate_equations(self, table, table_input, input_size):
        columns = ['linreg_m1_ones', 'Height_Inches', 'Weight_Pounds', 'BMI']

        for i in range(input_size):
            sum_statements = []
            for j in range(input_size+1):
                if j < input_size:
                    sum_statements.append("sum(" + columns[i] + "*" + columns[j] + ") FROM " + table_input + "),")
                else:
                    sum_statements.append("sum(" + columns[i] + "*" + columns[j] + ") FROM " + table_input+")")

            query_string = sql_templates.tmpl['calculate_equations'].render(table=table, table_input=table_input, sum_statements=sum_statements)
            self.db_connection.execute_query_without_result(query_string)



