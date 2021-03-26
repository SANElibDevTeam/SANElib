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
        query_string = '''
                        SELECT * FROM test_2.bmi_short
                        '''
        data = self.db_connection.execute_query('', query_string)
        print(np.asarray(data)[:, [1, 2, 3]])
        print(self.__get_column_names("test_2", "bmi_short"))

    def accuracy(self):
        pass

    def predict(self):
        pass

    def get_coefficients(self):
        pass

    def __get_column_names(self, database, table):
        query_string = sql_templates.tmpl['table_columns'].render(database=database, table=table)
        data = self.db_connection.execute_query('table_columns', query_string)
        return str(np.asarray(data))

    def __init_calculation_table(self):
        # CREATE
        # TABLE
        # new_table
        # SELECT
        # col, col2, col3
        # FROM
        # existing_table;
        pass

    def __init_result_table(self):
        pass

    def __add_ones_column(self):
        pass

    def __calculate_xtx(self):
        pass

    def __calculate_xty(self):
        pass


