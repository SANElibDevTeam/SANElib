from util.database_connection import Database
import numpy as np


class LinearRegression:
    def __init__(self, db):
        self.db_connection = Database(db)

    def estimate(self):
        query_string = '''
                        SELECT * FROM test_2.bmi_short
                        '''
        data = self.db_connection.execute_query('TEST', query_string)
        print(np.asarray(data)[:,[1,2,3]])

    def rank(self):
        pass

    def accuracy(self):
        pass

    def predict(self):
        pass
