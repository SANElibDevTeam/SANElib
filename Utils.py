from sqlalchemy import text
import pandas as pd
class Utils():
    def __init__(self):
        pass
    def execute(self, desc, query):
        connection = self.get_connection()
        print(desc + '\nQuery: ' + query)
        connection.execute(text(query))
        connection.close()
        print('OK: ' + desc)
        print()

    def executeQuery(self, desc, query):
        connection = self.get_connection()
        print('Query: ' + query)
        results = connection.execute(text(query))
        results = results.fetchall()
        connection.close()
        print('OK: ' + desc)
        print()

        return results

    def materializedView(self, desc, tablename, query):
        self.execute('Dropping table ' + tablename, '''
            drop table if exists {}'''
                .format(tablename))
        self.execute(desc, '''
            create table {} as '''
                .format(tablename) + query)

