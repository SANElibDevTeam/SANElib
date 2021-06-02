from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import text

class SaneDataBase:

    def __init__(self, db):
        self.engine = create_engine(URL(**db))

    def get_connection(self):
        """
        :return: Connection to self.engine
        """
        return self.engine.connect()


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


    def createView(self, desc, viewName, query, materialized=True):
        self.execute('Dropping table ' + viewName, '''
            drop table if exists {}'''
                     .format(viewName))
        self.execute('Dropping view ' + viewName, '''
                    drop view if exists {}'''
                     .format(viewName))
        if materialized:
            self.execute(desc, '''
                create table {} as '''
                         .format(viewName) + query)
        else:  # dynamic view
            self.execute(desc, '''
                            create or replace view {} as '''
                         .format(viewName) + query)