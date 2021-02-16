from sqlalchemy.engine.url import URL
import pandas as pd
from sqlalchemy import create_engine
import Analysis
from sqlalchemy import text

class Database():
    def __init__(self, db_connection=None,dataframe = None, dfname = None):
        if db_connection != None:
            self.engine = self.set_connection(db_connection)
            print("Created engine")
        elif dataframe != None:
            self.import_df(dataframe,dfname)
        else:
            print("You need to pass a db connection or a dataframe")

    def import_df(self,dataframe,name):
        dataframe.to_sql(name= name,con= self.engine, if_exists='replace')


    def set_connection(self, db):
        """
        :param db: dict with connection information for DB
        :return: SQLAlchemy Engine
        """
        return create_engine(URL(**db))

    def get_connection(self):
        """
        :return: Connection to self.engine
        """
        return self.engine.connect()

    def disconnect_connection(self):
        """
        Close connection to self.engine
        """
        self.engine.close()

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

    def analyze(self,dataset,seed=None,model_id='table_name',ratio=1.0):
        return Analysis.Analysis(dataset,seed,model_id,ratio)

    def test(self):
        print("hello")



