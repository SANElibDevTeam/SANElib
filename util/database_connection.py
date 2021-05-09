import pandas as pd
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import text
import pyodbc
import pymssql

class Database:
    def __init__(self, db_connection=None, dataframe=pd.DataFrame({'A': []}), dfname="DF_NAME"):
        if db_connection is not None:
            #self.engine = create_engine(URL(**db_connection), pool_pre_ping=True)
            #self.engine = create_engine('sqlite:///C:\\Users\\nedeo\\dbml.db')
            self.engine = create_engine('mssql+pyodbc://desktop-g9l3lcq/dbml?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')
        elif not dataframe.empty:
            self.import_df(dataframe, dfname)
        else:
            raise ValueError("You need to pass a db connection or a dataframe")

    def import_df(self, dataframe, name):
        self.engine = create_engine('sqlite://', echo=False)
        dataframe.to_sql(name=name, con=self.engine, if_exists="replace")

    def disconnect_connection(self):
        """
        Close connection to self.engine
        """
        self.engine.close()

    def get_connection(self, engine):
        """
        :return: Connection to self.engine
        """
        return self.engine.connect()

    def execute(self, query, engine=None):
        if engine is None:
            connection = self.get_connection(self.engine)
        else:
            connection = self.get_connection(engine)
        with connection.execution_options(autocommit=True) as conn:
            conn.execute(text(query))
        connection.close()

    def execute_query(self, query, engine=None, as_df=False):
        if engine is None:
            connection = self.get_connection(self.engine)
        else:
            connection = self.get_connection(engine)
        results = connection.execute(text(query))
        keys = results.keys()
        results = results.fetchall()
        connection.close()
        if as_df:
            return pd.DataFrame(results, columns=keys)
        else:
            return results

    def materializedView(self, desc, tablename, query, engine):
        print("MaterializedView: " + desc)
        self.execute('''
            drop table if exists {}'''
                     .format(tablename), engine)
        if 'mysql' in self.engine.name:
            self.execute('''
                create table {} as '''
                         .format(tablename) + query, engine)
        elif 'mssql' in self.engine.name:
            self.execute(query, engine)