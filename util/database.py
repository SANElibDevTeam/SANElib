import pandas as pd
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import text


class Database:
    def __init__(self, db_connection=None, dataframe=pd.DataFrame({'A': []}), dfname="DF_NAME"):
        self.database_name = db_connection["database"]
        self.connection = None

        if db_connection is not None:
            self.engine = create_engine(URL(**db_connection), pool_pre_ping=True)
        elif not dataframe.empty:
            self.import_df(dataframe, dfname)
        else:
            raise ValueError("You need to pass a db connection or a dataframe")

    def import_df(self, dataframe, name):
        self.engine = create_engine('sqlite://', echo=False)
        dataframe.to_sql(name=name, con=self.engine, if_exists="replace")

    def disconnect(self):
        """
        Close connection to self.engine
        """
        self.connection.close()

    def connect(self):
        """
        :return: Connection to self.engine
        """
        self.connection = self.engine.connect()

    def execute(self, query, engine=None):
        self.connect()
        self.connection.execute(text(query))
        self.disconnect()

    def execute_query(self, query, engine=None):
        self.connect()
        result = self.connection.execute(text(query))
        result = result.fetchall()
        self.disconnect()
        return result
