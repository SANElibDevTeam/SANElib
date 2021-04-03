import logging

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL


class Database:
    def __init__(self, db_connection=None, dataframe=pd.DataFrame({'A': []}), dfname="DF_NAME"):
        if db_connection is not None:
            self.engine = create_engine(URL(**db_connection), pool_pre_ping=True)
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

    def execute_query_without_result(self, query, engine=None):
        if engine is None:
            connection = self.get_connection(self.engine)
        else:
            connection = self.get_connection(engine)
        connection.execute(text(query))
        connection.close()

    def execute_query(self, query, engine=None):
        if engine is None:
            connection = self.get_connection(self.engine)
        else:
            connection = self.get_connection(engine)
        results = connection.execute(text(query))
        results = results.fetchall()
        connection.close()

        return results

    def create_materialized_view(self, tablename, query, engine=None):
        self.execute_query_without_result(f"drop table if exists {tablename}", engine)
        self.execute_query_without_result(f"create table {tablename} as {query}", engine)
