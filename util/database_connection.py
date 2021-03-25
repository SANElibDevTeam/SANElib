import pandas as pd
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import text


class Database():
    def __init__(self, db_connection=None,dataframe = pd.DataFrame({'A' : []}), dfname = "DF_NAME"):
        if db_connection != None:
            self.engine = self.set_connection(db_connection)
            print("Created engine")
        elif dataframe.empty == False:
            self.import_df(dataframe,dfname)
        else:
            raise ValueError("You need to pass a db connection or a dataframe")

    def import_df(self,dataframe,name):
        self.engine = create_engine('sqlite://', echo=False)
        dataframe.to_sql(name= name,con= self.engine,if_exists="replace")

    def set_connection(self, db):
        """
        :param db: dict with connection information for DB
        :return: SQLAlchemy Engine
        """
        return create_engine(URL(**db), pool_pre_ping=True)

    def disconnect_connection(self):
        """
        Close connection to self.engine
        """
        self.engine.close()

    def get_connection(self, engine):
        """
        :return: Connection to self.engine
        """
        return engine.connect()

    def execute(self, desc, query, engine):
        connection = self.get_connection(engine)
        print(desc + '\nQuery: ' + query)
        connection.execute(text(query))
        connection.close()
        print('OK: ' + desc)
        print()

    def executeQuery(self, desc, query, engine):
        connection = self.get_connection(engine)
        print('Query: ' + query)
        results = connection.execute(text(query))
        results = results.fetchall()
        connection.close()
        print('OK: ' + desc)
        print()

        return results

    def materializedView(self, desc, tablename, query, engine):
        self.execute('Dropping table ' + tablename, '''
            drop table if exists {}'''
                .format(tablename), engine)
        self.execute(desc, '''
            create table {} as '''
                .format(tablename) + query, engine)

