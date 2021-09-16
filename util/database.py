import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL


class Database:
    def __init__(self, db_connection=None, dataframe=pd.DataFrame({"A": []}), dfname="DF_NAME"):
        self.driver_name = db_connection["drivername"]
        self.database_name = db_connection["database"]
        self.connection = None

        if db_connection is not None:
            if db_connection["drivername"] == "mysql+mysqlconnector":
                self.engine = create_engine(URL(**db_connection), pool_pre_ping=True)
            elif db_connection['drivername'] == 'mssql+pyodbc':
                self.engine = create_engine(
                    'mssql+pyodbc://' + db_connection['host'] + '/' + db_connection['database'] +
                    '?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')
            elif db_connection["drivername"] == "sqlite":
                self.engine = create_engine("sqlite:///" + db_connection["path"], pool_pre_ping=True)
                if not dataframe.empty:
                    self.import_df(dataframe, dfname)
        else:
            raise ValueError("You need to pass a db connection or a dataframe")

    def import_df(self, dataframe, name):
        dataframe.to_sql(name=name, con=self.engine, if_exists="replace", index=False)

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

    def execute(self, statement):
        self.connect()
        if 'mssql' in self.engine.name:
            with self.connection.execution_options(autocommit=True) as conn:
                conn.execute(text(statement))
        else:
            self.connection.execute(text(statement))
        self.disconnect()

    def execute_query(self, statement, as_df=False):
        self.connect()
        result = self.connection.execute(text(statement))
        keys = result.keys()
        result = result.fetchall()
        self.disconnect()
        if as_df:
            return pd.DataFrame(result, columns=keys)
        else:
            return result

    def materializedView(self, desc, tablename, query):
        print("MaterializedView: " + desc)
        self.execute('''drop table if exists {}'''.format(tablename))
        if 'mssql' in self.engine.name:
            self.execute(query)
        else:
            self.execute('''create table {} as '''.format(tablename) + query)
