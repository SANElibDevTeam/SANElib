from sqlalchemy.engine.url import URL
import pandas as pd
from sqlalchemy import create_engine
import Analysis
from sqlalchemy import text
from Utils import *
import constants as cons

class Database():
    def __init__(self, db_connection=None,dataframe = pd.DataFrame({'A' : []}), dfname = "DF_NAME"):
        if db_connection != None:
            self.engine = set_connection(db_connection)
            print("Created engine")
        elif dataframe.empty == False:
            self.import_df(dataframe,dfname)
        else:
            raise ValueError("You need to pass a db connection or a dataframe")

    def import_df(self,dataframe,name):
        self.engine = create_engine('sqlite://', echo=False)
        dataframe.to_sql(name= name,con= self.engine,if_exists="replace")
