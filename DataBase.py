from sqlalchemy.engine.url import *
from sqlalchemy import *
from sqlalchemy.engine.reflection import *
import time

class DataBase:

    def __init__(self, db):
        print('-- connecting ...' )
        self.engine = create_engine(URL(**db))
        self.metadata = MetaData(self.engine)
        self.metadata.reflect()
        #self.metadata.reflect(views=True)
        self.inspector = Inspector.from_engine(self.engine)
        print('-- connected')

    def get_connection(self):
        """
        :return: Connection to self.engine
        """
        return self.engine.connect()

    def get_all_col_types(self, table_name):
        """
        get column names for table, either all or just numeric or categoric
        :return: list of two lists, list[0] column names, list[1] column types
        """
        list = []
        list.append([])
        list.append([])

        #table = self.metadata.tables[table_name]
        table = Table(table_name, self.metadata, autoload=True)

        for c in table.c:
            list[0].append(c.key)
            list[1].append(c.type)
        return list

    def getLastCol(self, table_name):
        """
        get last column name for table
        :return: name of last column
        """
        table = Table(table_name, self.metadata, autoload=True)
        for c in table.c:
            if str(c.type) in ['DOUBLE', 'INTEGER']:
                list += [c.key]

        return list[len(list)-1]

    def get_num_cols(self, table_name):
        """
        get column names of numeric type for table <table_name>
        :return: list of column names
        """
       # table = self.metadata.tables[table_name]
        table = Table(table_name, self.metadata, autoload=True)
        # https://docs.sqlalchemy.org/en/14/core/metadata.html
        list = []
        for c in table.c:
            if str(c.type) in ['DOUBLE', 'INTEGER']:
                list += [c.key]
        return list

    def get_cat_cols(self, table_name):
        """
        get column names of numeric type for table <table_name>
        :return: list of column names
        """
        #table = self.metadata.tables[table_name]
        table = Table(table_name, self.metadata, autoload=True)
        # https://docs.sqlalchemy.org/en/14/core/metadata.html
        list = []
        for c in table.c:
            if str(c.type) in ('TEXT'):
                list += [c.key]
        return list

    def does_table_exist(self, table_name):
        """
        Check if table <table_name> exists in database
        :return: true if table exists, else false
        """
        return table_name in self.inspector.get_view_names() | table_name in self.inspector.get_table_names()

    def execute(self, query, desc=''):
        start = time.time()
        connection = self.get_connection()
        print('-- ' + desc + '\n' + query)
        connection.execute(text(query))
        connection.close()
        print(f"-- Runtime of query was {time.time() - start} seconds")
        print(";\n")

    def executeQuery(self, query, desc=''):
        start = time.time()
        connection = self.get_connection()
        print('-- ' + desc + '\n' + query)
        results = connection.execute(text(query))
        results = results.fetchall()
        connection.close()
        print(f"-- Runtime of query was {time.time() - start} seconds")
        print(";\n")
        return results

    def createView(self, viewName, query, materialized=True, desc=''):
        self.execute('''drop table if exists {}'''
                     .format(viewName), 'Dropping table ' + viewName)
        self.execute('''drop view if exists {}'''
                     .format(viewName), 'Dropping view ' + viewName)
        if materialized:
            self.execute('''create table {} as \n'''
                         .format(viewName) + query, desc)
        else:  # dynamic view
            self.execute('''create or replace view {} as \n'''
                         .format(viewName) + query, desc)

