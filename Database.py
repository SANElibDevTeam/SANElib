from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import text

class SaneDataBase:

    def __init__(self, db):
        self.engine = create_engine(URL(**db))
        self.metadata = MetaData(bind=self.engine)
        self.metadata.reflect()

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
        table = self.metadata.tables[table_name]
        for c in table.c:
            list[0].append(c.key)
            list[1].append(c.type)
        return list

    def get_num_cols(self, table_name):
        """
        get column names of numeric type for table <table_name>
        :return: list of column names
        """
        table = self.metadata.tables[table_name]
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
        table = self.metadata.tables[table_name]
        # https://docs.sqlalchemy.org/en/14/core/metadata.html
        list = []
        for c in table.c:
            if str(c.type) in ('TEXT'):
                list += [c.key]
        return list

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