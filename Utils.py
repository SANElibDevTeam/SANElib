from sqlalchemy import text
import pandas as pd
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from sqlalchemy import text



def set_connection(db):
    """
    :param db: dict with connection information for DB
    :return: SQLAlchemy Engine
    """
    return create_engine(URL(**db),pool_pre_ping=True)


def get_connection(engine):
    """
    :return: Connection to self.engine
    """
    return engine.connect()



def disconnect_connection(engine):
    """
    Close connection to self.engine
    """
    engine.close()



def execute(desc, query, engine):
    connection = get_connection(engine)
    print(desc + '\nQuery: ' + query)
    connection.execute(text(query))
    connection.close()
    print('OK: ' + desc)
    print()



def executeQuery(desc, query, engine):
    connection = get_connection(engine)
    print('Query: ' + query)
    results = connection.execute(text(query))
    results = results.fetchall()
    connection.close()
    print('OK: ' + desc)
    print()

    return results


def materializedView(desc, tablename, query, engine):
    execute('Dropping table ' + tablename, '''
        drop table if exists {}'''
                     .format(tablename), engine)
    execute(desc, '''
        create table {} as '''
                     .format(tablename) + query, engine)