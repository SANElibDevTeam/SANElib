# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

import lib
import config as conf
from util.database import Database

if conf.DB_TYPE == "MYSQL":
    driver_name = 'mysql+mysqlconnector'
    db_config = {
        'drivername': driver_name,
        'host': conf.DB_HOST,
        'port': conf.DB_PORT,
        'username': conf.DB_USER,
        'password': conf.DB_PW,
        'database': conf.DB_NAME,
        'query': {'charset': 'utf8'}
    }
elif conf.DB_TYPE == "SQLITE":
    driver_name = 'sqlite'
    db_config = {
        'drivername': driver_name,
        'database': conf.DB_NAME,
        'path': conf.DB_PATH
    }
else:
    raise Exception('No valid DB_TYPE (config.py) provided! Please provide one of the following types: \n MYSQL\n SQLITE')

db = Database(db_config)

mdh = lib.mdh.MDH(db)
linear_regression = lib.linear_regression.LinearRegression(db)
