# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

import lib
import config as conf
from util.database import Database

db_config = {
    'drivername': 'mysql+mysqlconnector',
    'host': conf.DB_HOST,
    'port': conf.DB_PORT,
    'username': conf.DB_USER,
    'password': conf.DB_PW,
    'database': conf.DB_NAME,
    'query': {'charset': 'utf8'}
}

db = Database(db_config)

mdh = lib.mdh.MDH(db)
linear_regression = lib.linear_regression.LinearRegression(db)
