# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

import lib
import config as cons

db = {
    'drivername': 'mysql+mysqlconnector',
    'host': cons.DB_HOST,
    'port': cons.DB_PORT,
    'username': cons.DB_USER,
    'password': cons.DB_PW,
    'database': cons.DB_NAME,
    'query': {'charset': 'utf8'}
}

mdh = lib.mdh.MDH(db)
