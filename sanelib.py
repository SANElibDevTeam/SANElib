# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

import constants as cons
import lib
import util.log_handler
from util.database_connection import Database

db_connection = {
    "drivername": "mysql+mysqlconnector",
    "host": cons.DB_HOST,
    "port": cons.DB_PORT,
    "username": cons.DB_USER,
    "password": cons.DB_PW,
    "database": cons.DB_NAME,
    "query": {"charset": "utf8"}
}

kmeans = lib.kmeans.KMeans(Database(db_connection))
