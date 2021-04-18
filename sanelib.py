# SANElib Prototype
# Standard-SQL Analytics for Numerical Estimation – SANE
# (c) 2020  Michael Kaufmann, Gabriel Stechschulte, Anna Huber, HSLU

import config as conf
import lib
import util.log_handler
from util.database_connection import Database

db_connection = {
    "drivername": "mysql+mysqlconnector",
    "host": conf.DB_HOST,
    "port": conf.DB_PORT,
    "username": conf.DB_USER,
    "password": conf.DB_PW,
    "database": conf.DB_NAME,
    "query": {"charset": "utf8"}
}

kmeans = lib.kmeans.KMeans(Database(db_connection))
