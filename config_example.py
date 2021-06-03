"""
Config for accessing database.
DB_TYPE: Type of database to connect to. Options: MYSQL, SQLITE, MSSQL (Only for Decision Tree Classifier as of now)
"""
DB_TYPE = "DB_TYPE"
DB_PATH = "DB_PATH"
DB_HOST = "DB_HOST"
DB_USER = "DB_USER"
DB_PW = "DB_PW"
DB_NAME = "DB_NAME"
DB_PORT = DB_PORT

"""

These fields are required for each connection type:

MySQL:
DB_TYPE = "DB_TYPE"
DB_PATH = "DB_PATH"
DB_HOST = "DB_HOST"
DB_USER = "DB_USER"
DB_PW = "DB_PW"
DB_NAME = "DB_NAME"
DB_PORT = DB_PORT

SQLite:
DB_TYPE = "DB_TYPE"
DB_PATH = "DB_PATH"
DB_NAME = "DB_NAME"

MSSQL:
DB_TYPE = "DB_TYPE"
DB_HOST = "DB_HOST"
DB_NAME = "DB_NAME"

"""