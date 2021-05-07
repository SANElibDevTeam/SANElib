import lib
import config as cons

if 'mysql' in cons.DB_ENGINE:
    drivername = 'mysql+mysqlconnector'
elif 'sqlite' in cons.DB_ENGINE:
    drivername = 'sqlite'
elif 'microsoft' or 'mssql' in cons.DB_ENGINE:
    drivername = 'mssql + pyodbc'

db = {
    'drivername': drivername,
    'host': cons.DB_HOST,
    'port': cons.DB_PORT,
    'username': cons.DB_USER,
    'password': cons.DB_PW,
    'database': cons.DB_NAME,
    'query': {'charset': 'utf8'}
}

mdh = lib.mdh.MDH(db)
dtc = lib.DecisionTreeClassifier
