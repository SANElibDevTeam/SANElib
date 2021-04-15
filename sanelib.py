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
dtc = lib.DecisionTreeClassifier
