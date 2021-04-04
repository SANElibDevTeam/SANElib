
class KMeans:
    def __init__(self, db):
        self.__db = db
    
    def __generate_sql(self, tablename, feature_names, k, model_identifier, n):
        d = len(feature_names)
        table_prefix = f"{tablename}_{model_identifier}"
        table_model = f"{table_prefix}_model"

        tables = {
            "model": table_model
        }

        statements = {
            "init_model": f"select {d} as d, {k} as k, {n} as n, 0 as iteration, -1 as avg_q from {tablename};",
            "get_information": f"select * from {table_model};"
        }
        return tables, statements

    def create_model(self, tablename, feature_names, k, model_identifier):
        count_rows_query = f"select count(*) from {tablename};"
        n = self.__db.execute_query(count_rows_query)[0][0]
        tables, statements = self.__generate_sql(tablename, feature_names, k, model_identifier, n)
        self.__db.create_materialized_view(tables["model"], statements['init_model'])     
        return KMeansModel(self.__db, tables, statements)

class KMeansModel:
    def __init__(self, db, tables, statements):
        self.__db = db
        self.__tables = tables
        self.__statements = statements

    def get_information(self):
        query_result = self.__db.execute_query(self.__statements['get_information'])
        return {
            "d": query_result[0][0],
            "k": query_result[0][1],
            "n": query_result[0][2],
            "iteration": query_result[0][3],
            "avg_q": query_result[0][4]
        }
