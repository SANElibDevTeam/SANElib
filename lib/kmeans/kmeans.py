import logging
from random import randrange

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame


class KMeans:
    def __init__(self, db):
        self.__db = db
    
    def __generate_sql(self, tablename, feature_names, k, model_identifier):
        # get number of rows and columns
        count_rows_query = f"select count(*) from {tablename};"
        n = self.__db.execute_query(count_rows_query)[0][0]
        d = len(feature_names)
        
        # statement parts
        table_prefix = f"{tablename}_{model_identifier}"
        table_model = f"{table_prefix}_model"
        table_x = f"{table_prefix}_x"
        table_c = f"{table_prefix}_c"
        features_with_alias = ", ".join([f"{feature_names[l]} as x_{l}" for l in range(d)])
        columns = ", ".join([f"x_{l} as x_{l}_{j}" for j in range(k) for l in range(d)])
        def get_setters_init(j):
            return ", ".join([f"x_{l}_{j} = x_{l}" for l in range(d)])
        def get_distances(j):
            distance_per_feature = " + ".join([f"power(({table_x}.x_{l} - {table_c}.x_{l}_{j}),2)" for l in range(d)])
            return f"({distance_per_feature}) as dist_{j}"
        distances_columns = ", ".join([f"dist_{j}" for j in range(k)])
        distances_to_clusters = ", ".join([get_distances(j) for j in range(k)])
        sub_query_distances = f"select i, {distances_to_clusters} from {table_x}, {table_c} group by i"
        case_dist_match = " ".join([f"when dist_{j} = sub_table.min_dist then {j}" for j in range(k)])
        def get_sub_selectors(j):
            return ", ".join([f"sum(x_{l})/count(*) as x_{l}_{j}" for l in range(d)])
        def get_setters_move(j):
            return ", ".join([f"{table_c}.x_{l}_{j} = case when sub_table.x_{l}_{j} is null then {table_c}.x_{l}_{j} else sub_table.x_{l}_{j} end" for l in range(d)])
        feature_aliases = ", ".join([f"x_{l}" for l in range(d)])
        cluster_examples = " UNION ".join([f"(select {feature_aliases}, j from {table_x} where j = {j} LIMIT 1000)" for j in range(k)])

        # statements
        statements = {
             # NOTE: update with join doesn't work in sqlite
            "create_table_model": f"select {d} as d, {k} as k, {n} as n, 0 as steps from {tablename};",
            "add_variance_column": f"alter table {table_model} add variance double;",
            "get_information": f"select * from {table_model};",
            "create_table_x": f"select row_number() over () as i, {features_with_alias} from {tablename};",
            "add_cluster_columns": f"alter table {table_x} add min_dist double, add j int;",
            "create_table_c": f"select {columns} from {table_x} where i = 1;",
            "init_table_c": [f"update {table_c} join {table_x} on i = {randrange(1, n)} set {get_setters_init(j)};" for j in range(k)],
            "set_clusters": f"update {table_x} join (select *, least({distances_columns}) as min_dist from ({sub_query_distances}) distances) sub_table on sub_table.i = {table_x}.i set {table_x}.min_dist = sub_table.min_dist, j = case {case_dist_match} end;",
            "update_tabel_model": f"update {table_model} set steps = steps + 1, variance = (select sum(min_dist)/{n} from {table_x});",  
            "update_table_c": [f"update {table_c}, (select {get_sub_selectors(j)} from {table_x} where j={j}) sub_table set {get_setters_move(j)};" for j in range(k)],
            "get_visualization":  f"{cluster_examples};"
        }

        # tables
        tables = {
            "model": table_model,
            "x": table_x,
            "c": table_c
        }

        return tables, statements

    def create_model(self, tablename, feature_names, k, model_identifier):
        tables, statements = self.__generate_sql(tablename, feature_names, k, model_identifier)

        # create and initialise table model
        self.__db.create_materialized_view(tables["model"], statements["create_table_model"])
        self.__db.execute_query_without_result(statements["add_variance_column"])
        
        # create and initialise table x
        self.__db.create_materialized_view(tables["x"], statements["create_table_x"])
        self.__db.execute_query_without_result(statements["add_cluster_columns"])
        
        # create and initialise table c
        self.__db.create_materialized_view(tables["c"], statements["create_table_c"])
        for statement in statements["init_table_c"]:
            self.__db.execute_query_without_result(statement)
        
        return KMeansModel(self.__db, statements)

class KMeansModel:
    def __init__(self, db, statements):
        self.__db = db
        self.__statements = statements

    def estimate(self, max_steps=100):        
        variance = -1
        step = 0
        while step < max_steps:
            step += 1
            self.__db.execute_query_without_result(self.__statements["set_clusters"])
            self.__db.execute_query_without_result(self.__statements["update_tabel_model"])

            last_variance = variance
            variance = self.get_information()["variance"]
            logging.info(f"step {step}: variance={variance}")
            if(last_variance == variance):
                break
            
            for statement in self.__statements["update_table_c"]:
                self.__db.execute_query_without_result(statement)

        return self

    def get_information(self):
        query_result = self.__db.execute_query(self.__statements["get_information"])
        return {
            "d": query_result[0][0],
            "k": query_result[0][1],
            "n": query_result[0][2],
            "steps": query_result[0][3],
            "variance": query_result[0][4]
        }

    def visualize(self, feature_names, axis_order=None):
        if axis_order is None:
            axis_order = range(len(feature_names))
        d = len(axis_order[:3])
        features = [f"x_{l}" for l in range(d)]

        query_result = self.__db.execute_query(self.__statements["get_visualization"])
        feature_names.append("j")
        df = DataFrame(query_result, columns=feature_names)

        x = df[feature_names].values
        y = df["j"].values

        fig = plt.figure(0, figsize=(9, 6))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=60, azim=270, auto_add_to_figure=False)
        fig.add_axes(ax)
        

        x_label, x_scatter,  = self.__get_axis(x, axis_order, feature_names, 0)
        y_label, y_scatter = self.__get_axis(x, axis_order, feature_names, 1)
        z_label, z_scatter = self.__get_axis(x, axis_order, feature_names, 2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        ax.scatter(x_scatter, y_scatter, z_scatter, c=y, edgecolor='k')

        plt.show()
        return self
    
    def __get_axis(self, x, axis_order, feature_names, axis_index):
        if axis_index < len(axis_order):
            label = feature_names[axis_order[axis_index]]
            scatter = x[:, axis_order[axis_index]]
        else:
            label = ''
            scatter = 0
        return label, scatter
