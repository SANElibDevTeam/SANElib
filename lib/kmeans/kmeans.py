import logging
from random import randrange

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame


class KMeans:
    def __init__(self, db):
        self.__db = db
    
    def __generate_sql_preparation(self, tablename, feature_names, k, table_prefix, normalization=None):
        table_model = f"{table_prefix}_model"
        table_x = f"{table_prefix}_x"
        table_c = f"{table_prefix}_c"
        
        count_rows_query = f"select count(*) from {tablename};"
        n = self.__db.execute_query(count_rows_query)[0][0]
        d = len(feature_names)

        # statement parts
        columns = ", ".join([f"x_{l} as x_{l}_{j}" for j in range(k) for l in range(d)])
        def get_setters_init(j):
            return ", ".join([f"x_{l}_{j} = x_{l}" for l in range(d)])
        if(normalization=="min-max"):
            features_with_alias = ", ".join([f"({feature_names[l]}-min_{l})/(max_{l}-min_{l}) as x_{l}" for l in range(d)])
            min_features = ", ".join([f"min({feature_names[l]}) as min_{l}" for l in range(d)])
            max_features = ", ".join([f"max({feature_names[l]}) as max_{l}" for l in range(d)])
            further_tables = f", (select {min_features}, {max_features} from {tablename}) min_max"
        elif(normalization=="z-score"):
            features_with_alias = ", ".join([f"({feature_names[l]}-avg_{l})/stdev_{l} as x_{l}" for l in range(d)])
            avg_features = ", ".join([f"avg({feature_names[l]}) as avg_{l}" for l in range(d)])
            stdev_features = ", ".join([f"stddev({feature_names[l]}) as stdev_{l}" for l in range(d)]) # TODO: spelling may vary: stdev / stddev
            further_tables = f", (select {avg_features}, {stdev_features} from {tablename}) z_score"
        else:
            features_with_alias = ", ".join([f"{feature_names[l]} as x_{l}" for l in range(d)])
            further_tables = ""

        # statements
        statements = {
            "create_table_model": f"create table {table_model} as select {n} as n, {d} as d, {k} as k, 0 as steps from {tablename} limit 1;",
            "add_variance_column": f"alter table {table_model} add variance double;",
            "create_table_x": f"create table {table_x} as select row_number() over () as i, {features_with_alias} from {tablename}{further_tables};",
            "add_cluster_columns": f"alter table {table_x} add min_dist double, add j int;",
            "create_table_c": f"create table {table_c} as select {columns} from {table_x} where i = 1;",
            "init_table_c": [f"update {table_c} join {table_x} on i = {randrange(1, n)} set {get_setters_init(j)};" for j in range(k)],
        }
        return statements

    def __generate_sql(self, table_prefix):
        table_model = f"{table_prefix}_model"
        table_x = f"{table_prefix}_x"
        table_c = f"{table_prefix}_c"

        get_information = f"select n, d, k, steps, variance from {table_model};"
        query_result = self.__db.execute_query(get_information)
        n = query_result[0][0]
        d = query_result[0][1]
        k = query_result[0][2]
    
        # statement parts
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
            "get_information": get_information,
            "set_clusters": f"update {table_x} join (select *, least({distances_columns}) as min_dist from ({sub_query_distances}) distances) sub_table on sub_table.i = {table_x}.i set {table_x}.min_dist = sub_table.min_dist, j = case {case_dist_match} end;",
            "update_tabel_model": f"update {table_model} set steps = steps + 1, variance = (select sum(min_dist)/{n} from {table_x});",  
            "update_table_c": [f"update {table_c}, (select {get_sub_selectors(j)} from {table_x} where j={j}) sub_table set {get_setters_move(j)};" for j in range(k)],
            "get_visualization":  f"{cluster_examples};"
        }

        return statements

    def create_model(self, tablename, feature_names, k, model_identifier, normalization=None):
        model_name = f"{tablename}_{model_identifier}"
        self.drop_model(model_name)

        # get statements to initialize the model
        statements = self.__generate_sql_preparation(tablename, feature_names, k, model_name, normalization)

        # create and initialize table model
        self.__db.execute(statements["create_table_model"])
        self.__db.execute(statements["add_variance_column"])
        
        # create and initialize table x
        self.__db.execute(statements["create_table_x"])
        self.__db.execute(statements["add_cluster_columns"])
        
        # create and initialize table c
        self.__db.execute(statements["create_table_c"])
        for statement in statements["init_table_c"]:
            self.__db.execute(statement)
        
        # get statements to train the model
        statements = self.__generate_sql(model_name)
        return KMeansModel(self.__db, statements)

    def load_model(self, model_name):
        # get statements to train the model
        statements = self.__generate_sql(model_name)
        return KMeansModel(self.__db, statements)

    def drop_model(self, model_name):
        tables = ['model', 'x', 'c']
        for table in tables:
            self.__db.execute(f"drop table if exists {model_name}_{table};")


    def get_model_names(self):
        get_models_query = f"select table_name from information_schema.tables where table_name like '%model';"
        rows = self.__db.execute_query(get_models_query)
        model_names = []
        for row in rows:
            model_names.append(row[0][:-6])
        return model_names


class KMeansModel:
    def __init__(self, db, statements):
        self.__db = db
        self.__statements = statements

    def estimate(self, max_steps=100):        
        variance = -1
        step = 0
        while step < max_steps:
            step += 1
            self.__db.execute(self.__statements["set_clusters"])
            self.__db.execute(self.__statements["update_tabel_model"])

            last_variance = variance
            variance = self.get_information()["variance"]
            logging.info(f"step {step}: variance={variance}")
            if(last_variance == variance):
                break
            
            for statement in self.__statements["update_table_c"]:
                self.__db.execute(statement)

        return self

    def get_information(self):
        query_result = self.__db.execute_query(self.__statements["get_information"])
        return {
            "n": query_result[0][0],
            "d": query_result[0][1],
            "k": query_result[0][2],
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
