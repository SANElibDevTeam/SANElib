import logging
from random import randrange, uniform

import matplotlib.pyplot as plt
from lib.kmeans.template_factory import get_templates
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame


class KMeans:
    def __init__(self, db):
        self.__db = db
        self.__templates = get_templates(db.driver_name)

    def __generate_sql(self, table_model, table_c, table_x, n, d, k):
        statements = {
            "select_information": self.__templates.get_select_information(table_model),
            "set_clusters": self.__templates.get_set_clusters(table_c, table_x, d, k),
            "update_table_model": self.__templates.get_update_table_model(table_model, n, table_x),  
            "update_table_c": self.__templates.get_update_table_c(table_c, d, k, table_x),
            "select_visualization": self.__templates.get_select_visualization(table_x, d, k),
            "select_silhouette_avg": self.__templates.get_select_silhouette_avg(table_x, d),
        }
        return statements

    def create_ideal_model(self, tablename, feature_names, k_list, model_identifier, normalization=None):
        if self.__db.driver_name == "sqlite":
            raise NotImplementedError("silhouette is not supported with sqlite")
        best_results = -10
        best_k = 0
        for k in k_list:
            result = self.create_model(tablename, feature_names, k, f"{model_identifier}_k{k}", normalization).estimate().get_silhouette_avg()
            logging.info(f"result for k={k}: {result}")
            if(best_results < result):
                best_results = result
                best_k = k
        return self.load_model(f"{tablename}_{model_identifier}_k{best_k}")

    def create_model(self, tablename, feature_names, k, model_identifier, normalization=None):
        model_name = f"{tablename}_{model_identifier}"
        table_model = f"{model_name}_model"
        table_x = f"{model_name}_x"
        table_c = f"{model_name}_c"
        
        self.drop_model(model_name)
        count_rows_query = self.__templates.get_row_count(tablename)
        n = self.__db.execute_query(count_rows_query)[0][0]
        d = len(feature_names)
        start_indexes = [randrange(1, n) for _ in range(k)]
        
        # statements
        statements = {
            "create_table_model": self.__templates.get_create_table_model(table_model, n, d, k, tablename),
            "add_variance_column": self.__templates.get_add_variance_column(table_model),
            "create_table_x": self.__templates.get_create_table_x(normalization, feature_names, d, tablename, table_x),
            "add_cluster_columns": self.__templates.get_add_cluster_columns(table_x),
            "create_table_c": self.__templates.get_create_table_c(d, k, table_c, table_x),
            "init_table_c": self.__templates.get_init_table_c(table_c, table_x, n, d, start_indexes),
        }

        # create and initialize table model
        self.__db.execute(statements["create_table_model"])
        self.__db.execute(statements["add_variance_column"])
        
        # create and initialize table x
        for statement in statements["create_table_x"]:
            self.__db.execute(statement)
        for statement in statements["add_cluster_columns"]:
            self.__db.execute(statement)
        
        # create and initialize table c
        self.__db.execute(statements["create_table_c"])
        for statement in statements["init_table_c"]:
            self.__db.execute(statement)

        statements = self.__generate_sql(table_model, table_c, table_x, n, d, k)
        return KMeansModel(self.__db, statements)

    def load_model(self, model_name):
        table_model = f"{model_name}_model"
        table_x = f"{model_name}_x"
        table_c = f"{model_name}_c"

        select_information = self.__templates.get_select_information(table_model)
        query_result = self.__db.execute_query(select_information)

        n = query_result[0][0]
        d = query_result[0][1]
        k = query_result[0][2]
        
        statements = self.__generate_sql(table_model, table_c, table_x, n, d, k)
        return KMeansModel(self.__db, statements)

    def drop_model(self, model_name):
        tables = ['model', 'x', 'c']
        for table in tables:
            self.__db.execute(self.__templates.get_drop_model(model_name, table))

    def get_model_names(self):
        select_models = self.__templates.get_select_models()
        rows = self.__db.execute_query(select_models)
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
            for statement in self.__statements["set_clusters"]:
                self.__db.execute(statement)
            self.__db.execute(self.__statements["update_table_model"])

            last_variance = variance
            variance = self.get_information()["variance"]
            logging.info(f"step {step}: variance={variance}")
            if(last_variance == variance):
                break
            
            for statement in self.__statements["update_table_c"]:
                self.__db.execute(statement)

        return self

    def get_information(self):
        query_result = self.__db.execute_query(self.__statements["select_information"])
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

        query_result = self.__db.execute_query(self.__statements["select_visualization"])
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

    def get_silhouette_avg(self):
        if self.__db.driver_name == "sqlite":
            raise NotImplementedError("silhouette is not supported with sqlite")
        query_result = self.__db.execute_query(self.__statements["select_silhouette_avg"])
        return query_result[0][0]
    
    def __get_axis(self, x, axis_order, feature_names, axis_index):
        if axis_index < len(axis_order):
            label = feature_names[axis_order[axis_index]]
            scatter = x[:, axis_order[axis_index]]
        else:
            label = ''
            scatter = 0
        return label, scatter
