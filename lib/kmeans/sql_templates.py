from random import randrange

import numpy as np


# factory method
def get_templates(driver_name):
    if driver_name== "sqlite":
        return SqliteTemplates()
    else:
        return SqlTemplates()


# parent class for default statements
class SqlTemplates:
    def get_row_count(self, tablename):
        return f"select count(*) as row_count from {tablename};"

    def get_create_table_model(self, table_model, n, d, k, tablename):
        return f"create table {table_model} as select {n} as n, {d} as d, {k} as k, 0 as steps from {tablename} limit 1;"

    def get_add_variance_column(self, table_model):
        return f"alter table {table_model} add variance double;"

    def get_create_table_x(self, normalization, feature_names, d, tablename, table_x):
        if(normalization=="min-max"):
            features_with_alias = ", ".join([f"({feature_names[l]}-min_{l})/(max_{l}-min_{l}) as x_{l}" for l in range(d)])
            min_features = ", ".join([f"min({feature_names[l]}) as min_{l}" for l in range(d)])
            max_features = ", ".join([f"max({feature_names[l]}) as max_{l}" for l in range(d)])
            further_tables = f", (select {min_features}, {max_features} from {tablename}) min_max"
        elif(normalization=="z-score"):
            features_with_alias = ", ".join([f"({feature_names[l]}-avg_{l})/stdev_{l} as x_{l}" for l in range(d)])
            avg_features = ", ".join([f"avg({feature_names[l]}) as avg_{l}" for l in range(d)])
            stdev_features = ", ".join([f"stddev({feature_names[l]}) as stdev_{l}" for l in range(d)])
            further_tables = f", (select {avg_features}, {stdev_features} from {tablename}) z_score"
        else:
            features_with_alias = ", ".join([f"{feature_names[l]} as x_{l}" for l in range(d)])
            further_tables = ""
        return f"create table {table_x} as select row_number() over () as i, {features_with_alias} from {tablename}{further_tables};"

    def get_add_cluster_columns(self, table_x):
        return [f"alter table {table_x} add min_dist double, add j int;"]

    def get_create_table_c(self, d, k, table_c, table_x):
        columns = ", ".join([f"x_{l} as x_{l}_{j}" for j in range(k) for l in range(d)])
        return f"create table {table_c} as select {columns} from {table_x} where i = 1;"

    def get_init_table_c(self, table_c, table_x, n, d, k):
        def get_setters_init(j):
            return ", ".join([f"x_{l}_{j} = x_{l}" for l in range(d)])
        return [f"update {table_c} join {table_x} on i = {randrange(1, n)} set {get_setters_init(j)};" for j in range(k)]
    
    def get_select_models(self):
        return f"select table_name from information_schema.tables where table_name like '%model';"

    def get_select_information(self, table_model):
        return f"select n, d, k, steps, variance from {table_model};"

    def get_set_clusters(self, table_c, table_x, d, k):
        def get_distances(j):
            distance_per_feature = " + ".join([f"power(({table_x}.x_{l} - {table_c}.x_{l}_{j}),2)" for l in range(d)])
            return f"({distance_per_feature}) as dist_{j}"
        distances_to_clusters = ", ".join([get_distances(j) for j in range(k)])
        sub_query_distances = f"select i, {distances_to_clusters} from {table_x}, {table_c} group by i"
        distances_columns = ", ".join([f"dist_{j}" for j in range(k)])
        case_dist_match = " ".join([f"when dist_{j} = sub_table.min_dist then {j}" for j in range(k)])
        return [f"update {table_x} join (select *, least({distances_columns}) as min_dist from ({sub_query_distances}) distances) sub_table on sub_table.i = {table_x}.i set {table_x}.min_dist = sub_table.min_dist, j = case {case_dist_match} end;"]
            
    def get_update_table_model(self, table_model, n, table_x):
        return f"update {table_model} set steps = steps + 1, variance = (select sum(min_dist)/{n} from {table_x});"

    def get_update_table_c(self, table_c, d, k, table_x):
        def get_sub_selectors(j):
            return ", ".join([f"sum(x_{l})/count(*) as x_{l}_{j}" for l in range(d)])
        def get_setters_move(j):
            return ", ".join([f"{table_c}.x_{l}_{j} = case when sub_table.x_{l}_{j} is null then {table_c}.x_{l}_{j} else sub_table.x_{l}_{j} end" for l in range(d)])
        return [f"update {table_c}, (select {get_sub_selectors(j)} from {table_x} where j={j}) sub_table set {get_setters_move(j)};" for j in range(k)]

    def get_select_visualization(self, table_x, d, k):
        feature_aliases = ", ".join([f"x_{l}" for l in range(d)])
        cluster_examples = " union ".join([f"(select {feature_aliases}, j from {table_x} where j = {j} limit 500)" for j in range(k)])
        return f"{cluster_examples};"

    def get_drop_model(self, model_name, table):
        return f"drop table if exists {model_name}_{table};"

# class for sqlite specific statements
class SqliteTemplates(SqlTemplates):
    # sqlite has no information_schema.tables
    def get_select_models(self):
        return "select name from sqlite_master;"

    # sqlite does not support stdev()
    def get_create_table_x(self, normalization, feature_names, d, tablename, table_x):
        if(normalization=="z-score"):
            raise NotImplementedError("z-score-normalisation is not supported in sqlite because of the missing standard deviation.")
        else:
            return super().get_create_table_x(normalization, feature_names, d, tablename, table_x)

    # sqlite doesn't allow multiple columns to be added in one statement
    def get_add_cluster_columns(self, table_x):
        return [f"alter table {table_x} add min_dist double;", f"alter table {table_x} add j int;"]

    # sqlite does not support update with join
    def get_init_table_c(self, table_c, table_x, n, d, k):
        def get_setters_init(j):
            rand_index = randrange(1, n)
            return ", ".join([f"x_{l}_{j} = (select x_{l} from {table_x} where i = {rand_index})" for l in range(d)])
        return [f"update {table_c} set {get_setters_init(j)};" for j in range(k)]
    
    # sqlite does not support update with join
    # sqlite also does not support power()
    # sqlite also does not support least()
    def get_set_clusters(self, table_c, table_x, d, k):
        def get_distances(j):
            distance_per_feature = " + ".join([f"(({table_x}.x_{l} - {table_c}.x_{l}_{j})*({table_x}.x_{l} - {table_c}.x_{l}_{j}))" for l in range(d)])
            return f"({distance_per_feature}) as dist_{j}"
        distances_to_clusters = ", ".join([get_distances(j) for j in range(k)])
        sub_query_distances = f"select i, {distances_to_clusters} from {table_x}, {table_c} group by i"
        distances_columns = ", ".join([f"dist_{j}" for j in range(k)])
        case_dist_match = " ".join([f"when min_dist = (select dist_{j} from temp where temp.i = {table_x}.i) then {j}" for j in range(k)])
        return [f"create view temp as select *, min({distances_columns}) as min_dist from ({sub_query_distances});", f"update {table_x} set min_dist = (select min_dist from temp where temp.i = {table_x}.i), j = case {case_dist_match} end;", "drop view temp;"]

    # sqlite does not support update with join
    def get_update_table_c(self, table_c, d, k, table_x):
        def get_sub_selectors(j):
            return ", ".join([f"sum(x_{l})/count(*) as x_{l}_{j}" for l in range(d)])
        def get_setters_move(j):
            return ", ".join([f"x_{l}_{j} = case when (select x_{l}_{j} from temp) is null then x_{l}_{j} else (select x_{l}_{j} from temp) end" for l in range(d)])
        return np.concatenate([(f"create view temp as select {get_sub_selectors(j)} from {table_x} where j={j};", f"update {table_c} set {get_setters_move(j)};", "drop view temp;") for j in range(k)])

    # sqlite can not handle union with part result with limit, the part result needs to be wrapped therefore
    def get_select_visualization(self, table_x, d, k):
        feature_aliases = ", ".join([f"x_{l}" for l in range(d)])
        cluster_examples = " union ".join([f"select * from (select {feature_aliases}, j from {table_x} where j = {j} limit 500)" for j in range(k)])
        return f"{cluster_examples};"
