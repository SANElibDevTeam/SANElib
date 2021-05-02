from random import randrange

import numpy as np
from lib.kmeans.sql_templates import SqlTemplates


# class for sqlite specific statements
class SqliteTemplates(SqlTemplates):
    # sqlite has no information_schema.tables
    def get_select_models(self):
        return "select name from sqlite_master where name like '%model';"

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
