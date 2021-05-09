
import numpy as np
from lib.kmeans.sql_templates import SqlTemplates


class SqliteTemplates(SqlTemplates):
    """
    This is the class for sqlite specific statements.
    """

    def get_select_models(self):
        # sqlite has no information_schema.tables
        return "select name from sqlite_master where name like '%model';"

    def get_create_table_x(self, normalization, feature_names, d, tablename, table_x):        
        # sqlite does not support stdev()
        # sqlite does not support alter table add primary key
        if(normalization=="min-max"):
            features_with_alias = ", ".join([f"({feature_names[l]}-min_{l})/(max_{l}-min_{l}) as x_{l}" for l in range(d)])
            min_features = ", ".join([f"min({feature_names[l]}) as min_{l}" for l in range(d)])
            max_features = ", ".join([f"max({feature_names[l]}) as max_{l}" for l in range(d)])
            further_tables = f", (select {min_features}, {max_features} from {tablename}) min_max"
        elif(normalization=="z-score"):
            raise NotImplementedError("z-score-normalisation is not supported in sqlite because of the missing standard deviation.")
        else:
            features_with_alias = ", ".join([f"{feature_names[l]} as x_{l}" for l in range(d)])
            further_tables = ""
        query = f"select row_number() over () as i, {features_with_alias} from {tablename}{further_tables}"
        return [
            f"create table {table_x} as {query};",
        ]
        
    def get_add_cluster_columns(self, table_x):
        # sqlite doesn't allow multiple columns to be added in one statement
        return [
            f"alter table {table_x} add min_dist double;", f"alter table {table_x} add j int;"
        ]

    def get_init_table_c(self, table_c, table_x, n, d, start_indexes):
        # sqlite does not support update with join
        k = len(start_indexes)
        def get_setters_init(j):
            return ", ".join([f"x_{l}_{j} = (select x_{l} from {table_x} where i = {start_indexes[j]})" for l in range(d)])
        return [
            f"update {table_c} set {get_setters_init(j)};" for j in range(k)
        ]
    
    def get_set_clusters(self, table_c, table_x, d, k):
        # sqlite does not support update with join
        # sqlite also does not support power()
        # sqlite also does not support least()
        def get_distances(j):
            distance_per_feature = " + ".join([f"(({table_x}.x_{l} - {table_c}.x_{l}_{j})*({table_x}.x_{l} - {table_c}.x_{l}_{j}))" for l in range(d)])
            return f"({distance_per_feature}) as dist_{j}"
        distances_to_clusters = ", ".join([get_distances(j) for j in range(k)])
        sub_query_distances = f"select i, {distances_to_clusters} from {table_x}, {table_c} group by i"
        distances_columns = ", ".join([f"dist_{j}" for j in range(k)])
        case_dist_match = " ".join([f"when min_dist = (select dist_{j} from temp where temp.i = {table_x}.i) then {j}" for j in range(k)])
        return [
            f"create view temp as select *, min({distances_columns}) as min_dist from ({sub_query_distances});", 
            f"update {table_x} set min_dist = (select min_dist from temp where temp.i = {table_x}.i), j = case {case_dist_match} end;", "drop view temp;"
        ]

    def get_update_table_c(self, table_c, d, k, table_x):
        # sqlite does not support update with join
        def get_sub_selectors(j):
            return ", ".join([f"sum(x_{l})/count(*) as x_{l}_{j}" for l in range(d)])
        def get_setters_move(j):
            return ", ".join([f"x_{l}_{j} = case when (select x_{l}_{j} from temp) is null then x_{l}_{j} else (select x_{l}_{j} from temp) end" for l in range(d)])
        return np.concatenate([(f"create view temp as select {get_sub_selectors(j)} from {table_x} where j={j};", f"update {table_c} set {get_setters_move(j)};", "drop view temp;") for j in range(k)])

    def get_select_visualization(self, table_x, d, k):
        # sqlite can not handle union with part result with limit, the part result needs to be wrapped therefore
        feature_aliases = ", ".join([f"x_{l}" for l in range(d)])
        cluster_examples = " union ".join([f"select * from (select {feature_aliases}, j from {table_x} where j = {j} limit 500)" for j in range(k)])
        return f"{cluster_examples};"

    def get_select_silhouette_avg(self, table_x, d):
        # sqlite does not support power()
        # sqlite also does not support the alias of the main query to be used in a subquery
        # e.g.: "select i as o, (select i from iris_example_x where i=o) from iris_example_x;"
        return f"query not available;"
