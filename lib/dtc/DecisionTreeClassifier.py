from util.database_connection import Database
from lib.dtc.tree import Node
from jinja2 import Template
import pandas as pd
import numpy as np
import concurrent.futures
import config as cons

if 'MSSQL' in cons.DB_TYPE:
    from lib.dtc import sqlTemplates_mssql as sql
elif 'MYSQL' in cons.DB_TYPE:
    from lib.dtc import sqlTemplates_mysql as sql
elif 'SQLITE' in cons.DB_TYPE:
    from lib.dtc import sqlTemplates_sqlite as sql


class DecisionTreeClassifier:
    def __init__(self, db, dataset='', target='', target_classes=None, max_samples=2, table_train='', table_eval='',
                 max_mutual_inf=0):
        self.db_connection = Database(db)
        self.engine = self.db_connection.engine
        self.dataset = dataset
        self.max_samples = max_samples
        self.table_indices = {}
        self.model = []
        self.max_mutual_inf = max_mutual_inf

        if target == '':
            colnames = self.db_connection.execute_query(
                Template(sql.tmplt['_getColumns']).render(input=self), self.engine, True)
            self.target = colnames.columns[-1]
        else:
            self.target = target

        if target_classes is None:
            target_classes = self.db_connection.execute_query(
                Template(sql.tmplt["_distinctValues"]).render(table=self.dataset, column=self.target),
                self.engine, True)
            self.target_classes = target_classes.to_numpy().flatten()
        else:
            self.target_classes = target_classes

        if table_train == '':
            self.table_train = self.dataset + '_train'
        else:
            self.table_train = table_train
        if table_eval == '':
            self.table_eval = self.dataset + '_eval'
        else:
            self.table_eval = table_eval

        # Feature Selection
        # Get first row of dataset to see the which features are numerical, which are not
        features = \
            self.db_connection.execute_query(Template(sql.tmplt['_getColumns']).render(input=self), self.engine, True)
        # Select only the non numerical features and putting them into a dataframe
        self.catFeatures = list(features.drop(columns=self.target).select_dtypes(include='object').columns)
        self.numFeatures = list(features.drop(columns=self.target).select_dtypes(exclude='object').columns)

    def train_test_split(self, ratio=0.8, seed=1, encode=False):
        self.ratio = ratio
        self.seed = seed
        if encode:
            self.__feature_encoding()
        else:
            self.db_connection.materializedView(
                'Splitting table into training set',
                self.table_train,
                Template(sql.tmplt['_train']).render(input=self), self.engine)

            self.db_connection.materializedView(
                'Splitting table into evaluation set',
                self.table_eval,
                Template(sql.tmplt['_eval']).render(input=self), self.engine)

    def __feature_encoding(self):
        print("Encoding categorical values into ordinal values")
        # Get all characteristics of each categorical feature
        features = {}
        for f in self.catFeatures:
            features[f] = np.array(
                self.db_connection.execute_query(
                    Template(sql.tmplt["_distinctValues"]).render(table=self.dataset, column=f),
                    self.engine))

        # if the the features contain a number, sort the characteristics by its number, if not natural sorting
        # todo

        self.catFeatures = features

        # rename categorical columns since the reformatted will have the same name and the originals will be dropped
        # later
        for key in self.catFeatures:
            self.db_connection.execute(
                Template(sql.tmplt['_renameColumn']).render(input=self, orig=key, to=key + '_orig'), self.engine)

        # in the encode table train eval template, the categorical features will be encoded
        # according to the capabilities of each database technology
        # since each database engine connector has different SQL capabilities, the _encodeTableTrainEval entry
        # has several array items. they need to be iterated, since a connector can't handle multiple statements
        # depending on the database engine
        for i in sql.tmplt['_encodeTableTrainEval']:
            self.db_connection.execute(
                Template(i).render(input=self), self.engine)

        for key in self.catFeatures:
            self.db_connection.execute(
                Template(sql.tmplt['_renameColumn']).render(input=self, orig=key + '_orig', to=key), self.engine)

        self.catFeatures = list(self.catFeatures.keys())

    def __create_cc_table(self, query=''):
        self.db_connection.materializedView(
            'creating counts table',
            self.dataset + '_CC_table',
            Template(sql.tmplt['_CC_table']).render(input=self,
                                                    subquery="(select * from {} {})".format(self.table_train, query)),
            self.engine)

    def __get_cc_table(self):
        cc_table = self.db_connection.execute_query(
            query='select * from {}_CC_table'.format(self.dataset),
            engine=self.engine, as_df=True)

        return cc_table

    def __calc_mutual_information(self, cc_table):
        if len(self.catFeatures) != 0:
            minfcat = self.__calc_mutual_information_cat(cc_table.loc[cc_table.f.isin(self.catFeatures)])
            minfnum = self.__calc_mutual_information_num(cc_table.loc[cc_table.f.isin(self.numFeatures)])
            minf = pd.concat([minfnum.rename(columns={'cum_nx_': 'nx_'}), minfcat])
        else:
            minfnum = self.__calc_mutual_information_num(cc_table.loc[cc_table.f.isin(self.numFeatures)])
            minf = minfnum.rename(columns={'cum_nx_': 'nx_'})
        return minf.sort_values(by='mi', ascending=False).head(1)

    def __calc_mutual_information_cat(self, cc_table):
        # Creating nx_,ndy_gx,min_nxy
        cc_table = cc_table.join(cc_table.groupby(['f', 'x']).nxy.agg(nx_='sum'), on=['f', 'x'])

        # Creating n__
        cc_table = cc_table.join(cc_table.groupby(['f']).nxy.agg(n__='sum'), on=['f'])

        # Creating n_y
        cc_table = cc_table.join(cc_table.groupby(['f', 'y']).nxy.agg(n_y='sum'), on=['f', 'y'])

        # Saving the sorted dataframe
        cc_table = cc_table.sort_values(by=['f', 'y', 'x'])

        # Calculating the probabiliy columns and the unsummized mutual information
        cc_table['p_xy'] = cc_table['nxy'] / cc_table['n__']
        cc_table['p_y'] = cc_table['n_y'] / cc_table['n__']
        cc_table['p_x'] = cc_table['nx_'] / cc_table['n__']
        cc_table['mi'] = cc_table['p_xy'] * np.log2(cc_table['p_xy'] / (cc_table['p_y'] * cc_table['p_x']))

        # Creating the mutual_information dataframe
        mutual_inf = cc_table.groupby(['f', 'x'], as_index=False).agg({"mi": "sum", "nx_": "min", "n__": "min"})

        # Creating the alt column, which is how many items are NOT in this group
        mutual_inf['alt'] = mutual_inf['n__'] - mutual_inf['nx_']

        # Overwriting mutual_inf with only the neccessary information where the max value of mi is
        mutual_inf = \
            mutual_inf.sort_values(by=['mi'], ascending=False)

        for i in self.target_classes:
            mutual_inf['ny' + str(i)] = cc_table.loc[
                (cc_table.y == i) & (cc_table.f == mutual_inf.f.values[0]) & (
                        cc_table.x <= mutual_inf.x.values[0])].nxy.sum()
        for i in self.target_classes:
            mutual_inf['alt_ny' + str(i)] = cc_table.loc[
                (cc_table.y == i) & (cc_table.f == mutual_inf.f.values[0]) & (
                        cc_table.x > mutual_inf.x.values[0])].nxy.sum()
        return mutual_inf.head(1)

    def __calc_mutual_information_num(self, cc_table):
        # Creating nx_,ndy_gx,min_nxy
        cc_table = cc_table.join(cc_table.groupby(['f', 'x']).nxy.agg(nx_='sum', ndy_gx='count', min_nxy='min'),
                                 on=['f', 'x'])

        # Creating n__
        cc_table = cc_table.join(cc_table.groupby(['f']).nxy.agg(n__='sum'), on=['f'])

        # Creating n_y
        cc_table = cc_table.join(cc_table.groupby(['f', 'y']).nxy.agg(n_y='sum'), on=['f', 'y'])

        # Creating ndygx - which is a division from the nx_ and ndy_gx columns
        # This interim step is being done for faster processing as opposed to a seperate function
        cc_table['ndygx'] = cc_table.nx_ / cc_table.ndy_gx

        # Calculating the cum_nx_ column
        cc_table = cc_table.merge( \
            cc_table.groupby(['f', 'x']).ndygx.agg('sum').groupby(level=0).agg(cum_nx_='cumsum').round().reset_index() \
            , on=['f', 'x']).sort_values(by=['f', 'y', 'x'])

        # Calculating the cum_nxy column
        cc_table['cum_nxy'] = cc_table.sort_values(by=['x']).groupby(['f', 'y']).nxy.transform(np.cumsum)

        # Saving the sorted dataframe
        cc_table = cc_table.sort_values(by=['f', 'y', 'x'])

        # Calculating the probabiliy columns and the unsummized mutual information
        cc_table['p_xy'] = cc_table['cum_nxy'] / cc_table['n__']
        cc_table['p_y'] = cc_table['n_y'] / cc_table['n__']
        cc_table['p_x'] = cc_table['cum_nx_'] / cc_table['n__']
        cc_table['mi'] = cc_table['p_xy'] * np.log2(cc_table['p_xy'] / (cc_table['p_y'] * cc_table['p_x']))

        # Creating the mutual_information dataframe
        mutual_inf = cc_table.groupby(['f', 'x'], as_index=False).agg({"mi": "sum", "cum_nx_": "min", "n__": "min"})

        # Creating the mx column, which is the max value of mi
        mutual_inf = mutual_inf.groupby(['f']).mi.agg(mx='max').merge(mutual_inf, on=['f'])

        # Creating the alt column, which is how many items are NOT in this group
        mutual_inf['alt'] = mutual_inf['n__'] - mutual_inf['cum_nx_']

        # Overwriting mutual_inf with only the neccessary information where the max value of mi is
        mutual_inf = \
            mutual_inf.loc[mutual_inf.mi == mutual_inf.mx][['f', 'x', 'mi', 'cum_nx_', 'alt', 'n__']].sort_values(
                by=['mi'], ascending=False)
        mutual_inf = mutual_inf.loc[mutual_inf.mi == mutual_inf.mi.agg('max')]

        for i in self.target_classes:
            mutual_inf['ny' + str(i)] = cc_table.loc[
                (cc_table.y == i) & (cc_table.f == mutual_inf.f.values[0]) & (
                        cc_table.x <= mutual_inf.x.values[0])].nxy.sum()
        for i in self.target_classes:
            mutual_inf['alt_ny' + str(i)] = cc_table.loc[
                (cc_table.y == i) & (cc_table.f == mutual_inf.f.values[0]) & (
                        cc_table.x > mutual_inf.x.values[0])].nxy.sum()
        return mutual_inf.head(1)

    def estimate(self):
        """Build decision tree classifier."""
        self.n_classes_ = len(self.target_classes)  # classes are assumed to go from 0 to n-1
        # self.tree_ = self.__initial_nodes()
        self.tree_ = self.__grow_tree()
        # self.db_connection.execute('drop table {}_criterion'.format(self.table_train), self.engine)

    def __binning(self):
        return 1

    def __initial_nodes(self):
        tree = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut = [executor.submit(self.__grow_tree, "where xq_Elevation = {}".format(i + 1)) for i in range(1000)]
            for r in concurrent.futures.as_completed(fut):
                tree.append(r.result())
        return tree

    def __grow_tree(self, query=''):
        """Build a decision tree by recursively finding the best split with the info table and the resulting mutual
        information table. """
        # if not query == '':
        #     self.db_connection.execute(
        #         'create or replace view {}_criterion as select * from {} {}'.format(self.table_train, self.table_train,
        #                                                                             query)
        #         , self.engine)
        # else:
        #     self.db_connection.execute(
        #         "create or replace view {}_criterion as select * from {}".format(self.table_train, self.table_train)
        #         , self.engine)

        self.__create_cc_table(query)
        cc_table = self.__get_cc_table()
        mutual_inf = self.__calc_mutual_information(cc_table)
        num_samples_per_class = "left {}, right {}".format(
            mutual_inf[(mutual_inf.filter(regex='^(ny)\d*').columns)].iloc[0].max(),
            mutual_inf[(mutual_inf.filter(regex='^(alt_ny)\d*').columns)].iloc[0].sum())
        predicted_class = int(mutual_inf[mutual_inf.filter(regex='^(ny)\d*').columns].head(1).idxmax(axis=1).values[0][2:])
        node = Node(
            mutual_inf=float(mutual_inf.mi.values[0]),
            num_samples=int(mutual_inf.n__.values[0]),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # If there is no mutual information the node can be returned
        if mutual_inf.empty:
            return node
        elif (self.max_mutual_inf != 0) & (mutual_inf.mi.values[0] < self.max_mutual_inf):
            return node
        # if there is only one class left, there is no point in splitting further
        elif len(cc_table.y.drop_duplicates().values) <= 1:
            return node
        # Split recursively until maximum depth is reached.
        elif self.max_samples <= int(mutual_inf.n__.values[0]):
            if query == '':
                if mutual_inf.f.values[0] not in self.catFeatures:
                    X_left = ' where {} <= {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = ' where {} > {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                else:
                    X_left = ' where {} = {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = ' where {} <> {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                self.table_indices[1] = mutual_inf.f.values[0]
                self.db_connection.execute(
                    Template(sql.tmplt['_addIndex']).render(input=self, idx=self.table_indices[1], enumidx=1),
                    self.engine
                )
            else:
                if mutual_inf.f.values[0] not in self.catFeatures:
                    X_left = query + ' and {} <= {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = query + ' and {} > {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                else:
                    X_left = query + ' and {} = {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = query + ' and {} <> {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                if mutual_inf.f.values[0] not in list(self.table_indices.values())[-1]:
                    # increment last key by 1 and concatenate this feature to last indexes to make a clustered index
                    self.table_indices[list(self.table_indices.keys())[-1] + 1] = \
                        list(self.table_indices.values())[-1] + ', {}'.format(mutual_inf.f.values[0])
                    if len(self.table_indices.keys()) < 64:
                        self.db_connection.execute(
                            Template(sql.tmplt['_addIndex']).render(
                                input=self, idx=self.table_indices[list(self.table_indices.keys())[-1]],
                                enumidx=len(self.table_indices.keys()) + 1), self.engine
                        )
            node.feature = mutual_inf.f.values[0]
            node.threshold = mutual_inf.x.values[0]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                node.left = executor.submit(self.__grow_tree, X_left).result()
                node.right = executor.submit(self.__grow_tree, X_right).result()
        return node

    def visualize_tree(self, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(self.numFeatures + self.catFeatures,
                         [self.target + " {}".format(i) for i in self.target_classes],
                         show_details)

    def create_model(self):
        self.model.append('CASE\n')
        self.__model_aux(self.tree_)
        self.model.append('\nELSE 0 \nEND')

    def __model_aux(self, node, path='1=1'):
        is_leaf = not node.right
        if is_leaf:
            self.model.append('WHEN {} THEN {}'.format(path, str(node.predicted_class)))
        # If not a leaf, must have two children.
        else:
            feature = node.feature
            threshold = node.threshold
            if feature not in self.catFeatures:
                self.__model_aux(node.left, '{} AND {} <= {}'.format(path, feature, threshold))
                self.__model_aux(node.right, '{} AND {} > {}'.format(path, feature, threshold))
            else:
                self.__model_aux(node.left, '{} AND {} = {}'.format(path, feature, threshold))
                self.__model_aux(node.right, '{} AND {} <> {}'.format(path, feature, threshold))

    def predict(self, X=None):
        if X is None:
            X = self.db_connection.execute_query("SELECT * FROM {}".format(self.table_eval), self.engine, as_df=True)
        pred = []
        for r in X.itertuples():
            pred.append(self.__predict(r))
        X['prediction'] = pred
        return X

    def __predict(self, r):
        node = self.tree_
        while node.left:
            if node.feature not in self.catFeatures:
                if (getattr(r, node.feature)) <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                if (getattr(r, node.feature)) == node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.predicted_class

    def predict_table(self, stored_procedure=False):
        self.create_model()
        print('Adding column Prediction to table {}'.format(self.table_eval))
        self.db_connection.execute(
            Template(sql.tmplt['_predictEval']).render(input=self),
            self.engine
        )
        if stored_procedure:
            if 'sqlite' in self.engine.name:
                print("sqlite does not support stored procedures")
            else:
                print('Creating Prediction as stored procedure')
                self.db_connection.execute("DROP PROCEDURE IF EXISTS predict_{};".format(self.dataset), self.engine)
                self.db_connection.execute(
                    Template(sql.tmplt['_predictionProcedure']).render(
                        input=self,
                        model=self.__model_procedure(self.tree_)),
                    self.engine
                )

    def score(self):
        numerator = self.db_connection.execute_query(
            "select count(*) from {} where {} = Prediction;".format(
                self.table_eval, self.target),
            self.engine, True)
        denominator = self.db_connection.execute_query(
            "select count(*) from {};".format(
                self.table_eval),
            self.engine, True)
        return numerator.values[0] / denominator.values[0]

    def __model_procedure(self, node):
        self.stmtprocedure = []
        if 'mysql' in self.engine.name:
            self.__model_procedure_mysql(node)
        elif 'mssql' in self.engine.name:
            self.__model_procedure_mssql(node)
        return self.stmtprocedure

    def __model_procedure_mysql(self, node, path='1=1'):
        is_leaf = not node.right
        if is_leaf:
            self.stmtprocedure.append('IF {} THEN SET predictedClass = {}'.format(path, str(node.predicted_class)))
        else:
            feature = node.feature
            threshold = node.threshold
            if feature not in self.catFeatures:
                self.__model_procedure_mysql(node.left, '{} AND {} <= {}'.format(path, feature, threshold))
                self.__model_procedure_mysql(node.right, '{} AND {} > {}'.format(path, feature, threshold))
            else:
                self.__model_procedure_mysql(node.left, '{} AND {} = {}'.format(path, feature, threshold))
                self.__model_procedure_mysql(node.right, '{} AND {} <> {}'.format(path, feature, threshold))

    def __model_procedure_mssql(self, node, path='1=1'):
        is_leaf = not node.right
        if is_leaf:
            self.stmtprocedure.append('IF {} \nBEGIN SET @pred = {} END'.format(path, str(node.predicted_class)))
        else:
            feature = node.feature
            threshold = node.threshold
            if feature not in self.catFeatures:
                self.__model_procedure_mssql(node.left, '{} AND @{} <= {}'.format(path, feature, threshold))
                self.__model_procedure_mssql(node.right, '{} AND @{} > {}'.format(path, feature, threshold))
            else:
                self.__model_procedure_mssql(node.left, '{} AND @{} = {}'.format(path, feature, threshold))
                self.__model_procedure_mssql(node.right, '{} AND @{} <> {}'.format(path, feature, threshold))
