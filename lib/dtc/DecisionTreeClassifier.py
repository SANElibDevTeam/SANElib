from util.database_connection import Database
from lib.dtc import sqlTemplates as sql
from lib.dtc.tree import Node
from jinja2 import Template
import pandas as pd
import numpy as np


class DecisionTreeClassifier:
    def __init__(self, db, dataset='', target='', table_train='', table_eval=''):
        self.db_connection = Database(db)
        self.engine = self.db_connection.engine
        self.dataset = dataset
        self.target = target
        self.max_samples = 1000
        if table_train == '':
            self.table_train = self.dataset + '_train'
        else:
            self.table_train = table_train
        if table_eval == '':
            self.table_eval = self.dataset + '_eval'
        else:
            self.table_eval = table_eval

    def train_test_split(self, ratio=0.8, seed=1):
        self.ratio = ratio
        self.seed = seed

        self.db_connection.materializedView(
            'Splitting table into training set',
            self.table_train,
            Template(sql.tmplt['_train']).render(input=self), self.engine)

        self.db_connection.materializedView(
            'Splitting table into evaluation set',
            self.table_eval,
            Template(sql.tmplt['_eval']).render(input=self), self.engine)

    def __create_table_info(self, numFeatures, catFeatures):
        self.numFeatures = numFeatures
        self.catFeatures = catFeatures
        self.db_connection.materializedView(
            'Computing information table with target',
            self.dataset + '_info',
            Template(sql.tmplt['_info']).render(input=self),
            self.engine)

    def __get_table_info(self):
        info = self.db_connection.execute_query(
            query='select * from {}_info'.format(self.dataset),
            engine=self.engine, as_df=True)

        return info

    def __calc_mutual_information(self, info):
        # Creating nx_,ndy_gx,min_nxy
        info = info.join(info.groupby(['f', 'x']).nxy.agg(nx_='sum', ndy_gx='count', min_nxy='min'), on=['f', 'x'])

        # Creating n__
        info = info.join(info.groupby(['f']).nxy.agg(n__='sum'), on=['f'])

        # Creating n_y
        info = info.join(info.groupby(['f', 'y']).nxy.agg(n_y='sum'), on=['f', 'y'])

        # Creating ndygx - which is a division from the nx_ and ndy_gx columns
        # This is done for faster processing as opposed to a function
        info['ndygx'] = info.nx_ / info.ndy_gx

        # Calculating the cum_nx_ column
        info = info.merge( \
            info.groupby(['f', 'x']).ndygx.agg('sum').groupby(level=0).agg(cum_nx_='cumsum').round().reset_index() \
            , on=['f', 'x']).sort_values(by=['f', 'y', 'x'])

        # Calculating the cum_nxy column
        info['cum_nxy'] = info.sort_values(by=['x']).groupby(['f', 'y']).nxy.transform(np.cumsum)

        # Saving the sorted dataframe
        info = info.sort_values(by=['f', 'y', 'x'])

        # Calculating the probabiliy columns and the unsummized mutual information
        info['p_xy'] = info['cum_nxy'] / info['n__']
        info['p_y'] = info['n_y'] / info['n__']
        info['p_x'] = info['cum_nx_'] / info['n__']
        info['mi'] = info['p_xy'] * np.log2(info['p_xy'] / (info['p_y'] * info['p_x']))

        # Creating the mutual_information dataframe
        mutual_inf = info.groupby(['f', 'x'], as_index=False).agg({"mi": "sum", "cum_nx_": "min", "n__": "min"})

        # Creating the mx column, which is the max value of mi
        mutual_inf = mutual_inf.groupby(['f']).mi.agg(mx='max').merge(mutual_inf, on=['f'])

        # Creating the alt column, which is how many items are NOT in this group
        mutual_inf['alt'] = mutual_inf['n__'] - mutual_inf['cum_nx_']

        # Overwriting mutual_inf with only the neccessary information where the max value of mi is
        mutual_inf = \
            mutual_inf.loc[mutual_inf.mi == mutual_inf.mx][['f', 'x', 'mi', 'cum_nx_', 'alt', 'n__']].sort_values(
                by=['mi'], ascending=False)
        mutual_inf = mutual_inf.loc[mutual_inf.mi == mutual_inf.mi.agg('max')]

        targets = [1, 2, 3, 4, 5, 6, 7]
        for i in targets:
            mutual_inf['ny' + str(i)] = info.loc[
                (info.y == i) & (info.f == mutual_inf.f.values[0]) & (info.x <= mutual_inf.x.values[0])].nxy.sum()
        for i in targets:
            mutual_inf['alt_ny' + str(i)] = info.loc[
                (info.y == i) & (info.f == mutual_inf.f.values[0]) & (info.x > mutual_inf.x.values[0])].nxy.sum()
        return mutual_inf

    def estimate(self, numFeatures, catFeatures):
        """Build decision tree classifier."""
        self.n_classes_ = len([1, 2, 3, 4, 5, 6, 7])  # classes are assumed to go from 0 to n-1
        self.numFeatures = numFeatures
        self.catFeatures = catFeatures
        self.tree_ = self.__grow_tree('')

    def __grow_tree(self, query=''):
        """Build a decision tree by recursively finding the best split with the info table and the resulting mutual information table."""
        if not query == '':
            self.db_connection.materializedView(
                'Creating the train table with criteria',
                self.table_train + '_criterion',
                'select * from {} {}'.format(self.table_train, query)
                , self.engine)
        else:
            self.db_connection.materializedView(
                'Creating initial criteria table from train table',
                self.table_train + '_criterion',
                'select * from {}'.format(self.table_train)
                , self.engine)

        self.__create_table_info(self.numFeatures, self.catFeatures)
        info = self.__get_table_info()
        mutual_inf = self.__calc_mutual_information(info)
        num_samples_per_class = mutual_inf.iloc[0, 6:13].max()
        predicted_class = int(mutual_inf.iloc[0:1, 6:13].idxmax(axis=1).values[0][2])
        node = Node(
            mutual_inf=float(mutual_inf.mi.values[0]),
            num_samples=int(mutual_inf.cum_nx_.values[0]),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if mutual_inf.empty:
            return node

        # Split recursively until maximum depth is reached.
        if self.max_samples <= int(mutual_inf.n__.values[0]):
            if query == '':
                X_left = ' where {} <= {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                X_right = ' where {} > {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
            else:
                X_left = query + ' and {} <= {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                X_right = query + ' and {} > {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
            node.feature = mutual_inf.f.values[0]
            node.threshold = mutual_inf.x.values[0]
            node.left = self.__grow_tree(X_left)
            node.right = self.__grow_tree(X_right)
        return node

    def visualize_tree(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    def predict(self, X):
        pred = []
        for r in X.itertuples():
            pred.append(self.__predict(r))
        return pred

    def __predict(self, r):
        node = self.tree_
        while node.left:
            if (getattr(r, node.feature)) <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
