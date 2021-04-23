from util.database_connection import Database
from lib.dtc import sqlTemplates as sql
from lib.dtc.tree import Node
from jinja2 import Template
import pandas as pd
import numpy as np
import concurrent.futures


class DecisionTreeClassifier:
    def __init__(self, db, dataset='', target='', target_classes=None, max_samples=2, table_train='', table_eval=''):
        self.db_connection = Database(db)
        self.engine = self.db_connection.engine
        self.dataset = dataset
        self.max_samples = max_samples

        if target == '':
            colnames = self.db_connection.execute_query("SELECT * FROM {} LIMIT 0".format(self.dataset), self.engine,
                                                        True)
            self.target = colnames.columns[-1]

        if target_classes is None:
            target_classes = self.db_connection.execute_query(
                "SELECT DISTINCT {} FROM {} ORDER BY {} ASC".format(self.target, self.dataset, self.target),
                self.engine, True)
        self.target_classes = target_classes.to_numpy().flatten()

        if table_train == '':
            self.table_train = self.dataset + '_train'
        else:
            self.table_train = table_train
        if table_eval == '':
            self.table_eval = self.dataset + '_eval'
        else:
            self.table_eval = table_eval

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
        # Get first row of dataset to see the which features are numerical, which are not
        features = self.db_connection.execute_query("select * from {} limit 1".format(self.dataset), self.engine, True)
        # Select only the non numerical features and putting them into a dataframe
        catFeature = features.select_dtypes(include='object').columns

        # Add id column to dataset, needed for grouping for ordinal encoding
        self.db_connection.execute(
            "ALTER TABLE {} ADD column `rownum` INT NOT NULL AUTO_INCREMENT unique first;".format(self.dataset),
            self.engine)

        # Get all characteristics of each categorical feature
        features = {}
        for f in catFeature:
            features[f] = np.array(
                self.db_connection.execute_query("select distinct {} from {}".format(f, self.dataset), self.engine))

        self.catFeatures = features

        # rename categorical columns since the reformatted will have the same name and the originals will be dropped
        # later
        for key in self.catFeatures:
            self.db_connection.execute(
                "alter table {} rename column {} to {}_orig".format(self.dataset, key, key), self.engine)

        self.db_connection.materializedView(
            'Creating categorical feature columns as ordinal values and splitting into training set',
            self.table_train,
            Template(sql.tmplt['_catFeatOrdinalTrain']).render(input=self), self.engine)

        self.db_connection.materializedView(
            'Creating categorical feature columns as ordinal values and splitting into evaluation set',
            self.table_eval,
            Template(sql.tmplt['_catFeatOrdinalEval']).render(input=self), self.engine)

        for key in self.catFeatures:
            self.db_connection.execute(
                "alter table {} drop column {}_orig".format(self.table_train, key), self.engine)
            self.db_connection.execute(
                "alter table {} drop column {}_orig".format(self.table_eval, key), self.engine)

        self.db_connection.execute(
            "alter table {} drop column rownum".format(self.dataset), self.engine)
        self.db_connection.execute(
            "alter table {} drop column rownum".format(self.table_train), self.engine)
        self.db_connection.execute(
            "alter table {} drop column rownum".format(self.table_eval), self.engine)

        for key in self.catFeatures:
            self.db_connection.execute(
                "alter table {} rename column {}_orig to {}".format(self.dataset, key, key), self.engine)

        self.catFeatures = self.catFeatures.keys()

    def __create_cc_table(self):
        self.db_connection.materializedView(
            'Computing information table with target',
            self.dataset + '_CC_table',
            Template(sql.tmplt['_CC_table']).render(input=self),
            self.engine)

    def __get_cc_table(self):
        cc_table = self.db_connection.execute_query(
            query='select * from {}_CC_table'.format(self.dataset),
            engine=self.engine, as_df=True)

        return cc_table

    def __calc_mutual_information(self, cc_table):
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
        return mutual_inf

    def estimate(self, numFeatures, catFeatures):
        """Build decision tree classifier."""
        self.n_classes_ = len(self.target_classes)  # classes are assumed to go from 0 to n-1
        self.numFeatures = numFeatures
        self.catFeatures = catFeatures
        self.tree_ = self.__grow_tree('')
        self.db_connection.execute('drop table {}_criterion'.format(self.table_train), self.engine)

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

        self.__create_cc_table()
        cc_table = self.__get_cc_table()
        mutual_inf = self.__calc_mutual_information(cc_table)
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
        elif mutual_inf.mi.values[0] < 0.01:
            return node

        # Split recursively until maximum depth is reached.
        if self.max_samples <= int(mutual_inf.n__.values[0]):
            if query == '':
                if mutual_inf.f.values[0] not in self.catFeatures:
                    X_left = ' where {} <= {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = ' where {} > {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                else:
                    X_left = ' where {} = {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = ' where {} <> {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                self.db_connection.execute(
                    "alter table {} add index ({})".format(self.table_train, mutual_inf.f.values[0]), self.engine
                )
            else:
                if mutual_inf.f.values[0] not in self.catFeatures:
                    X_left = query + ' and {} <= {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = query + ' and {} > {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                else:
                    X_left = query + ' and {} = {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                    X_right = query + ' and {} <> {} '.format(mutual_inf.f.values[0], mutual_inf.x.values[0])
                indexes = self.db_connection.execute_query(
                    "SHOW indexes from {}".format(self.table_train), self.engine, True)
                if len(indexes.index) < 64:
                    prev_index = \
                        indexes.groupby(['Key_name'])['Column_name'].apply(','.join).reset_index().iloc[-1:]
                    if mutual_inf.f.values[0] not in prev_index['Column_name'].values[0]:
                        self.db_connection.execute(
                            "alter table {} add index ({})".format(self.table_train,
                                                                   prev_index['Column_name'].values[0] + ',' +
                                                                   mutual_inf.f.values[0]),
                            self.engine
                        )

            node.feature = mutual_inf.f.values[0]
            node.threshold = mutual_inf.x.values[0]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                node.left = executor.submit(self.__grow_tree, X_left).result()
                node.right = executor.submit(self.__grow_tree, X_right).result()
            # node.left = self.__grow_tree(X_left)
            # node.right = self.__grow_tree(X_right)
        return node

    def visualize_tree(self, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(self.numFeatures.append(self.catFeatures),
                         [self.target + " {}".format(i - 1) for i in self.target_classes],
                         show_details)

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