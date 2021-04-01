from jinja2 import Template
from lib.mdh import sql_templates as sql
from plotnine import ggplot, aes, geom_line, geom_point, geom_col, labs, theme, element_text, theme_bw, facet_wrap
import pandas as pd
from util.database_connection import Database


class MDH:
    def __init__(self, db):
        self.db_connectionn = Database(db)
        self.engine = self.db_connectionn.engine
        self.dataset = "table_train"
        self.model_id = 'covtyptest2'
        self.seed = 1
        self.ratio = 0.8
        self.train = self.dataset
        self.eval = self.train

        # Model
        self.analysis = self
        self.model_id = self.model_id

    def initialize(self):
        if self.ratio != 1.0:
            self.train_test_split()
            self.train = self.model_id + '_train'
            self.eval = self.model_id + '_table_eval'
            # if self.target is None:
            #     col = self.executeQuery('If last column (DESC), else first column (ASC)', '''
            #         SELECT
            #         COLUMN_NAME
            #         ORDINAL_POSITION
            #         FROM INFORMATION_SCHEMA.COLUMNS
            #         WHERE TABLE_NAME = '{}'  #table_train
            #         ORDER BY ORDINAL_POSITION DESC
            #         LIMIT 1;
            #         '''.format(self.dataset))
            #
            #     # ".fetchall" returns a list of RowProxys (SQLAlchemy) so this
            #     # for loop manually parses for the last column in table
            #     for index, element in enumerate(col):
            #         element = tuple(element)
            #         self.target = element[index]
            #         print(self.target)
            # else:
            self.target = 'Cover_Type'

    def get_cat_feat(self):

        return self.catFeatures

    def get_num_feat(self):
        return self.numFeatures

    # TODO: Write get drop func
    def drop(self):
        pass

    def train_test_split(self):
        """
        Splitting table into training set and evaluation set
        Update eval
        """

        self.db_connectionn.materializedView(
            'Splitting table into training set',
            self.model_id + '_train',
            Template(sql.tmplt['_train']).render(input=self), self.engine)

        self.db_connectionn.materializedView(
            'Splitting table into test set',
            self.model_id + '_table_eval',
            Template(sql.tmplt['_table_eval']).render(input=self), self.engine)

    def rank(self, table_train, catFeatures, numFeatures, bins):
        self.numFeatures = numFeatures
        self.bins = bins
        self.catFeatures = catFeatures
        self.db_connectionn.materializedView(
            'Computing 1d contingecies with target',
            self.model_id + '_m1d',
            Template(sql.tmplt['_m1d']).render(input=self), self.engine)
        results = self.db_connectionn.execute_query(Template(sql.tmplt["_m1d_mi"]).render(input=self), self.engine)
        df = pd.DataFrame(results)
        df.columns = ['f', 'mi']
        print(df)
        return self

    def estimate(self, catFeatures, bins=50, numFeatures=None):
        """
        1.) Need to be feeding the train set (ratio) of original table into this --> training phase
        2.) Then, invoking the predict method on the test set (1-ratio) of the original table --> prediction phase on new data

        This function is the training phase:
        - input data is training set
        - the input data table is quantized (equal size) and indexed.
        - This quantized index then represents an in-database model for probability estimation

        This function sets the hyperparameters
        - features to estimate the probability of the target
        - features = right now it is just a string, but I think we may want to explore
                     having the user put the feature(s) into a numpy array. Similiar to
                     how the scikit-learn ML algorithms want them. (format:str)
        - Bins/buckets = # of bins
        """

        if numFeatures is None:
            allColumns = self.db_connectionn.execute_query('''
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{}';  #table_train
            '''.format(self.train), self.engine)

            self.numFeatures = [element for index, tuple in enumerate(allColumns) for index, element in
                                enumerate(tuple)]
        else:
            self.numFeatures = numFeatures

        self.bins = bins
        self.catFeatures = catFeatures

        # TODO Generate queries using n features x1, x2, ..., xn; differentiate between numerical and categorical

        self.db_connectionn.materializedView(
            'Quantization of training table',
            self.model_id + '_qt',
            Template(sql.tmplt['_qt']).render(input=self), self.engine)
        self.db_connectionn.materializedView(
            'Quantization metadata for training table',
            self.model_id + '_qmt',
            Template(sql.tmplt['_qmt']).render(input=self), self.engine)
        self.db_connectionn.materializedView(
            'Computing predictive model as contingency table',
            self.model_id + '_m',
            Template(sql.tmplt['_m']).render(input=self), self.engine)

        return self

    # Model

    def visualize1D(self, feature1, target):

        if feature1 in self.numFeatures:
            index = self.numFeatures.index(feature1) + 1

            for i in range(1, len(self.numFeatures) + 1):
                if index == i:
                    feature1 = 'xn{}'.format(i)
                    bins = 'xq{}'.format(i)
                    minimum = 'mn{}'.format(i)
                    maximum = 'mx_{}'.format(i)

            num = self.db_connectionn.execute_query('''
                        select distinct {} as xq, {} as mn, {} as mx, 
                        ({}+{})/2 as x_,
                        concat({}, ': ]', {}, ',', {}, ']') as bin, 
                        cast(y_ as char) as {},
                        sum(nxy)over(partition by {}, y_)*1.0/
                        sum(nxy) over()  as p 
                        from {}_m 
                        order by {}, cast(y_ as char);'''.format(bins, minimum, maximum,
                                                                 maximum, minimum,
                                                                 bins, minimum, maximum,
                                                                 target,
                                                                 bins,
                                                                 self.model_id,
                                                                 bins), self.analysis.engine)

            hist_df = pd.DataFrame(num)
            columns = ['xq', 'mn', 'mx', 'x_', 'bin', 'p']
            columns.insert(6, target)
            hist_df.columns = columns
            hist_df[['p']] = hist_df[['p']].apply(pd.to_numeric)
            hist_df[['x_']] = hist_df[['x_']].apply(pd.to_numeric)

            ylabel = 'p(Q({} | {}))'.format(feature1, target)

            p = (
                    ggplot(hist_df)
                    + aes('x_', 'p', color=target, group=target)
                    + geom_point()
                    + geom_line()
                    + labs(y=ylabel, x='bin', title='1d Histogram Probability Estimation')
                    + theme(axis_text_x=element_text(rotation=90, hjust=1))
            )

            print(p)

        elif feature1 in self.catFeatures:
            index = self.catFeatures.index(feature1) + 1

            for i in range(1, len(self.catFeatures) + 1):
                if index == i:
                    feature1 = 'xc{}'.format(i)
                    bins = 'xq{}'.format(i)

            # Categorical
            cat = self.db_connectionn.execute_query('''
                        select distinct {} as xc, 
                        cast(y_ as char) as {},
                        sum(nxy)over(partition by {}, y_)*1.0/
                        sum(nxy) over()  as p 
                        from {}_m 
                        order by {}, cast(y_ as char);'''.format(feature1, target, bins,
                                                                 self.model_id, feature1), self.analysis.engine)

            res_df = pd.DataFrame(cat)
            columns = ['xc', 'p']
            columns.insert(1, target)
            res_df.columns = columns
            res_df[['p']] = res_df[['p']].apply(pd.to_numeric)

            ylabel = 'p({}, {})'.format(feature1, target)

            p = (ggplot(res_df, aes('xc', 'p', fill=target))
                 + theme_bw()
                 + geom_col(position='dodge')
                 + labs(y=ylabel, x=feature1)
                 )

            print(p)

        else:
            raise ValueError('Feature variable {} does not exist'.format(feature1))
        return self

    def visualize2D(self, numFeat, catFeat, target):

        if numFeat in self.numFeatures and catFeat in self.catFeatures:

            index = self.catFeatures.index(catFeat) + 1
            for i in range(1, len(self.catFeatures) + 1):
                if index == i:
                    feature1 = 'xc{}'.format(i)
                    bins = 'xq{}'.format(i)
                    minimum = 'mn{}'.format(i)
                    maximum = 'mx_{}'.format(i)

            multi = self.db_connectionn.execute_query('''
                        select distinct {} as xq, {} as mn, {} as mx, 
                        ({}+{})/2 as x_,
                        concat({}, ': ]', {}, ',', {}, ']') as bin,
                         {} as xc,  
                        cast(y_ as char) as {},
                        sum(nxy)over(partition by {}, {}, y_)*1.0/
                        sum(nxy) over()  as p 
                        from {}_m 
                        order by {}, xc, cast(y_ as char);'''.format(bins, minimum, maximum,
                                                                     maximum, minimum,
                                                                     bins, minimum, maximum,
                                                                     feature1,
                                                                     target,
                                                                     feature1, bins,
                                                                     self.model_id,
                                                                     bins), self.analysis.engine)

            dim2 = pd.DataFrame(multi)
            columns = ['xq', 'mn', 'mx', 'x_', 'bin', 'xc', 'p']
            columns.insert(6, target)
            dim2.columns = columns

            dim2[['p']] = dim2[['p']].apply(pd.to_numeric)
            dim2[['x_']] = dim2[['x_']].apply(pd.to_numeric)

            ylabel = 'p(Q({}) x {}, {})'.format(numFeat, catFeat, target)

            p = (
                    ggplot(dim2)
                    + aes('x_', 'p', color=target, group=target)
                    + geom_point()
                    + geom_line()
                    + facet_wrap('xc')
                    + labs(y=ylabel, x=numFeat)
            )

            print(p)
        return self

    def predict(self):
        return self

    # Prediction

    def predict(self, eval):  # table_eval is the test set
        """
        This function estimates the probabilities for the evaluation data
        """
        self.eval = eval
        self.db_connectionn.materializedView(
            'Quantization metadata for evaluation table',
            self.model_id + '_qe',
            Template(sql.tmplt['_qe']).render(input=self), self.engine)  ## generate SQL using Jinja 2 template
        self.db_connectionn.execute('Creating index _qe ',
                                    Template(sql.tmplt['_qe_ix']).render(input=self), self.engine)
        self.db_connectionn.materializedView(
            'Class prediction for evaluation dataset',
            self.model_id + '_p',
            Template(sql.tmplt['_p']).render(input=self), self.engine)  ## generate SQL using Jinja 2 template
        self.db_connectionn.execute('Updating prediction with default prediction for null predictions',
                                    Template(sql.tmplt['_p_update']).render(input=self), self.engine)
        return self

    def accuracy(self):
        """
        Computing the accuracy of the model and returning the results to the user
        """

        results = self.db_connectionn.execute_query('''
            select
            count(distinct id) as  cases,
            sum(case when e.y = p.y_ then 1 else 0 end) as tp,
            sum(case when e.y = p.y_ then 1 else 0 end) /  count(distinct id)  as accuracy
            from {}_qe e 
            left outer join {}_p p using(id);
                '''.format(
            self.model_id,
            self.model_id), self.engine)

        df = pd.DataFrame(results)
        df.columns = ['Total', 'TP', 'Accuracy']
        print(df)
