
import sqlTemplates as sql
from jinja2 import Template
import pandas as pd

from plotnine import ggplot, aes, geom_line, geom_point, scale_x_continuous, \
    scale_x_discrete, geom_col, theme_light, labs, theme, element_text, theme_bw, facet_wrap

from Database import SaneDataBase



class SaneProbabilityEstimator:

    def __init__(self, conn, base_table, target=None, model_id=None, bins=50):
        """
        This method takes the most commonly used parameters from the user during initialization of the classifier
        - conn = database connection (format: variable)
        - target = target variable (what you are wanting to predict) (format: str)
        - table_name = name of the table that is in the database (format: str)
        """

        self.db = SaneDataBase(conn)

        self.base_table = base_table
        self.table_train = base_table
        self.table_eval = base_table # default: eval on training data with full dataset
        if model_id is None:
            self.model_id = base_table;
        else:
            self.model_id = model_id
        self.bins = bins

        self.all_col_types = self.db.get_all_col_types(self.base_table)
        self.all_numFeatures= self.db.get_num_cols(self.base_table)
        self.all_catFeatures = self.db.get_cat_cols(self.base_table)
        self.numFeatures=self.all_numFeatures
        self.catFeatures=self.all_catFeatures
#TODO get last column
        if target is None:
            col = self.executeQuery('If last column (DESC), else first column (ASC)', '''
                SELECT
                COLUMN_NAME
                ORDINAL_POSITION
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{}'  #table_train
                ORDER BY ORDINAL_POSITION DESC
                LIMIT 1;
                '''.format(table_train))

            # ".fetchall" returns a list of RowProxys (SQLAlchemy) so this
            # for loop manually parses for the last column in table
            for index, element in enumerate(col):
                element = tuple(element)
                self.target = element[index]
                print(self.target)
        else:
            self.target = target



    def train_test_split(self, seed, ratio, train_table_name=None,  eval_table_name=None):
        """
        Splitting table into training set and evaluation set
        sets the train and eval tables automatically using model id or according to param
        :return: table_eval
        """

        if train_table_name is None:
            train_table_name =  self.model_id + '_train'
        if eval_table_name is None:
            eval_table_name = self.model_id + '_eval'

        self.table_train = train_table_name
        self.table_eval = eval_table_name

        self.seed = seed
        self.ratio = ratio

        self.db.createView(
             'Create table train',
             train_table_name,
             Template(sql.tmplt['_train']).render(input=self))

        self.table_train = self.model_id + '_train'

        self.db.createView(
             'Create table eval',
             eval_table_name,
             Template(sql.tmplt['_eval']).render(input=self))

        self.table_eval= self.model_id + '_eval'


    def rank(self, table_train=None, catFeatures=None, numFeatures=None):
        if table_train is None:
            table_train = self.table_train
        if catFeatures is None:
            catFeatures = self.all_catFeatures
        self.catFeatures = catFeatures
        if numFeatures is None:
            numFeatures = self.all_numFeatures
        self.numFeatures = numFeatures
        test = Template(sql.tmplt['_m1d']).render(input=self)
        self.db.createView(
            'Computing 1d contingecies with target',
            self.model_id + '_m1d',
            Template(sql.tmplt['_m1d']).render(input=self))
        results = self.db.executeQuery('Computing mutual information with target',
                                       Template(sql.tmplt["_m1d_mi"]).render(input=self))
        df = pd.DataFrame(results)
        df.columns = ['f', 'mi']
        print(df)
        self.ranked_columns = df
        return df

#TODO get cat features & num features automatically and set as default
#TODO set dictionnary as param, or set this with object instantiation
    def train(self, table_train=None, catFeatures = None, numFeatures = None):
        """

        """
        if table_train is None:
            table_train = self.table_train
        if catFeatures is None:
            catFeatures = self.all_catFeatures
        self.catFeatures = catFeatures
        if numFeatures is None:
            numFeatures = self.all_numFeatures
        self.numFeatures = numFeatures

        self.db.createView(
            'Get aggregation values',
            self.model_id + '_agg',
            Template(sql.tmplt['_agg']).render(input=self))
        self.db.createView(
            'Quantization of training table',
            self.model_id + '_qt',
            Template(sql.tmplt['_qt']).render(input=self))

    def predict(self, table_eval=None):  # table_eval is the test set
        """
        This function estimates the probabilities for the evaluation data
        """
        if not(table_eval is None):
            self.table_eval = table_eval
        self.db.createView(
            'Quantization metadata for evaluation table',
            self.model_id + '_qe',
            Template(sql.tmplt['_qe']).render(input=self)) ## generate SQL using Jinja 2 template
        self.db.createView(
            'Class prediction for evaluation dataset',
            self.model_id + '_p',
            Template(sql.tmplt['_p']).render(input=self))  ## generate SQL using Jinja 2 template
        self.db.execute('Updating prediction with default prediction for null predictions',
            Template(sql.tmplt['_p_update']).render(input=self))

    def accuracy(self):
        """
        Computing the accuracy of the model and returning the results to the user
        """
        results = self.db.executeQuery('Computing evaluation accuracy', '''
            select
            count(distinct id) as  cases,
            sum(case when e.y = p.y_ then 1 else 0 end) as tp,
            sum(case when e.y = p.y_ then 1 else 0 end) /  count(distinct id)  as accuracy
            from {}_qe e 
            left outer join {}_p p using(id);
                '''.format(
                    self.model_id,
                    self.model_id))

        df = pd.DataFrame(results)
        df.columns = ['Total', 'TP', 'Accuracy']
        print(df)

    def gauss_cdf_approx(self):



#######

        # todo: visualize num, cat, num + cat, cat+cat, cat+num, then get data type and decide which function based on input
        def visualize1D(self, feature1, target):

            if feature1 in self.numFeatures:
                index = self.numFeatures.index(feature1) + 1

                for i in range(1, len(self.numFeatures) + 1):
                    if index == i:
                        feature1 = 'xn{}'.format(i)
                        bins = 'xq{}'.format(i)
                        minimum = 'mn{}'.format(i)
                        maximum = 'mx_{}'.format(i)

                num = self.db.executeQuery('Discretization 1d Numerical Histogram', '''
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
                                                                         bins))

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
                cat = self.db.executeQuery('Discretization 1d Categorical Histogram', '''
                                select distinct {} as xc, 
                                cast(y_ as char) as {},
                                sum(nxy)over(partition by {}, y_)*1.0/
                                sum(nxy) over()  as p 
                                from {}_m 
                                order by {}, cast(y_ as char);'''.format(feature1, target, bins,
                                                                         self.model_id, feature1))

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

        def visualize2D(self, numFeat, catFeat, target):

            if numFeat in self.numFeatures and catFeat in self.catFeatures:

                index = self.catFeatures.index(catFeat) + 1
                for i in range(1, len(self.catFeatures) + 1):
                    if index == i:
                        feature1 = 'xc{}'.format(i)
                        bins = 'xq{}'.format(i)
                        minimum = 'mn{}'.format(i)
                        maximum = 'mx_{}'.format(i)

                multi = self.db.executeQuery('2d Discretization Histogram Estimation', '''
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
                                                                             bins))

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


