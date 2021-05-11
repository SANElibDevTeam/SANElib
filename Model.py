import Analysis
from jinja2 import Template
import sqlTemplates as sql
import matplotlib.pyplot as plt
import pandas as pd
import Utils
from plotnine import ggplot, aes, geom_line, geom_point, labs, theme, element_text
from plotnine.data import economics, mpg
import plotnine as pn
import Prediction
from Utils import *
from plotnine import ggplot, aes, geom_line, geom_point, scale_x_continuous, \
    scale_x_discrete, geom_col, theme_light, labs, theme, element_text, theme_bw, facet_wrap

class Model():
    def __init__(self,analysis_obj):
        if analysis_obj != None:
            self.analysis = analysis_obj
            self.catFeatures = self.analysis.get_cat_feat()
            self.numFeatures = self.analysis.get_num_feat()
            self.model_id = self.analysis.model_id
            self.bins = self.analysis.bins
        else: raise ValueError('Analysis object needs to be passed')

    def visualize1D(self, feature1, target):

        if feature1 in self.numFeatures:
            index = self.numFeatures.index(feature1) + 1

            for i in range(1, len(self.numFeatures) + 1):
                if index == i:
                    feature1 = 'xn{}'.format(i)
                    bins = 'xq{}'.format(i)
                    minimum = 'mn{}'.format(i)
                    maximum = 'mx_{}'.format(i)

            # Numerical
            num = executeQuery('Discretization 1d Numerical Histogram', '''
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
                                                                 bins),self.analysis.engine)

            hist_df = pd.DataFrame(num)
            columns = ['xq', 'mn', 'mx', 'x_', 'bin', 'p']
            columns.insert(5, target)
            hist_df.columns = columns
            hist_df[['p']] = hist_df[['p']].apply(pd.to_numeric)
            hist_df[['x_']] = hist_df[['x_']].apply(pd.to_numeric)

            print(hist_df.head())

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
            cat = executeQuery('Discretization 1d Categorical Histogram', '''
                        select distinct {} as xc, 
                        cast(y_ as char) as {},
                        sum(nxy)over(partition by {}, y_)*1.0/
                        sum(nxy) over()  as p 
                        from {}_m 
                        order by {}, cast(y_ as char);'''.format(feature1, target, bins,
                                                                 self.model_id, feature1),self.analysis.engine)

            res_df = pd.DataFrame(cat)
            columns = ['xc', 'p']
            columns.insert(1, target)
            res_df.columns = columns
            res_df[['p']] = res_df[['p']].apply(pd.to_numeric)

            print(res_df.head())

            ylabel = 'p({}, {})'.format(feature1, target)

            p = (ggplot(res_df, aes('xc', 'p', fill=target))
                 + theme_bw()
                 + geom_col(position='dodge')
                 + labs(y=ylabel, x=feature1)
                 + theme(axis_text_x=element_text(angle=90, hjust=1))
                 )

            print(p)

        else:
            raise ValueError('Feature variable {} does not exist'.format(feature1))
        return Prediction.Prediction(self)


    def visualize2D(self, target, numFeat=None, catFeat=None):

        if numFeat != None and catFeat != None:
            if numFeat in self.numFeatures and catFeat in self.catFeatures:
                index = self.catFeatures.index(catFeat) + 1
                for i in range(1, len(self.catFeatures) + 1):
                    if index == i:
                        feature1 = 'xc{}'.format(i)
                        bins = 'xq{}'.format(i)
                        minimum = 'mn{}'.format(i)
                        maximum = 'mx_{}'.format(i)

                multi = executeQuery('2d Discretization Histogram Estimation', '''
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

                print(dim2.head())

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
                        + theme(axis_text_x=element_text(angle=90, hjust=1))
                )

                print(p)

        elif type(numFeat) is list:
            index_dict = {}
            if len(numFeat) <= 2:
                for feat in numFeat:
                    for feats in self.numFeatures:
                        if feat == feats:
                            index = self.numFeatures.index(feat) + 1
                            index_dict.update({feat: index})
            else:
                raise ValueError("Too high dimensionality to visualize")

            indexList = list(index_dict.values())

            binsDIM1 = 'xq{}'.format(indexList[0])
            minDIM1 = 'mn{}'.format(indexList[0])
            maxDIM1 = 'mx_{}'.format(indexList[0])
            binsDIM2 = 'xq{}'.format(indexList[1])
            minDIM2 = 'mn{}'.format(indexList[1])
            maxDIM2 = 'mx_{}'.format(indexList[1])

            multi = executeQuery('2d Discretization Histogram Estimation', '''
                            select distinct {} as xq, {} as mn, {} as mx, 
                            {} as xq2, {} as mn2, {} as mx2,
                            ({}+{})/2 as x_,
                            ({}+{})/2 as x_2,
                            concat({}, ': ]', {}, ',', {}, ']') as bin_1,
                            concat({}, ': ]', {}, ',', {}, ']') as bin_2,  
                            cast(y_ as char) as {},
                            sum(nxy)over(partition by {}, {}, y_)*1.0/
                            sum(nxy) over()  as p 
                            from {}_m 
                            order by {}, {}, cast(y_ as char);'''.format(binsDIM1, minDIM1, maxDIM1,
                                                                         binsDIM2, minDIM2, maxDIM2,
                                                                         maxDIM1, minDIM1,
                                                                         maxDIM2, minDIM2,
                                                                         binsDIM1, minDIM1, maxDIM1,
                                                                         binsDIM2, minDIM2, maxDIM2,
                                                                         target,
                                                                         binsDIM1, binsDIM2,
                                                                         self.model_id,
                                                                         binsDIM1, binsDIM2), self.analysis.engine)

            dim2 = pd.DataFrame(multi)
            columns = ['xq', 'mn', 'mx', 'xq2', 'mn2', 'mx2', 'x_', 'x_2', 'bin_1', 'bin_2', 'p']
            columns.insert(10, target)
            dim2.columns = columns

            print(dim2.head())

            dim2[['p']] = dim2[['p']].apply(pd.to_numeric)
            dim2[['x_']] = dim2[['x_']].apply(pd.to_numeric)
            dim2[['x_2']] = dim2[['x_2']].apply(pd.to_numeric)

            ylabel = 'p(Q({}) x {}, {})'.format(numFeat, catFeat, target)

            p = (
                    ggplot(dim2)
                    + aes('x_', 'p', color=target, group=target)
                    + geom_point()
                    + geom_line()
                    #+ facet_wrap('xc')
                    + labs(y=ylabel, x=numFeat)
                    + theme(axis_text_x=element_text(angle=90, hjust=1))
            )

            print(p)


        elif type(catFeat) is list:
            index_dict = {}
            if len(catFeat) <= 2:
                for feat in catFeat:
                    for feats in self.catFeatures:
                        if feat == feats:
                            index = self.catFeatures.index(feat) + 1
                            index_dict.update({feat: index})
            else:
                raise ValueError("Too high dimensionality to visualize")

            indexList = list(index_dict.values())

            catDIM1 = 'xc{}'.format(indexList[0])
            catDIM2 = 'xc{}'.format(indexList[1])
            bins = 'xq1'

            multi = executeQuery('Discretization 1d Categorical Histogram', '''
                        select distinct {} as xc, {} as xc2,
                        cast(y_ as char) as {},
                        sum(nxy)over(partition by {}, y_)*1.0/
                        sum(nxy) over()  as p 
                        from {}_m 
                        order by {}, {}, cast(y_ as char);'''.format(catDIM1, catDIM2,
                                                                     target, bins,
                                                                     self.model_id,
                                                                     catDIM1, catDIM2), self.analysis.engine)

            dim2 = pd.DataFrame(multi)
            columns = ['xc', 'xc2', 'p']
            columns.insert(2, target)
            dim2.columns = columns

            #print(dim2.head())

            dim2[['p']] = dim2[['p']].apply(pd.to_numeric)
            ylabel = 'p(Q({}) x {}, {})'.format(catFeat[0], catFeat[1], target)

            p = (ggplot(dim2, aes('xc2', 'p', fill=target))
                 + theme_bw()
                 + geom_col(position='dodge')
                 + facet_wrap('xc')
                 + labs(y=ylabel, x=catFeat)
                 + theme(axis_text_x=element_text(angle=90, hjust=1))
                 )

            print(p)

        return Prediction.Prediction(self)

    def predict(self):
        return Prediction.Prediction(self)
