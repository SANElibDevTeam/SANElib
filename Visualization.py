from plotnine import ggplot, aes, geom_line, geom_point, scale_x_continuous, \
    scale_x_discrete, geom_col, theme_light, labs, theme, element_text, theme_bw, facet_wrap

#todo: visualize num, cat, num + cat, cat+cat, cat+num, then get data type and decide which function based on input
def visualize1D(self, feature1, target):

    if feature1 in self.numFeatures:
        index = self.numFeatures.index(feature1) + 1

        for i in range(1, len(self.numFeatures) + 1):
            if index == i:
                feature1 = 'xn{}'.format(i)
                bins = 'xq{}'.format(i)
                minimum = 'mn{}'.format(i)
                maximum = 'mx_{}'.format(i)

        num = self.db.executeQuery('''
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
                                                             bins), 'Discretization 1d Numerical Histogram')

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
        cat = self.db.executeQuery('''
                    select distinct {} as xc, 
                    cast(y_ as char) as {},
                    sum(nxy)over(partition by {}, y_)*1.0/
                    sum(nxy) over()  as p 
                    from {}_m 
                    order by {}, cast(y_ as char);'''.format(feature1, target, bins,
                                                             self.model_id, feature1),
                                   'Discretization 1d Categorical Histogram')

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

        multi = self.db.executeQuery('''
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
                                                                 bins), '2d Discretization Histogram Estimation')

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
