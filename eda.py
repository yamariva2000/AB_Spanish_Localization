import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import seaborn as sns


mpl.rcParams['font.size'] = 20.0
pd.set_option('display.width', 1000)



areas={1:'Central America', 2:'Mexico',3:'South America',4:'Spain',5:'Argentina',6:'Uruguay'}
map={ 'Honduras': 1, 'Peru': 3, 'Uruguay': 6, 'El Salvador': 3,
     'Nicaragua': 1, 'Panama': 1, 'Mexico': 2, 'Costa Rica': 1, 'Guatemala': 1,
     'Chile': 3, 'Ecuador': 1, 'Colombia': 3, 'Paraguay': 1, 'Argentina': 5,
     'Bolivia': 1, 'Venezuela': 3, 'Spain': 4}




def combine_tables():
    '''left outer joining the test table with 3the user table'''
    test = pd.read_csv('test_table.csv')
    user = pd.read_csv('user_table.csv')
    c = pd.merge(test, user, how='left', left_on='user_id', right_on='user_id', suffixes=('_l', '_r'), indicator=True)

    c=c.query("country == country ")
    c['regions']=c['country'].apply(lambda x: areas [map [x] ] )

    return c

def model_inputs():
    '''remove unnecessary data and dummifying categorical variables'''
    
    d.drop(['age', '_merge', 'date', 'user_id'], axis=1, inplace=True)
    y = d.pop('conversion')
    dummies = pd.get_dummies(d, columns=['sex', 'ads_channel', 'source', 'device', 'browser_language', 'browser',
                                         'country'])
    x = pd.concat([dummies, d['test']], axis=1)
    return x, y


def undersample(df, c=None):
    '''undersample majority class'''
    positive = df[df[c] == 1]
    negative = df[df[c] == 0]
    minsamp = min(len(positive), len(negative))
    positive = positive.sample(n=minsamp)
    negative = negative.sample(n=minsamp)
    combine = pd.concat([positive, negative], axis=0)
    return combine


def classify():
    ''''''
    
    data = combine_tables().query("country not in ['Spain']")
    data = undersample(data, c='conversion')
    y = data.pop('conversion')

    dummies = pd.get_dummies(data, columns=['country', 'browser', 'browser_language', 'device', 'ads_channel', 'source',
                                            'sex'])
    dummies.drop(['user_id', 'date', 'age', '_merge', 'test'], inplace=True, axis=1)

    x = dummies
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)

    model = RandomForestClassifier()

    fit = model.fit(xtrain, ytrain)

    ypredict = model.predict(xtest)

    print 'confusion matrix'
    conf = confusion_matrix(ytest, ypredict)
    print conf
    f1 = f1_score(ytest, ypredict)
    print 'f1 score =', f1

    most_import_index = np.argsort(model.feature_importances_)[::-1]
    print 'feature importances'
    print x.columns
    for i in most_import_index:
        print x.columns[i]


def pivot_table(query=None,field=None,columns='test',values=None):
    '''create pivot tables for graphs and eda '''
    data = combine_tables()

    if query:
        data=data.query(query)
    #    "country not in ['Spain','Argentina','Uruguay']")
    #fields = ['country', 'sex', 'browser_language', 'browser', 'source']
    #columns = 'test'
    #values = 'conversion'

    
    total = pd.pivot_table(data, values=values, index=field, columns=columns, aggfunc="count")
    total.columns=['a_views','b_views']
    convert = pd.pivot_table(data, values=values, index=field, columns=columns, aggfunc="sum")
    convert.columns=['a_clicks','b_clicks']
    rate = pd.pivot_table(data, values=values, index=field, columns=columns, aggfunc='mean')
    rate.columns=['a_rate','b_rate']
    m = pd.concat([total, convert, rate], axis=1)
    return m

def plot_conversions():

    data=pivot_table(query = "country != 'Spain'", field=['country'],columns='test',values='conversion')
    # query = "country != 'Spain'"


    data[['a_rate','b_rate']].sort_index(ascending=False).plot.barh()
     #   .plot(label='control',fontsize=14).bar()
    #data['b_rate'].plot(label='exp',fontsize=14,)

#    ax.set_xticklabels(fontsize=13, rotation=45)
    plt.title('Fig 3: Conversion Rate by Country',fontsize=25)
    plt.yticks(fontsize=20)
    plt.ylabel('country', fontsize=20)
    plt.xlabel('conversion rate',fontsize=20)
    plt.legend(loc=0, fontsize=20)

    plt.show()
    return

def plot_pie():

    for fig, query  in zip(xrange(1,3),[None,"country not in ['Argentina','Spain','Uruguay']"]):
        data=pivot_table(field='regions',values='conversion',query=query)
        data.fillna(0,inplace=True)
        f = plt.figure(num=fig, figsize=(23, 10))
        if query:
            extra_title='where ' + query
        else:
            extra_title=''
        f.suptitle('Fig {}: Percentage of Samples by Country {}'.format(fig,extra_title),fontsize=25)
        labels = data.index
        for ct,group in zip(xrange(1,3),['a_views','b_views']):
            ax = f.add_subplot(1,2,ct)
            sizes = data[group]
            colors = None
            explode = [.2 if i in ['Argentina', 'Spain'] else 0 for i in labels]

            patches, texts, autotexts=ax.pie(sizes, explode=explode, labels=labels, labeldistance=1.2,
               autopct='%1.1f%%', shadow=True, startangle=140 )

            # #patches, texts = ax.pie(sizes, labels=labels,autopct='%1.1f%%',
            #                         colors=colors,explode=explode,startangle=10,
            #                         # labeldistance=0.8)
            #
            for t in texts:
                #t.set_horizontalalignment('left')
                t.set_fontsize(16)

            if ct%2!=0:

                plt.legend(bbox_to_anchor=(1, 1),
                           bbox_transform=plt.gcf().transFigure,fontsize=12)
            plt.title(group, fontsize=26)


        plt.show()






plot_pie()

def algo():
    
    
    'create tables for chi-square contingency test'
    data = combine_tables().query("country not in ['Spain']")
    fields = ['sex', 'country', 'browser_language', 'browser']
    columns = 'test'
    values = 'conversion'

    m = pd.pivot_table(data, values=values, index=fields, columns=columns, aggfunc="count")
    m.columns = ['control', 'exp']
    m = m[m.control.notnull()]
    m = m[m.exp.notnull()]

    chi2, p, ddof, expected = st.chi2_contingency([m.control, m.exp])
    msg = "Test Statistic: {}\np-value: {}\nDegrees of Freedom: {}"
    print(msg.format(chi2, p, ddof))

    print 'Samples are balanced: ', p > .05

