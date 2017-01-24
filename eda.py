from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.cluster.k_means_ import MiniBatchKMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import scipy.stats as st

from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit,LogitResults
from sklearn.metrics import confusion_matrix,f1_score

from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import numpy as np
pd.set_option('display.width',1000)

def combine_tables():
    test=pd.read_csv('test_table.csv')
    user=pd.read_csv('user_table.csv')
    c=pd.merge(test,user, how='left',left_on='user_id',right_on='user_id', suffixes=('_l','_r'),indicator=True)
    return c



def model_inputs():

    d.drop(['age','_merge','date','user_id'],axis=1,inplace=True)
    y=d.pop('conversion')
    dummies=pd.get_dummies(d,columns=['sex','ads_channel','source','device','browser_language','browser' ,'country'])
    x=pd.concat([dummies,d['test']],axis=1)
    return   x,y


def undersample(df,c=None):
    positive=df[df[c]==1]
    negative=df[df[c]==0]
    minsamp=min(len(positive),len(negative))
    positive=positive.sample(n=minsamp)
    negative=negative.sample(n=minsamp)
    combine=pd.concat([positive,negative],axis=0)
    return combine




def classify():

    data = combine_tables().query("country not in ['Spain']")
    data=undersample(data,c='conversion')
    y=data.pop('conversion')

    dummies=pd.get_dummies(data,columns=['country','browser','browser_language','device','ads_channel','source','sex'])
    dummies.drop(['user_id','date','age','_merge','test'],inplace=True, axis=1)

    x=dummies
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)

    model=RandomForestClassifier()

    fit=model.fit(xtrain,ytrain)

    ypredict=model.predict(xtest)

    print 'confusion matrix'
    conf=confusion_matrix(ytest,ypredict)
    print conf
    f1=f1_score(ytest,ypredict)
    print 'f1 score =' ,f1

    most_import_index= np.argsort(model.feature_importances_)[::-1]
    print 'feature importances'
    print x.columns
    for i in most_import_index:
        print x.columns[i]

def eda():
    data=combine_tables().query("country not in ['Spain','Argentina','Uruguay']")
    fields=['country','sex','browser_language','browser','source']
    columns='test'
    values='conversion'

    for i in fields:
        total=pd.pivot_table (data,values=values,index=i,columns=columns,aggfunc="count")
        convert=pd.pivot_table(data, values=values, index=i, columns=columns, aggfunc="sum")
        rate=pd.pivot_table(data,values=values,index=i,columns=columns,aggfunc='mean')
        m=pd.concat([total, convert, rate],axis=1)
        print m



        if i=='country':

            f=plt.figure(3,figsize=(25,15))
            f.suptitle('Fig 3: Conversion Rate by Country',fontsize=22)
            ax=f.add_subplot(111)
            plt.plot(xrange(len(m)),m.iloc[:,4], label='control')
            plt.plot(xrange(len(m)), m.iloc[:, 5],label='exp')
            ax.set_xticklabels(m.index,fontsize=14,rotation=45)
            plt.yticks(fontsize=16)
            plt.legend(loc=0,fontsize=20)

            f=plt.figure(num=1,figsize=(23,12))

            f.suptitle('Fig 2: Percentage of Samples by Country (Spain, Argentina, Uruguay Removed)',fontsize=20)
            ax=f.add_subplot(121)

            labels = m.index
            sizes = m.iloc[:,1]
            colors=None
            explode = [.2 if i in ['Argentina','Spain'] else 0 for i in labels]

            plt.pie(sizes,explode=explode , labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140,)
            plt.title('Control',fontsize=16)


            ax2=f.add_subplot(122)
            sizes=m.iloc[:,0]
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140,)
            plt.title('Experiment',fontsize=16)
            plt.show()

def algo():
    data = combine_tables().query("country not in ['Spain']")
    fields = ['sex', 'country', 'browser_language', 'browser']
    columns = 'test'
    values = 'conversion'

    m= pd.pivot_table(data, values=values, index=fields, columns=columns, aggfunc="count")
    m.columns=['control','exp']
    m = m[m.control.notnull()]
    m = m[m.exp.notnull()]


    chi2, p, ddof, expected = st.chi2_contingency([m.control, m.exp])
    msg = "Test Statistic: {}\np-value: {}\nDegrees of Freedom: {}"
    print(msg.format(chi2, p, ddof))

    print 'Samples are balanced: ', p > .05

eda()