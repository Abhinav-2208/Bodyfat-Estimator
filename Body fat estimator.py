#!/usr/bin/env python
# coding: utf-8

# # Importing necesary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('bodyfat.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


def plotdistplots(col):
    plt.figure(figsize=(12,5))
    sns.distplot(df['BodyFat'],color='magenta',hist=False,label='Bodyfat')
    sns.distplot(df[col],color='red',hist=False,label=col)
    plt.legend()
    plt.show()
    
cols=list(df.columns)
for i in cols:
    print(f'Distribution plots for {i} is shown below')
    plotdistplots(i)
    #print("-"*100)


# In[6]:


import scipy.stats as stats


# In[7]:


def drawplots(df,col):
    plt.figure(figsize=(15,7))
    plt.subplot(1,3,1)
    plt.hist(df[col],color='magenta')
    
    plt.subplot(1,3,2)
    stats.probplot(df[col],dist='norm',plot=plt)
    
    plt.subplot(1,3,3)
    sns.boxplot(df[col],color='magenta')
    
    plt.show()
    
cols=list(df.columns)
for i in range(len(cols)):
    print(f'Distribution plots for the feature {cols[i]} are shown below')
    drawplots(df,cols[i])
    print("="*100)


# # Checking for outliers

# In[8]:


upperlimit=[]
lowerlimit=[]
for i in df.columns:
    upperlimit.append(df[i].mean()+(df[i].std())*4)
    lowerlimit.append(df[i].mean()-(df[i].std())*4)


# In[9]:


cols=list(df.columns)
j=0
for i in range(len(cols)):
    temp=df.loc[(df[cols[i]]>upperlimit[j])&(df[cols[i]]<lowerlimit[j])]
    j+=1


# In[10]:


temp


# # Using ExtraTrees Regressor for Feature Selection 

# In[11]:


data=df.copy()
test=data['BodyFat']
train=data.drop(['BodyFat'],axis=1)


# In[12]:


from sklearn.ensemble import ExtraTreesRegressor
er=ExtraTreesRegressor()
er.fit(train,test)


# In[13]:


er.feature_importances_


# In[14]:


series=pd.Series(er.feature_importances_,index=train.columns)
series.nlargest(5).plot(kind='barh',color='green')


# # Using Mutual Information gain for feature selection

# In[15]:


from sklearn.feature_selection import mutual_info_regression


# In[16]:


mr=mutual_info_regression(train,test)


# In[17]:


plotdata=pd.Series(mr,index=train.columns)
plotdata.nlargest(5).plot(kind='barh',color='green')


# # Removing Correlation

# In[18]:


data


# In[19]:


plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,cmap='plasma')


# In[20]:


def correlation(df,thresold):
    colcor=set()
    cormat=df.corr()
    for i in range(len(cormat)):
        for j in range(i):
            
            
            """ for each cell get the value of that cell by .iloc[i][j],
            where i is the row and j is the col if that abs(value) is greater than the thresold
            ,get the col_name and add it in the set""" 
            if abs(cormat.iloc[i][j]>thresold):
                colname=cormat.columns[i]
                colcor.add(colname)
    return colcor
ans=correlation(train,thresold=0.85)
ans


# # From the above feature selection techniques we can say that the features are recomended by Extra Tree Regressor and the mutual
# #_information_gain are correct and from the correlation map we get to observe the similar pattern we noticed that Abdomen and Hip
# #having similar features they are having colinearity same goes with Knee and Thigh,we can keep either any of them and we noticed 
# #that feature Abdomen give more feature importance score in comparision to Hip ,so i will be selecting that

# In[21]:


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[22]:


temp=data[list(data.columns)]
info=pd.DataFrame()
info['VIF']=[variance_inflation_factor(temp.values,i) for i in range(temp.shape[1])]
info['column']=temp.columns
info


# In[23]:


cols1=list(series.nlargest(5).index)
cols2=list(plotdata.nlargest(5).index)
cols1,cols2


# # We will go with the weight and Hip as Hip and Thigh are very much related,so we will select the cols1 feature and drop every
# #other feature,if that doesn't produce any further importance we will try with some other feature.

# In[24]:


totrain=train[cols1]
totrain.head()


# In[25]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn import metrics
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso,Ridge


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(totrain,test,test_size=0.2)
x_train.shape,x_test.shape


# In[27]:


reg=DecisionTreeRegressor()
reg.fit(x_train,y_train)
plt.figure(figsize=(15,7))
tree.plot_tree(reg,filled=True)


# In[28]:


path=reg.cost_complexity_pruning_path(x_train,y_train)
ccp_alpha=path.ccp_alphas


# In[29]:


alphalist=[]
for i in range(len(ccp_alpha)):
    reg=DecisionTreeRegressor(ccp_alpha=ccp_alpha[i])
    reg.fit(x_train,y_train)
    alphalist.append(reg)


# In[30]:


trainscore=[alphalist[i].score(x_train,y_train) for i in range(len(alphalist))]
testscore=[alphalist[i].score(x_test,y_test) for i in range(len(alphalist))]
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.plot(ccp_alpha,trainscore,marker='o',label='training',color='magenta',drawstyle='steps-post')
plt.plot(ccp_alpha,testscore,marker='+',label='testing',color='red',drawstyle='steps-post')
plt.legend()
plt.show()


# # Normal approach

# In[31]:


clf=DecisionTreeRegressor(ccp_alpha=1)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(f'Decision Tree Normal Approach :{metrics.r2_score(y_test,y_pred)}')

rf=RandomForestRegressor(n_estimators=1000,ccp_alpha=1)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)
print(f'Random Forest Normal Approach : {metrics.r2_score(y_test,y_pred_rf)}')


# In[32]:


params={
    'RandomForest':{
        'model':RandomForestRegressor(),
        'params':{
            'n_estimators':[int(x) for x in np.linspace(start=1,stop=1200,num=10)],
            'criterion':['mse','mae'],
            'max_depth':[int(x) for x in np.linspace(start=1,stop=30,num=5)],
            'min_samples_split':[2,5,10,12],
            'min_samples_leaf':[2,5,10,12],
            'max_features':['auto','sqrt'],
            'ccp_alpha':[1,2,2,5,3,3,5,4,5],
        }
    },
    
    'D-tree':{
        'model':RandomForestRegressor(),
        'params':{
            'criterion':['mse','mae'],
           # 'splitter':['best','random'],
            'min_samples_split':[1,2,5,10,12],
            'min_samples_leaf':[1,2,5,10,12],
            'max_features':['auto','sqrt'],
            'ccp_alpha':[1,2,2,5,3,3,5,4,5],
        }
    },
    'SVM':{
        'model':SVR(),
        'params':{
            'C':[0.25,0.50,0.75,1.0],
            'tol':[1e-10,1e-5,1e-4,0.025,0.50,0.75],
            'kernel':['linear','poly','rbf','sigmoid'],
            'max_iter':[int(x) for x in np.linspace(start=1,stop=250,num=10)]
        }
    }
}


# In[33]:


scores=[]
for modelname,mp in params.items():
    clf=RandomizedSearchCV(mp['model'],param_distributions=mp['params'],
                          cv=5,n_jobs=-1,n_iter=10,scoring='neg_mean_squared_error')
    
    clf.fit(x_train,y_train)
    scores.append({
        'model_name':modelname,
        'best_score':clf.best_score_,
        'best_estimator':clf.best_estimator_
    })


# In[35]:


scoresdf=pd.DataFrame(scores,columns=['model_name','best_score','best_estimator'])
scoresdf


# In[36]:


scores[0]['best_estimator']


# In[37]:


rf=scores[0]['best_estimator']
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print(metrics.r2_score(y_test,y_pred))


# In[38]:


totrainlist=np.array(totrain)
predicted=[]
for i in range(len(totrainlist)):
    predicted.append(rf.predict([totrainlist[i]]))
    
totrain['Actual Result']=test
totrain['predicted Result']=np.array(predicted)
totrain


# In[40]:


sns.distplot(totrain['Actual Result'],label='Actual Result',hist=False,color='magenta')
sns.distplot(totrain['predicted Result'],label='predicted Result',hist=False,color='red')
plt.legend()
plt.plot()


# In[41]:


#Saving the model
import pickle
file=open('bodyfatmodel1.pkl','wb')
pickle.dump(rf,file)
file.close()


# # Coding of UI part

# In[45]:


from flask import flask,request,render_template


# In[ ]:




