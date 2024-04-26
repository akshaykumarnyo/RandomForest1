#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Random Forest Classifier With Pipeline And Hyperparameter


# In[11]:


import seaborn as sns
df=sns.load_dataset("tips")


# In[12]:


df


# In[13]:


df.isnull().sum()


# In[14]:


df.describe()


# In[15]:


df.info()


# In[16]:


df["day"].unique()


# In[17]:


df["time"].unique()


# In[18]:


df["time"]


# In[19]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df["time"]=encoder.fit_transform(df["time"])


# In[20]:


df['time']


# In[21]:


## Independent and dependent feature


# In[22]:


x=df.drop(labels=["time"],axis=1)
y=df["time"]


# In[23]:


x


# In[24]:


y


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[26]:


x.head()


# In[27]:


y


# In[28]:


x["day"].value_counts()


# In[29]:


## Pipeline


# In[30]:


from sklearn.pipeline import Pipeline


# In[31]:


from sklearn.impute import SimpleImputer ### Handle Missing Values


# In[32]:


from sklearn.preprocessing import StandardScaler ### Feature Scaling 
from sklearn.preprocessing import OneHotEncoder ## categorical to numerical
from sklearn.compose import ColumnTransformer 


# In[33]:


categorical_cols=["sex","smoker","day"]
numerical_cols=["total_bill","tip","size"]


# In[34]:


## Feature Engineering Automation



## Numerical Pipelines

num_pipeline=Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")), ## Missing Values
        ("scaler",StandardScaler())   ### feature Scaling 
    ]
)

### categorical Pipeline
cat_pipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), ## handling Missing values 
                ("onehotencoder",OneHotEncoder())  ### Categorical feature to numerical
                 ]
             )


# In[35]:


preprocessor=ColumnTransformer([
    ("num_pipeline",num_pipeline,numerical_cols),
    ("cat_pipeline",cat_pipeline,categorical_cols)
])


# In[36]:


x_train=preprocessor.fit_transform(x_train)


# In[37]:


x_test=preprocessor.transform(x_test)


# In[38]:


x_train


# In[39]:


x_test


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

models = {
    "Random Forest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier()
}

def evaluate_model(x_train, y_train, x_test, y_test, models):
    report = {}
    for name, model in models.items():
        # Train model
        model.fit(x_train, y_train)

        # Predict testing data
        y_test_pred = model.predict(x_test)
        
        # Get accuracy score for test data
        test_model_score = accuracy_score(y_test, y_test_pred)
        
        report[name] = test_model_score
    
    return report

evaluate_model(x_train, y_train, x_test, y_test, models)


# In[43]:


classifier=RandomForestClassifier()


# In[44]:


### Hyperparameter Tunning 
params={"max_depth":[3,5,10,None],
       "n_estimators":[100,200,300],
       "criterion":["gini","entropy"]
       }


# In[45]:


from sklearn.model_selection import RandomizedSearchCV


# In[48]:


classifier,param_distributions=params,scoring="accuracy",cv=5,verbose=3cv=RandomizedSearchCV()


# In[49]:


cv.fit(x_train,y_train)


# In[50]:


cv.best_params_


# In[52]:


param={'n_estimators': 300, 'max_depth': 10, 'criterion': 'gini'}


# In[60]:


Forest=RandomForestClassifier(n_estimators=300,max_depth=10,criterion="gini")


# In[61]:


Forest.fit(x_train,y_train)


# In[63]:


y_pred=Forest.predict(x_test)


# In[64]:


## accuracy score
print(accuracy_score(y_test,y_pred))


# In[66]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[67]:


print(f"accuracy_score:\n{accuracy_score(y_pred, y_test)}")
print(f"Confusion matrix:\n{confusion_matrix(y_pred, y_test)}")
print(f"classification_report:\n{classification_report(y_pred, y_test)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




