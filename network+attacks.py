
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# In[40]:


df1=pd.read_csv("train_data.csv")
df2=pd.read_csv("test_data.csv")


# In[41]:


df1=df1.drop("connection_id",axis=1)
df1.head()


# In[42]:


df2=df2.drop("connection_id",axis=1)
df2.head()


# In[43]:


X_train=df1.drop("target",axis=1)
y_train=df1["target"]


# In[44]:


X_test=df2


# In[45]:


rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)


# In[46]:


predictions=rfc.predict(X_test)


# In[47]:


predictions


# In[48]:


prediction = pd.DataFrame(predictions, columns=['target']).to_csv('final.csv')


# In[ ]:




