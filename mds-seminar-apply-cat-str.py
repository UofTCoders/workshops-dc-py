#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
from sinfo import sinfo


sinfo()


# Pandas has options to control the dispalyed data frame output.

# In[2]:


pd.describe_option('row')


# In[3]:


pd.set_option('display.max_rows', 9)


# # Using apply

# In[4]:


iris = sns.load_dataset('iris')
iris.info()


# What does the `+` mean?

# In[5]:


iris.info(memory_usage='deep')


# In[6]:


iris.memory_usage()


# In[7]:


iris.memory_usage(deep=True)


# Will get back to this when we talk abot categories

# In[8]:


iris.mean()


# In[9]:


iris.agg('mean')


# In[10]:


iris.agg(['mean', 'median'])


# If we want to use a function that is not available through pandas, we can use apply.

# In[11]:


iris[['sepal_length', 'sepal_width']].apply(np.mean)


# The built in aggregation functions automatically drop non-numerical values. Apply does not, so an error is thrown with non-numerical cols.
# Throws an error
iris.apply(np.mean)
# We could drop the string columns if there are just a few and we know which.

# In[12]:


iris.drop(columns='species').apply(np.mean)


# If there are many, it is easier to use `.select_dtypes()`.

# In[13]:


iris.select_dtypes('number').apply(np.mean)


# ## Lambda functions
# 
# Unnamed functions that don't need to be defined.

# In[14]:


def my_fun(x):
    return x + 1

my_fun(5)


# In[15]:


my_lam = lambda x: x + 1

my_lam(5)


# In[16]:


(lambda x: x + 1)(5)


# Can be used together with apply to create any transformation to the dataframe values.

# In[17]:


iris.select_dtypes('number').apply(lambda x: x + 1)


# In[18]:


iris.select_dtypes('number').apply(lambda x: x + 1) - iris.select_dtypes('number')


# It looks like all are correct but when we check equality it seems not to be.

# In[19]:


(iris.select_dtypes('number').apply(lambda x: x + 1) - iris.select_dtypes('number')) == 1


# This is because of floating point error. Something that is good to be aware of and know that it can be fixed with `np.isclose`.

# In[20]:


np.isclose(iris.select_dtypes('number').apply(lambda x: x + 1) - iris.select_dtypes('number'), 1).all()


# By default, `.apply` (and other functions), work on column-wise, but can be set to work row-wise also.

# In[21]:


iris_num = iris.select_dtypes('number')


# In[22]:


# The highest value in any of the rows for each column.
iris_num.apply(lambda col: col.max())


# In[23]:


# The highest value in any of the columns for each row.
iris_num.apply(lambda row: row.max(), axis=1)


# In[24]:


# The highest value in any of the columns for each row.
iris_num.idxmax()


# In[25]:


# The highest value in any of the columns for each row.
iris_num.idxmax(axis=1)


# Sepal length seems to be the highest value for all rows.

# In[26]:


iris_num.idxmax(axis=1).value_counts()


# Built in pandas methods are optimized to be faster with pandas dataframees than applying a standard python method, so always use these when possible.

# In[27]:


# TODO change to iris if there is no diff with axis=1, there should be based on the mem layout...
square = pd.DataFrame(np.random.rand(2000, 2000))
square.shape


# In[28]:


get_ipython().run_cell_magic('timeit', '', 'square.max()')


# In[29]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(np.max)')


# In[30]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(max)')


# In[31]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(max, axis=1)')


# In[32]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(lambda x: x ** x, axis=0)')


# In[33]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(lambda x: x ** x, axis=1)')


# # Working with categorical data

# In[34]:


# titanic = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv') #sns.load_dataset('titanic')
titanic = pd.read_csv('/home/joel/Downloads/train.csv')
titanic


# Some of these columns I will not touch, so we're dropping them to fit the df on the screen.

# In[35]:


titanic = titanic.drop(columns=['SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
titanic


# In[36]:


titanic.info()


# In[37]:


titanic.memory_usage(deep=True) #.sum()


# In[38]:


titanic.select_dtypes('number').head()


# Survived and Pclass are not numerical variables, they are categorical.

# In[39]:


# import re
# titanic.rename(columns=lambda x: re.sub('(?!^)([A-Z]+)', r'_\1', x).lower())
titanic = titanic.rename(columns=str.lower)
titanic = titanic.set_index('passengerid')
titanic['survived'] = titanic['survived'] == 1
titanic['pclass'] = titanic['pclass'].map({1: 'first', 2: 'second', 3: 'third'})
titanic


# In[40]:


titanic.memory_usage(deep=True)


# Boolean takes less space and strings take more.

# In[41]:


titanic.dtypes


# In[42]:


pd.Categorical(titanic['sex'])


# In[43]:


titanic['sex'] = pd.Categorical(titanic['sex'])


# In[44]:


titanic.memory_usage(deep=True)


# In[45]:


# Stored as integers with a mapping, which can be seen with the cat accessor
titanic['sex'].cat.codes


# Categories can be ordered which allows comparisons.

# In[46]:


titanic['pclass'] = pd.Categorical(titanic['pclass'], categories=['third', 'second', 'first'], ordered=True)


# In[47]:


# Note that comparisons with string also work, but it is just comparing alphabetical order.
titanic['pclass'] > 'third'


# The order is also respected by pandas and seaborn.

# In[48]:


# mode, min and max work
titanic['pclass'].mode()


# In[49]:


titanic.groupby('pclass').size()


# In[50]:


sns.catplot(x='pclass', y='age', data=titanic, kind='swarm')


# In[51]:


# Value counts sorts based on value, not index.
titanic['pclass'].value_counts(normalize=True)


# In[52]:


titanic.dtypes


# In[53]:


# titanic.apply(lambda x: x + 1)
titanic.select_dtypes('number').apply(lambda x: x + 1)


# In[54]:


titanic.describe()


# In[55]:


# 'number', 'category', 'object' ,'bool'
titanic.select_dtypes('category').describe()


# In[56]:


# describe has an built-in way of doing this also, but it is more versatile to learn select dtype
titanic.describe(include='category')


# # String processing

# Could use lambda and the normal python string functions.

# In[57]:


'First Last'.lower()


# In[58]:


titanic['name'].apply(lambda x: x.lower())


# Pandas has built in accessor method for many string methods so that we don't have to use lambda.

# In[59]:


titanic['name'].str.lower()


# Note that these work on Series, not dataframes. So either use on one series at a time or a dataframe with a lmabda experssion.

# In[60]:


titanic


# ## What are the longest lastnames

# In[61]:


titanic['name'].str.split(',')


# In[62]:


titanic['name'].str.split(',', expand=True)


# Can be assigned to multiple columns, or select one column with indexing.

# In[63]:


titanic[['lastname', 'firstname']] = titanic['name'].str.split(',', expand=True)
titanic


# In[64]:


titanic['lastname_length'] = titanic['lastname'].str.len()
titanic


# In[65]:


titanic.sort_values('lastname_length', ascending=False).head()


# In[66]:


# Shortcut for sorting
titanic.nlargest(5, 'lastname_length')


# In[67]:


sns.distplot(titanic['lastname_length'], bins=20)


# How many times are lastnames duplicated.

# In[68]:


titanic['lastname'].value_counts().value_counts()


# How can we view the duplicated ones.

# In[69]:


titanic[titanic.duplicated('lastname', keep=False)].sort_values(['lastname'])


# Duplication is often due to women being registered under their husbands name. 

# We can get an idea, by checking how many vaues include a parenthesis.

# In[70]:


titanic.loc[titanic['name'].str.contains('\('), 'sex'].value_counts()


# In[71]:


titanic.loc[titanic['name'].str.contains('\('), 'sex'].value_counts(normalize=True)


# How to negate a boolean expression.

# In[72]:


titanic.loc[~titanic['name'].str.contains('\('), 'sex'].value_counts()


# There seems to be several reasons for parenthesis in the name. The ones we want to change are the ones who have 'Mrs' and a parenthesis in the name.

# In[73]:


# It is beneficial to break long method or indexeing chains in to several rows surrounded by parenthesis.
(titanic
    .loc[(titanic['name'].str.contains('\('))
        & (titanic['name'].str.contains('Mrs'))
        , 'sex']
    .value_counts()
)


# Dropped all male and 4 female passengers. Which females were dropped?

# In[74]:


(titanic
    .loc[(titanic['name'].str.contains('\('))
        & (~titanic['name'].str.contains('Mrs'))
        & (titanic['sex'] == 'female')
        , 'name']
)


# Even more precisely, we only want to keep the ones with a last and first name in the parentheiss. We can use the fact that these seems to be separated by a space.

# In[75]:


# Explain regex above
# titanic.loc[(titanic['name'].str.contains('\(')) & (titanic['sex'] == 'female'), 'sex'].value_counts()
titanic.loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'sex'].value_counts()


# From these passengers, we can extract the name in the parenthesis.

# In[76]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
)


# In[77]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str.partition(')')[0]
)


# In this case I could also have used string indexing to strip the last character, but this would give us issues if there are spaces at the end.

# In[78]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str[:-1]
)


# There is a more advanced way of getting this with regex directly, using a matching group to find anything in the parenthesis.

# In[79]:


# %%timeit
(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.extract("\((.+)\)")
)


# The two way partition method is just fine, and regex can feel a bit magical sometime, but it is good to know about if you end up working a lot with strings or need to extract complicated patterns.
# 
# Now lets get just the last names from this column and assign them back to the dataframe.

# In[80]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str.partition(')')[0]
    .str.rsplit(n=1, expand=True)
)


# All the lastnames without parenthsis will remain the same.

# In[81]:


titanic['real_last'] = titanic['lastname']


# Overwrite only the relevant columns.

# In[82]:


titanic.loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'real_last'] = (
    titanic
        .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
        .str.partition('(')[2]
        .str.partition(')')[0]
        .str.rsplit(n=1, expand=True)
        [1]
)


# In[83]:


titanic


# In[84]:


titanic['lastname'].value_counts().value_counts()


# In[85]:


titanic['real_last'].value_counts().value_counts()


# ## Extras
# Every command in this notebook
%hist# Grep through all history
%hist -g select# For easier version control
#!jupyter-nbconvert mds-seminar-apply-cat-str.ipynb --to python