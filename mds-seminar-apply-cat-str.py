#!/usr/bin/env python
# coding: utf-8

# When presenting, vertical whitespace matters. I tend to do both maximize my browser (`F11`) and go into single document mode. To get to single document mode, we can use the command palette, either by clicking it in the left sidebar or by typing `Ctrl+Shift+c`. The command palette is great becaues it also show the shortcut we could use to get into single document mode directly via the shortcut `Ctrl+Shift+d`. When we're done with the sidebar we can close it with `Ctrl+b`.

# First create a markdown cell with a header.
# 
# # MDS seminar
# 
# I usually add a few import that I am sure I will use up front and then more as I go. If I do a lot of prototyping, I just add them in the cell I am currently working in and then move them to the first cell when I am ready to commit something.

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
from sinfo import sinfo


sinfo() # Writes dependencies to `sinfo-requirements.txt` by default


# # Initial textual EDA

# There are many ways to get sample data to work with, including scikit-learn, statsmodels, and quilt. For small examples, I tend to use seaborn.

# In[2]:


iris = sns.load_dataset('iris')
iris.head()


# It is a little bit annoying to type head everytime I want to look at a dataframe. Pandas has options to control the dispalyed data frame output and even a nice search interface to find 

# In[3]:


pd.describe_option('row')


# In[4]:


pd.set_option('display.max_rows', 9)


# We can see that this has changed the current value.

# In[5]:


pd.describe_option('max_row')


# And if we type the `iris` now, we wont get flooded with 60 rows.

# In[6]:


iris


# I like that this shows the beginning and the end of the data frame, as well as the dimensions which don't show up with head. The only drawback is that you need to set it back if you want to display more rows, or override it temporarily with the context manager. So that is worth keeping in mind.

# To get the default back, we could use `pd.reset_option('max_row')`.
# 
# A good follow up would be to check if there are any NaNs and which data types pandas has identified (we already have a good idea from the above).

# In[7]:


iris.info()


# We can see that there are no NaNs since every columns has the same number of non-null entries as the number of entries in the index (150). The data types and index type match up with what we might expect from glancing at the values in previously. We can find out the number of unique values in each column via `nunique()`.

# In[8]:


iris.nunique()


# `describe()` shows descriptive summary statistics.

# In[9]:


iris.describe()


# Note that describe by default only show numerical columns (if there are any), but we can specify that we want to include other column types.

# In[10]:


iris.describe(include='object')


# We can also tell it to include all column and control the displayed percentiles.

# In[11]:


iris.describe(percentiles=[0.5], include='all')


# # Using apply
# 
# Aggregation functions can be specified in many different ways in pandas. From highly optimized built in functions to highly flexible arbitrary functions. 

# In[12]:


iris.mean()


# `agg()` is a different interface to summary functions, which also allows for multiple functions to be past in the same call.

# In[13]:


iris.agg('mean')


# In[14]:


iris.agg(['mean', 'median'])


# If we want to use a function that is not available through pandas, we can use apply.

# In[15]:


iris[['sepal_length', 'sepal_width']].apply(np.mean)


# The built in aggregation functions automatically drop non-numerical values. Apply does not, so an error is thrown with non-numerical cols.
# Throws an error
iris.apply(np.mean)
# We could drop the string columns if there are just a few and we know which.

# In[16]:


iris.drop(columns='species').apply(np.mean)


# If there are many, it is easier to use `.select_dtypes()`.

# In[17]:


iris.select_dtypes('number').apply(np.mean)


# ## Lambda functions
# 
# Unnamed functions that don't need to be defined.

# In[18]:


def add_one(x):
    return x + 1

add_one(5)


# Lambda functions can be used without being named, so they are effective for throwaway functions that you are likely only to use only once.

# In[19]:


(lambda x: x + 1)(5)


# Lambda functions can be assigned to a variable name if so desired.

# In[20]:


my_lam = lambda x: x + 1

my_lam(5)


# Just as with functions, there is nothing special with the letter `x`, it is just a variable name and you can call it whatever you prefer.

# In[21]:


(lambda a_good_descriptive_name: a_good_descriptive_name + 1)(5)


# Custom function, both named and unnamed, can be used together with apply to create any transformation to the dataframe values.

# In[22]:


iris_num = iris.select_dtypes('number')
iris_num.apply(add_one)


# In[23]:


iris_num.apply(lambda x: x + 1)


# We can check if they are correct by surrounding with parentheses and asser equality.

# In[24]:


iris_num.apply(lambda x: x + 1) == iris_num.apply(add_one)


# In[25]:


(iris_num.apply(lambda x: x + 1) == iris_num.apply(add_one)).all()


# We could also have checked if the new df minus the old one ends up with all 1.

# In[26]:


iris_num.apply(lambda x: x + 1) - iris_num


# It looks like all are correct but when we check equality it seems not to be.
# 
# #TODO short explanation and link

# In[27]:


(iris_num.apply(lambda x: x + 1) - iris_num) == 1


# This is because of floating point error. Something that is good to be aware of and know that it can be fixed with `np.isclose`.

# In[28]:


np.isclose(iris_num.apply(lambda x: x + 1) - iris_num, 1).all()


# ### Row and column wise apply
# 
# By default, `.apply` (and other functions), work column-wise, but can be set to work row-wise instead.

# In[29]:


# The highest value in any of the rows for each column.
iris_num.apply(lambda col: col.max())


# In[30]:


# The highest value in any of the columns for each row.
iris_num.apply(lambda row: row.max(), axis=1)


# In[31]:


# The highest value in any of the columns for each row.
iris_num.idxmax()


# In[32]:


# The highest value in any of the columns for each row.
iris_num.idxmax(axis=1)


# Sepal length seems to be the highest value for all rows.

# In[33]:


iris_num.idxmax(axis=1).value_counts()


# ### Testing performance
# 
# Built in pandas methods are optimized to be faster with pandas dataframees than applying a standard python method, so always use these when possible.

# In[34]:


# TODO change to iris if there is no diff with axis=1, there should be based on the mem layout...
many_cols = pd.DataFrame(np.random.rand(2, 5000))
many_rows = pd.DataFrame(np.random.rand(5000, 2))
square = pd.DataFrame(np.random.rand(5000, 5000))


# Columns are faster than rows since each column is a numpy array

# In[35]:


get_ipython().run_cell_magic('timeit', '', 'many_rows.mean()')


# In[36]:


get_ipython().run_cell_magic('timeit', '', 'many_rows.mean(axis=1)')


# In[37]:


get_ipython().run_cell_magic('timeit', '', 'many_cols.mean()')


# In[38]:


get_ipython().run_cell_magic('timeit', '', 'many_cols.mean(axis=1)')


# In[39]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(np.mean, axis=1)')


# In[40]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(np.mean, axis=0)')


# In[41]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(lambda x: sum(x) / len(x), axis=1)')


# In[42]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(lambda x: sum(x) / len(x), axis=0)')


# In[ ]:





# In[43]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(lambda x: x ** x, axis=0)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'square.apply(lambda x: x ** x, axis=1)')


# In[ ]:


# snakeviz


# # Working with categorical data

# In[ ]:


iris.info()


# but how should we interpret the `+` sign under memory usage? In the help we can see that there is one option that affects memory usage, let's use it.

# In[ ]:


iris.info(memory_usage='deep')


# What happened? How could the data frame almost tripple in size? The `info()` method's docstring explains why:
# 
# > Without deep introspection a memory estimation is made based in column dtype and number of rows assuming values consume the same memory amount for corresponding dtypes. With deep memory introspection, a real memory usage calculation is performed at the cost of computational resources.
# 
# So deep memory introspection shows the real memory usage, but it is still a bit cryptic what part of the dataframe is responsible for this extra size. To find this out, it is helpful to understand that pandas dataframes essentially consist of numpy arrays held together with some super smart glue. Knowing that, it would be interesting to inspect whether any of the columns report different size measures with and without deep memory introspection. Instead of the more general `info()` method, we can use one specific to memory usage to find this out.

# In[ ]:


iris.memory_usage()


# In[ ]:


iris.memory_usage(deep=True)


# From this, it is clear that it is the species column that changes, everything else remains the same. Above we saw that this column is of dtype "object". To understand what is happening, we first need to know that a numpy array is stored in the computer's memory as a contiguous (uninterupted) segment. This is one of the reasons why numpy is so fast, it only needs to find the start of the array and then access a sequential length from the start point instead of going to find every single object (which is how a lists work in python). However, in order for numpy to store objects sequentially in memory, it needs to allocate a certain space for each object. This is fine for integer up to a certain size or floats up to a certain precision, but with strings (or more complex object such as lists and dictionaries), numpy cannot fit them into the same sized chunks in an effective manner and the actual object is stored outside the array. So what is inside the array? Just a reference (also called a pointer) to where in memory the actual object is stored and these references are of a fixed size:
# 
# ![](./img/int-vs-pointer-memory-lookup.png)
# 
# [Image source](https://stackoverflow.com/questions/21018654/strings-in-a-dataframe-but-dtype-is-object/21020411#21020411)
# 
# What happens when we specify to use the deep memory introspection is that pandas finds and calculates the size of each of the objects in memory. With the shallow introspection, it simply reports the values of the references that are actually stored in the array. 
# 
# Note that memory usage is not the same as disk usage. Objects can take up additional space in memory depending on how they are constructed.

# In[ ]:


iris.to_csv('iris.csv')
get_ipython().system('ls -lh iris*')


# In[ ]:


# titanic = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv') #sns.load_dataset('titanic')
titanic = pd.read_csv('/home/joel/Downloads/train.csv')
titanic


# Some of these columns I will not touch, so we're dropping them to fit the df on the screen.

# In[ ]:


titanic = titanic.drop(columns=['SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
titanic


# In[ ]:


titanic.info()


# In[ ]:


titanic.memory_usage(deep=True) #.sum()


# In[ ]:


titanic.select_dtypes('number').head()


# Survived and Pclass are not numerical variables, they are categorical.

# In[ ]:


# import re
# titanic.rename(columns=lambda x: re.sub('(?!^)([A-Z]+)', r'_\1', x).lower())
titanic = titanic.rename(columns=str.lower)
titanic = titanic.set_index('passengerid')
titanic['survived'] = titanic['survived'] == 1
titanic['pclass'] = titanic['pclass'].map({1: 'first', 2: 'second', 3: 'third'})
titanic


# In[ ]:


titanic.memory_usage(deep=True)


# Boolean takes less space and strings take more.

# In[ ]:


titanic.dtypes


# In[ ]:


pd.Categorical(titanic['sex'])


# In[ ]:


titanic['sex'] = pd.Categorical(titanic['sex'])


# In[ ]:


titanic.memory_usage(deep=True)


# In[ ]:


# Stored as integers with a mapping, which can be seen with the cat accessor
titanic['sex'].cat.codes


# Categories can be ordered which allows comparisons.

# In[ ]:


titanic['pclass'] = pd.Categorical(titanic['pclass'], categories=['third', 'second', 'first'], ordered=True)


# In[ ]:


# Note that comparisons with string also work, but it is just comparing alphabetical order.
titanic['pclass'] > 'third'


# The order is also respected by pandas and seaborn.

# In[ ]:


# mode, min and max work
titanic['pclass'].mode()


# In[ ]:


titanic.groupby('pclass').size()


# In[ ]:


sns.catplot(x='pclass', y='age', data=titanic, kind='swarm')


# In[ ]:


# Value counts sorts based on value, not index.
titanic['pclass'].value_counts(normalize=True)


# In[ ]:


titanic.dtypes


# In[ ]:


# titanic.apply(lambda x: x + 1)
titanic.select_dtypes('number').apply(lambda x: x + 1)


# In[ ]:


titanic.describe()


# In[ ]:


# 'number', 'category', 'object' ,'bool'
titanic.select_dtypes('category').describe()


# In[ ]:


# describe has an built-in way of doing this also, but it is more versatile to learn select dtype
titanic.describe(include='category')


# # String processing

# Could use lambda and the normal python string functions.

# In[ ]:


'First Last'.lower()


# In[ ]:


titanic['name'].apply(lambda x: x.lower())


# Pandas has built in accessor method for many string methods so that we don't have to use lambda.

# In[ ]:


titanic['name'].str.lower()


# Note that these work on Series, not dataframes. So either use on one series at a time or a dataframe with a lmabda experssion.

# In[ ]:


titanic


# ## What are the longest lastnames

# In[ ]:


titanic['name'].str.split(',')


# In[ ]:


titanic['name'].str.split(',', expand=True)


# Can be assigned to multiple columns, or select one column with indexing.

# In[ ]:


titanic[['lastname', 'firstname']] = titanic['name'].str.split(',', expand=True)
titanic


# In[ ]:


titanic['lastname_length'] = titanic['lastname'].str.len()
titanic


# In[ ]:


titanic.sort_values('lastname_length', ascending=False).head()


# In[ ]:


# Shortcut for sorting
titanic.nlargest(5, 'lastname_length')


# In[ ]:


sns.distplot(titanic['lastname_length'], bins=20)


# How many times are lastnames duplicated.

# In[ ]:


titanic['lastname'].value_counts().value_counts()


# How can we view the duplicated ones.

# In[ ]:


titanic[titanic.duplicated('lastname', keep=False)].sort_values(['lastname'])


# Duplication is often due to women being registered under their husbands name. 

# We can get an idea, by checking how many vaues include a parenthesis.

# In[ ]:


titanic.loc[titanic['name'].str.contains('\('), 'sex'].value_counts()


# In[ ]:


titanic.loc[titanic['name'].str.contains('\('), 'sex'].value_counts(normalize=True)


# How to negate a boolean expression.

# In[ ]:


titanic.loc[~titanic['name'].str.contains('\('), 'sex'].value_counts()


# There seems to be several reasons for parenthesis in the name. The ones we want to change are the ones who have 'Mrs' and a parenthesis in the name.

# In[ ]:


# It is beneficial to break long method or indexeing chains in to several rows surrounded by parenthesis.
(titanic
    .loc[(titanic['name'].str.contains('\('))
        & (titanic['name'].str.contains('Mrs'))
        , 'sex']
    .value_counts()
)


# Dropped all male and 4 female passengers. Which females were dropped?

# In[ ]:


(titanic
    .loc[(titanic['name'].str.contains('\('))
        & (~titanic['name'].str.contains('Mrs'))
        & (titanic['sex'] == 'female')
        , 'name']
)


# Even more precisely, we only want to keep the ones with a last and first name in the parentheiss. We can use the fact that these seems to be separated by a space.

# In[ ]:


# Explain regex above
# titanic.loc[(titanic['name'].str.contains('\(')) & (titanic['sex'] == 'female'), 'sex'].value_counts()
titanic.loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'sex'].value_counts()


# From these passengers, we can extract the name in the parenthesis.

# In[ ]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
)


# In[ ]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str.partition(')')[0]
)


# In this case I could also have used string indexing to strip the last character, but this would give us issues if there are spaces at the end.

# In[ ]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str[:-1]
)


# There is a more advanced way of getting this with regex directly, using a matching group to find anything in the parenthesis.

# In[ ]:


# %%timeit
(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.extract("\((.+)\)")
)


# The two way partition method is just fine, and regex can feel a bit magical sometime, but it is good to know about if you end up working a lot with strings or need to extract complicated patterns.
# 
# Now lets get just the last names from this column and assign them back to the dataframe.

# In[ ]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str.partition(')')[0]
    .str.rsplit(n=1, expand=True)
)


# All the lastnames without parenthsis will remain the same.

# In[ ]:


titanic['real_last'] = titanic['lastname']


# Overwrite only the relevant columns.

# In[ ]:


titanic.loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'real_last'] = (
    titanic
        .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
        .str.partition('(')[2]
        .str.partition(')')[0]
        .str.rsplit(n=1, expand=True)
        [1]
)


# In[ ]:


titanic


# In[ ]:


titanic['lastname'].value_counts().value_counts()


# In[ ]:


titanic['real_last'].value_counts().value_counts()


# ## Extras
# Every command in this notebook
%hist# Grep through all history
%hist -g select# For easier version control
!jupyter-nbconvert mds-seminar-apply-cat-str.ipynb --to python