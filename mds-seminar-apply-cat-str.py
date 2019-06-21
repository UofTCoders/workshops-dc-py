#!/usr/bin/env python
# coding: utf-8

# I will try to cover my general workflow including small things that might be skipped otherwise.
# 
# - Avoid setting conda Python first on Path for unix systems since this can interfere with other application. Instead use the conda init script to activate the environment when you need to use it.
# - Use aliases for commonly accessed functions.
# - When presenting, vertical whitespace matters. I tend to do both maximize my browser (`F11`) and go into single document mode.
# - To get to single document mode, we can use the command palette, either by clicking it in the left sidebar or by typing `Ctrl+Shift+c`.
# - The command palette is great because it also show the shortcut we could use to get into single document mode directly via the shortcut `Ctrl+Shift+d`.
# - When we're done with the sidebar we can close it with `Ctrl+b`.

# First create a markdown cell with a header. We can use the menu up here or `m`, `y`, and `r` to switch between cell modes. Numbers switch between header levels.
# 
# # MDS seminar
# 
# - I usually add a few import that I am sure I will use up front and then more as I go. If I do a lot of prototyping, I just add them in the cell I am currently working in and then move them to the first cell when I am ready to commit something.
# - I usually open the object inspector on the side to get help with functions, but not for presentations because of screen real estate, instead I press `Shift+Tab` or use `?` to view docstrings.

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
from sinfo import sinfo


sinfo() # Writes dependencies to `sinfo-requirements.txt` by default


# # Pandas display options

# There are many ways to get sample data to work with, including `scikit-learn.datasets`, `statsmodels.datasets` (includes all R datasets), and `quilt` (a package manager for data sets). More [details can be found in this SO answer](https://stackoverflow.com/a/29956221/2166823). For small examples, I tend to use `seaborn.load_dataset()` since I will import seaborn anyways (note these datasets are just there for the `seaborn` documentation and may change without notice).

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
# 
# To get the default back, we could use `pd.reset_option('max_row')`.

# # Initial textual EDA

# - Now that we have three headings, it can be good to check out the ToC extension. It allows us to see an overview of the document and to jump directly to specific code cell.
# - Extensions like this can be installed via the command line or the new extension manager, which for now needs to be enabled in the advanced settings menu (but there will be a menu item in JL-1.0).
# - Another useful extension is the spellchecking one which I have enabled, this is why some words show up in red background color.
# - There is an extension that allows interactivity with `matplotlib`, for example panning and zooming.
# - I also like the system-monitor topbar extension that shows how much RAM the current notebook is using and `nbdime` which shows jupyter notebook diffs formatted in a nice waay without the json markup, more on that in the end if we have time. The dark theme I normally use is also available via an extension.

# A good follow up from viewing the dataframe is to check if there are any NaNs and which data types pandas has identified (we already have a good idea from the above).

# In[7]:


iris.info()


# We can see that there are no NaNs since every columns has the same number of non-null entries as the number of entries in the index (150). The data types and index type match up with what we might expect from glancing at the values previously. We can find out the number of unique values in each column via `nunique()`, which is useful to understand which variables are categorical.

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


# As we saw, the default behavior for `describe()` is to include all numerical columns. If there are no numerical columns, `describe()` will instead show summary statistics for whichever columns are available.

# In[12]:


iris['species'].describe()


# If there are many non-numerical columns, it can be tedious to write each one out. The `select_dtypes()` method can be used to get all column of a certain datatype.

# In[13]:


iris.select_dtypes('number').columns


# In[14]:


iris.select_dtypes('object').describe()


# # Performance profiling 

# Above we have seen three approaches to getting the summary statistics of the species column, so which one should we use? There are several factors leading into this decision, including code clarity, consistency, and performance (both memory and time). Assessing how long a chunk of code takes to run is referred to as profiling or benchmarking. We will walk through a couple of ways to do this in JupyterLab, but before we start it is important to note that code optimization should only be done when necessary, as once written by [Donald Knuth](http://wiki.c2.com/?PrematureOptimization):
# 
# > Programmers waste enormous amounts of time thinking about, or worrying about, the speed of noncritical parts of their programs, and these attempts at efficiency actually have a strong negative impact when debugging and maintenance are considered. We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. Yet we should not pass up our opportunities in that critical 3%."
# 
# With that in mind, let's find out how we can profile code and find out if optimization is needed. For this example, we will compare the three `describe()` approaches above. These all run fast enough not to need optimization but provide an instructive example for how to do profiling. With the magic functions `%timeit` and `%%timeit` we can time a line or an entire cell, respectively.

# In[15]:


get_ipython().run_cell_magic('timeit', '', "iris.select_dtypes('object').describe()")


# In[16]:


get_ipython().run_cell_magic('timeit', '', "iris.describe(include='object')")


# In[17]:


get_ipython().run_cell_magic('timeit', '', "iris['species'].describe()")


# From this benchmarking, it is clear which approach is faster and which is slowest, but we do not have a clear idea why.

# In[18]:


get_ipython().run_cell_magic('prun', '-l 10 -s cumulative # show the top 7 lines only', "iris.select_dtypes('object').describe()")


# The names in parentheses indicates which function was called and the cumtime column is the cumulative time spent inside that function. This table has all the detailed timings, but it makes it difficult to get an overview of the call stack hierarcy, e.g. why was `select_dtypes()` called twice? To explore this further we can use `%%snakeviz` (this magic funtion comes from the third party module with the same name).

# In[19]:


get_ipython().run_line_magic('load_ext', 'snakeviz')


# In[20]:


get_ipython().run_cell_magic('snakeviz', '-t # -t is to open in new tab which is necessary in JupyterLab', "iris.select_dtypes('object').describe()")


# In the graph, we can see the same information as with `%%prun`, but see which functions were called downstream of other functions. It is now clear that `select_dtypes()` was called once from within the `describe()` function in addition to us explicitly invoking the function.

# In[21]:


get_ipython().run_cell_magic('snakeviz', '-t', "iris.describe(include='object')")


# Here, there is only one invocation of `select_dtypes()` as expected.

# In[22]:


get_ipython().run_cell_magic('snakeviz', '-t', "iris['species'].describe()")


# Interestingly, we can see that this cell goes straight to the `describe_1d()` internal function, without having to call `select_dtypes()` and `concat()`, which saves a notable chunk of time.
# 
# Finally, we can inspect the source code to confirm the observed behavior. Any function method followed by `?` will show the docstring for that function and appending `??` will show the complete source code.

# In[23]:


get_ipython().run_line_magic('pinfo2', 'iris.describe')


# After a bit of argument checking and setting up helper functions, we get to the main part of the function. Here we can see that, if teh dimensions of the input data i 1, it will directly return , that is what happened when we passed the series. By defult it will use seelct dtypes for numbers and only switched if there were no columns detected, maybe more eff the other way around. Last if we explicitly pass select dtypes.
# 
# For long outputs or inputs it can be a good idea to collapse them to reduce the scrolling needed to navigate the notebook. This can be done by clicking the blue vertical bars to the left. Currently, collapsed input cells are still being executed, but in `JupyterLab` 1.0, they will be skipped. 
# 
# We now have a good intuition of what is happening when `describe()` is called on different input data.

# # Pandas styling options
# 
# For HTML output, such as Jupyter Notebooks, dataframes can use the `style` attribute.

# In[24]:


iris.corr()


# In[25]:


iris.corr().style.set_precision(3)


# Note that we only changed the precision of the *displayed* numbers. The actual values in the dataframe remain the same. The styling attribute can also change the color of the background and foreground of each cell, e.g. to highlight the max values.

# In[26]:


iris.corr().style.highlight_max()


# It is also possible to create a heatmap.

# In[27]:


iris.corr().style.background_gradient()


# The style methods are configurable via parameters just like other methods.

# In[28]:


iris.corr().style.background_gradient(cmap='Greens')


# This might remind you of conditional formatting in spreadsheet software and the stylized output can actually be exported and opened in a spreadsheet program.

# In[29]:


iris.corr().style.background_gradient(cmap='Greens').to_excel('style-test.xlsx')


# It is also possible to append the `render()` method to output the HTML, which can then be written to file.

# # DataFrame aggregations
# 
# Aggregation functions can be specified in many different ways in pandas. From highly optimized built-in functions to highly flexible arbitrary functions. If the functionality you need is available as a DataFrame method, use it. These methods tend to have their most time consuming internals written in C and thus performs very well.

# In[30]:


iris.mean()


# `agg()` is a different interface to the built-in methods, which allows for multiple functions to be past in the same call.

# In[31]:


iris.agg('mean')


# In[32]:


iris.agg(['mean', 'median'])


# If we want to use a function that is not available through pandas, we can use apply.

# While the built in aggregation methods automatically drop non-numerical values, apply does not. Instead, an error is thrown with non-numerical cols.
# Throws an error
iris.apply(np.mean)
# We could drop the string columns if there are just a few and we know which.

# In[33]:


iris.drop(columns='species').apply(np.mean)


# If there are many, it is easier to use `.select_dtypes()`.

# In[34]:


iris_num = iris.select_dtypes('number')
iris_num.apply(np.mean)


# ## User-defined functions
# 
# ### Named functions
# 
# Apply works with any function, including those you write youself.

# In[35]:


def add_one(x):
    return x + 1

add_one(5)


# In[36]:


iris_num.apply(add_one)


# In[37]:


iris_num.apply(add_one) - iris_num


# ### Unnamed lambda functions
# 
# Lambda functions can be used without being named, so they are effective for throwaway functions that you are likely to use only once.

# In[38]:


(lambda x: x + 1)(5)


# Lambda functions can be assigned to a variable name if so desired. This looks more like the standard syntax for a function definition, but lambda functions are rarely used like this.

# In[39]:


my_lam = lambda x: x + 1

my_lam(5)


# Just as with named functions, there is nothing special with the letter `x`, it is just a variable name and you can call it whatever you prefer.

# In[40]:


(lambda a_descriptive_name: a_descriptive_name + 1)(5)


# Unnamed lambda functions can be used together with apply to create any transformation to the dataframe values.

# In[41]:


iris_num.apply(lambda x: x + 1)


# We can check if they are correct by surrounding with parentheses and asser equality.

# In[42]:


iris_num.apply(lambda x: x + 1) == iris_num.apply(add_one)


# In[43]:


(iris_num.apply(lambda x: x + 1) == iris_num.apply(add_one)).all()


# A better way to assert that two dataframes are equal is to use the `assert_frame_equal()` from `pandas.testing`.

# In[44]:


# This will throw a detailed error if the assert does not pass.
pd.testing.assert_frame_equal(
    iris_num.apply(lambda x: x + 1),
    iris_num.apply(add_one))


# ### Row and column wise aggregations
# 
# By default, aggregation methods are applied column-wise, but can be set to work row-wise instead.

# In[45]:


# The row with the highest value in for each column.
iris_num.idxmax()


# In[46]:


# The column with the highest value in for each row.
iris_num.idxmax(axis=1)


# Sepal length seems to be the highest value for all rows,which we can confirm with `value_counts()`.

# In[47]:


iris_num.idxmax(axis=1).value_counts()


# Be careful when using apply to iterate over rows. This operation is very inefficient and there is often a another solution that takes advantage of the optimized pandas functions to create significant speedups.

# In[48]:


get_ipython().run_cell_magic('timeit', '', "iris.apply(lambda x: x['sepal_length'] + x['sepal_width'], axis=1)")


# In[49]:


get_ipython().run_cell_magic('timeit', '', "iris['sepal_length'] + iris['sepal_width']")


# In[50]:


pd.testing.assert_series_equal(
    iris.apply(lambda x: x['sepal_length'] + x['sepal_width'], axis=1),
    iris['sepal_length'] + iris['sepal_width'])


# # Working with categorical data

# In[51]:


titanic = pd.read_csv('titanic.csv')
# titanic.columns = titanic.columns.str.lower()
# titanic['pclass'] = titanic['pclass'].map({3:'3rd', 2:'2nd', 1:'1st'})
# titanic.to_csv('titanic.csv', index=False)
titanic


# Some of these columns I will not touch, so we're dropping them to fit the df on the screen.
titanic = titanic.drop(columns=['sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked'])
titanic
# In[53]:


titanic.info()


# How should we interpret the `+` sign under memory usage? In the docstring for `info()`, there is one option that affects memory usage, let's try it.

# In[54]:


titanic.info(memory_usage='deep')


# What happened? Why is the memory usage listed as around six times what we saw previously? The `info()` method's docstring explains why:
# 
# > Without deep introspection a memory estimation is made based in column dtype and number of rows assuming values consume the same memory amount for corresponding dtypes. With deep memory introspection, a real memory usage calculation is performed at the cost of computational resources.
# 
# So deep memory introspection shows the real memory usage, but it is still a bit cryptic what part of the dataframe's size was hidden previously. To find this out, it is helpful to understand that pandas dataframes essentially consist of numpy arrays held together with the pandas dataframe block manager. Knowing that, it would be interesting to inspect whether any of the columns (the separate numpy arrays) report different size measures with and without deep memory introspection. Instead of the more general `info()` method, we can use one specific to memory usage to find this out.

# In[55]:


titanic.memory_usage()


# In[56]:


titanic.memory_usage(deep=True)


# From this, it is clear that it is the `Name` and `Sex` columns that change, everything else remains the same. To understand what is happening, we first need to know that a numpy array is stored in the computer's memory as a contiguous (uninterupted) segment. This is one of the reasons why numpy is so fast, it only needs to find the start of the array and then access a sequential length from the start point instead of trying to look up every single object (which is how a lists work in Python).
# 
# ![](./img/array_vs_list.png)
# 
# [Image source](https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html)

# However, in order for numpy to store objects sequentially in memory, it needs to allocate a certain number of bits for each object. For example, to store a binary value, only one bit is required which can be either zero or one. To store integers, it is however many bits are needed to count up to that integer, e.g. two bits for the number 3 (`11` in binary), three bits for the number 4 (`100` in binary).
# 
# ![](img/binary-count.png)
# 
# [Image source](https://en.wikipedia.org/wiki/Binary-coded_decimal)

# This is fine for integers (up to a certain size) or floats (up to a certain precision), but with strings of variable length (and more complex object such as lists and dictionaries), numpy cannot fit them into the same sized chunks in an effective manner (strings of fixed length would technically work fine) and the actual string object is stored outside the array. So what is inside the array? Just a reference (also called a pointer) to where in memory the actual object is stored and these references are of a fixed size:
# 
# ![](./img/int-vs-pointer-memory-lookup.png)
# 
# [Image source](https://stackoverflow.com/questions/21018654/strings-in-a-dataframe-but-dtype-is-object/21020411#21020411)

# What happens when we specify to use the deep memory introspection is that pandas finds and calculates the size of each of the objects in memory. With the shallow introspection, it simply reports the values of the references that are actually stored in the array (and by default these are the same size as the stored integers and floats). 
# 
# Note that memory usage is not the same as disk usage. Objects can take up additional space in memory depending on how they are constructed.

# In[57]:


ls -lh titanic.csv


# For columns with a unique string for each row, there is currently no way around storing these as the object dtype. This is the case for the Name column in the titanic dataset. However, columns with repeating strings can preferentially be treated as categoricals, which both reduces memory usage and enables additional functionality. For example, for the `Sex` column, it is inefficient to store `'Male'` and `Female` for each row, especially taking into account the storage limitations mentioned above. Instead, it would be beneficial to store and integer for each row, and then have a separate dictionary that translates these integer into their respective strings (which are only stored once).

# `Categorical` can be used to convert a string-based object column into categorical values.

# In[58]:


pd.Categorical(titanic['sex'])


# In[59]:


titanic['sex'] = pd.Categorical(titanic['sex'])


# In[60]:


titanic.dtypes


# The dtype has now changed to `category`.

# In[61]:


titanic.memory_usage(deep=True)


# Now the `Sex` column takes up 50x less space in memory. It actually even takes up less space than the other integer columns, how is that possible? The answer is that when storing integers, Pandas by default uses 64-bit precision to allow for large numbers to be stored (and added to the dataframe without making a new copy). When creating the categorical series, pandas uses the lowest needed precision (`int8` in this case) since it is unlikely that many new categories will be added.

# In[62]:


titanic['sex'].cat.codes


# Note that if we try to store an object with unique strings as a category, we actually *increase* the memory usage, because we are still storing all the unique strings once in the dictionary, and on top of that we have added a unique number for each string.

# In[63]:


titanic['cat_name'] = pd.Categorical(titanic['name'])
titanic.memory_usage(deep=True)


# In addition to memory savings, categories are beneficial for certain types of operations. In the titanic dataset, there are a few more variables that does not have the correct data type.

# In[64]:


titanic


# `survived` and `pclass` are not numerical variables, they are categorical. `survived` can be stored as a boolean variable, which takes exactly one byte per row.

# In[65]:


titanic['survived'] = titanic['survived'].astype('bool')
titanic.memory_usage(deep=True)


# In[66]:


7128 / 891


# `pclass` is an ordered categorical, where first class is the highest class and third class is the lowest. Note that this is not the same as a numerical, e.g. it is non-sensical to say that second class is double first class.

# In[67]:


pd.Categorical(titanic['pclass'], categories=['3rd', '2nd', '1st'], ordered=True)


# In[68]:


titanic['pclass'] = pd.Categorical(titanic['pclass'], categories=['3rd', '2nd', '1st'], ordered=True)


# With an ordered categorical, comparisons can be made. We can get everything that is higher than third class.

# In[69]:


# Note that comparisons with string also work, but it is just comparing alphabetical order.
titanic['pclass'][titanic['pclass'] > '3rd'].value_counts()


# The order is also respected by pandas and seaborn, such as in the sort performed by default in groupby.

# In[70]:


titanic.groupby('pclass').size()


# In[71]:


titanic.groupby('pclass').describe()


# For methods that don't sort, this will not be in order, e.g. `groupby.head()`.

# In[72]:


titanic.groupby('pclass').head(2)


# `values_counts()` sorts based on value, not index.

# In[73]:


titanic['pclass'].value_counts(normalize=True)


# Which we can see if we profile it.

# In[74]:


get_ipython().run_cell_magic('prun', '-l 5', "titanic['pclass'].value_counts(normalize=True)")


# seaborn will also sort the values according to the categorical order from left to right.

# In[75]:


sns.catplot(x='pclass', y='age', data=titanic, kind='swarm')


# When `catplot()` was added for categorical plots, the `seaborn` author also added another human companion plot as an easter egg.

# In[76]:


sns.dogplot()


# # String processing

# Could use lambda and the normal python string functions.

# In[77]:


'First Last'.lower()


# In[78]:


titanic['name'].apply(lambda x: x.lower())


# Pandas has built in accessor method for many string methods so that we don't have to use lambda.

# In[79]:


titanic['name'].str.lower()


# Note that these work on Series, not dataframes. So either use on one series at a time or a dataframe with a lambda experssion.

# ## What are the longest lastnames

# In[80]:


titanic['name'].str.split(',')


# In[81]:


titanic['name'].str.split(',', expand=True)


# Can be assigned to multiple columns, or select one column with indexing.

# In[82]:


titanic[['lastname', 'firstname']] = titanic['name'].str.split(',', expand=True)
titanic


# In[83]:


titanic['lastname_length'] = titanic['lastname'].str.len()
titanic


# In[84]:


titanic.sort_values('lastname_length', ascending=False).head()


# In[85]:


# Shortcut for sorting
titanic.nlargest(5, 'lastname_length')


# In[86]:


sns.distplot(titanic['lastname_length'], bins=20)


# How many times are lastnames duplicated.

# In[87]:


titanic['lastname'].value_counts()


# In[88]:


titanic['lastname'].value_counts().value_counts()


# How can we view the duplicated ones.

# In[89]:


titanic[titanic.duplicated('lastname', keep=False)].sort_values(['lastname'])


# Duplication is often due to women being registered under their husbands name. 

# We can get an idea, by checking how many vaues include a parenthesis via the `str.contains` method.

# In[90]:


titanic.loc[titanic['name'].str.contains('\('), 'sex'].value_counts()


# It is mostly female passengers who have a parenthesis in the name. Before we proceed, a note about method chaining in pandas. Method chaining is when multiple methods are appended after each other like we did in the cell above. This can be done in one line, but often it is preferential to spread it out over multiple lines.
# 
# Note that we selected both the rows and the columns by separating them with a comma within a single call to `loc[]`. It is important to select them this way rather than using multiple chained calls or selecting columns with `[]` afterwards, since these approaches might give you a `SettingWithCopyWarning` later meaning that pandas does not know if you are operating on new (copy) dataframe or trying to change values within the original data frame.
# 
# Multi-line statements can be formatted in a few different ways in Python, e.g. we could use the explicit line continuation character `\` for a syntax similar to how R uses `%>%`.

# In[91]:


titanic     .loc[titanic['name'].str.contains('\('), 'sex']     .value_counts()


# This is perfectly fine according to the style guides and easy to read so feel free to use this style if it is easy to remember due it's similarity to pipes in `dplyr`. More common in Python is to use implicit line continuation with an open `(` to indicate the line continues below.

# In[92]:


(titanic
    .loc[titanic['name'].str.contains('\('), 'sex']
    .value_counts())


# One of the advantages here is that we don't have to remember to put a continuation character on each lines when we add new lines. The closing `)` could also be put a new line for maximal flexibility.
# 
# `value_counts()` can also create normalized counts.

# In[93]:


(titanic
    .loc[titanic['name'].str.contains('\('), 'sex']
    .value_counts(normalize=True))


# How to negate a boolean expression? If we want all the names without parenthesis we could use the `~` operator to negate (or invert) the boolean expression.

# In[94]:


(titanic
    .loc[~titanic['name'].str.contains('\('), 'sex']
    .value_counts())


# There seems to be several reasons for parenthesis in the name. The ones we want to change are the ones who have 'Mrs' and a parenthesis in the name. To combine boolean expression we can surround each one with `()` and then use the bitwise comparison operators: `&` for "and", `|` for "or".  These compare each row for the two separate boolean expressions and outputs a single boolean matrix.

# In[95]:


(titanic
    .loc[(titanic['name'].str.contains('\('))
        & (titanic['name'].str.contains('Mrs'))
        , 'sex']  # On a separate row for clarity
    .value_counts())


# Dropped all male and 4 female passengers. Which females were dropped?

# In[96]:


(titanic
    .loc[(titanic['name'].str.contains('\('))
        & (~titanic['name'].str.contains('Mrs'))
        & (titanic['sex'] == 'female')
        , 'name'])


# Even more precisely, we only want to keep the ones with a last and first name in the parentheses. We can use the fact that these seems to be separated by a space. `str.contains` accepts regular expression, and we will use `.*` which means "match any characters", similar to how wildcards work in the unix shell.

# In[97]:


# Explain regex above
# titanic.loc[(titanic['name'].str.contains('\(')) & (titanic['sex'] == 'female'), 'sex'].value_counts()
(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'sex']
    .value_counts())


# From these passengers, we can extract the name in the parenthesis using `str.partition`, which we saw previously.

# In[98]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2])


# In[99]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str.partition(')')[0])


# In this case I could also have used string indexing to strip the last character, but this would give us issues if there are spaces at the end.
(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str[:-1]
)
# There is a more advanced way of getting this with regex directly, using a matching group to find anything in the parenthesis.

# In[100]:


# %%timeit
(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.extract("\((.+)\)"))


# The two way partition method is just fine, and regex can feel a bit magical sometime, but it is good to know about if you end up working a lot with strings or need to extract complicated patterns.
# 
# Now lets get just the last names from this column and assign them back to the dataframe.

# In[101]:


(titanic
    .loc[titanic['name'].str.contains('Mrs.*\(.* .*\)'), 'name']
    .str.partition('(')[2]
    .str.partition(')')[0]
    .str.rsplit(n=1, expand=True))


# All the lastnames without parenthsis will remain the same and we will only overwrite those that match the criteria we set up above.

# In[102]:


titanic['real_last'] = titanic['lastname']
# Separate for clarity
mrs_paren = titanic['name'].str.contains('Mrs.*\(.* .*\)')
titanic.loc[mrs_paren, 'real_last'] = (titanic
    .loc[mrs_paren, 'name']
    .str.partition('(')[2]
    .str.partition(')')[0]
    .str.rsplit(n=1, expand=True)[1]
)


# In[103]:


titanic


# We can see that this seems to have worked. Passengers with "Mrs" in their name have a new value under `real_last`, but others don't.

# In[104]:


titanic['lastname'].value_counts().value_counts()


# In[105]:


titanic['real_last'].value_counts().value_counts()


# The value counts also changed slightly, we can see a more detailed comparison with a paired barplot of counts.

# In[106]:


titanic['real_last_length'] = titanic['real_last'].str.len()


# In[107]:


molten_titanic = (titanic
    .loc[titanic['sex'] == 'female']
    .melt(value_vars=['lastname_length', 'real_last_length']))
sns.catplot(x='value', hue='variable', data=molten_titanic, kind='count')
molten_titanic.groupby('variable').agg(['mean', 'median'])


# ## Extras

# In[108]:


# import re
# titanic.rename(columns=lambda x: re.sub('(?!^)([A-Z]+)', r'_\1', x).lower())
# nbdime command line?
# git add -p?

# Every command in this notebook
%hist# Grep through all history
%hist -g mrs_paren
# In[109]:


# For easier version control, or use jupytext
get_ipython().system('jupyter-nbconvert mds-seminar-apply-cat-str.ipynb --to python')


# # JupyterLab 1.0
# 
# - Search and replace in the entire notebook!
# - Code folding in text files and notebooks, e.g. to collapse a loop.
# - Insert colored banners with HTML divs (Bootstrap-like functionality).
# - Status bar.
# - Printing directly from the notebook.
# - Inspector renamed Interactive Help.
# - Menu item to enable the extension manager.
# - Rulers to keep an eye on code length.
# - Scroll past the end of notebooks.
