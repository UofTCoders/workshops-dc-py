2. Discussion.
- Boxplot shows a few statistics of the distribution (medain, quartiles, whiskers).
- Violin plot approximates the distribution with a smoothened histogram, which can be more informative for big data set, but misleading for small datasets. 
- Swarmplot shows every observations which is great for small to medium data set, but becomes clusttered on large data sets.
- Combinations of these plots can bring the best of both worlds as we will see soon!

3.

First part
```
ax = sns.violinplot(x='species', y='sepal_length', data=iris, color='lightgrey', inner=None)
ax = sns.swarmplot(x='species', y='sepal_length', data=iris) 
ax.set_ylabel('Sepal Length', fontsize=14)
ax.set_xlabel('')
sns.despine()
```

Second part
```
ax = sns.boxplot(x='species', y='sepal_length', data=iris, color='lightgrey')
ax = sns.stripplot(x='species', y='sepal_length', data=iris, jitter=True) 
ax.set_ylabel('Sepal Length', fontsize=14)
ax.set_xlabel('')
sns.despine()
```

4. Ask them to share which color they chose! Show how to search online for "matplolib markers" which should take them to this page https://matplotlib.org/api/markers_api.html listing all possible markers.

```
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.scatter(x, y, marker='<', color='rebeccapurple')
ax2.plot(x, y, linestyle='dotted', color='salmon')

ax1.set_title('Scatter plot')
ax2.set_title('Line plot')
fig.tight_layout()
```

5. Save one of the previous figures and upload it on that website, or ask the students to share one of their own.