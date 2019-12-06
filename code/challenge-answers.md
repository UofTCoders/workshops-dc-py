# Lecture 1 - Intro to programming

1. Tuples

    ```
    1. Error because tuple are immutable
    2. tuple
    ```

2. Dictionary reassignment

    ```
    1. rev
    2. rev[2] = 'apple-sauce'
    3. rev
    ```
 
# Lecture 2 - Data analysis in Python

1. surveys.info

    ```
    1. DataFrame
    2. rows: 34786,  columns: 13
    3. some values are NAs (they are missing), for example an animal escaped before it was measured, we will go more into NAs later
    ```

2. Subset data frame

    ```
    1. surveys_200 = surveys.iloc[199]
    2. surveys_last = surveys.iloc[surveys.shape[0]-1]
    3. surveys_last <- surveys.iloc[-1]
    ```

3. Filter data frame

    ```
    1. surveys.loc[surveys['year'] < 1995, ['year', 'sex', 'weight']]
    ```

4. Hindfoot half

    ```
    1.
    surveys_hh = surveys[['species_id', 'hindfoot_length']].dropna()
    2.
    surveys_hh['hindfoot_half'] = surveys_hh['hindfoot_length'] / 2 
    surveys_hh = surveys_hh.loc[surveys_hh['hindfoot_half'] < 30]
    3.
    surveys_hh.shape == (31436, 3)
    ```
5. Groupby

    ```
    1. surveys.grouby('species')['hindfoot_length'].agg(['mean', 'min', 'max'])
    ```

6. Size

    ```
    1. surveys.groupby(['plot_type']).size()
    2. surveys.groupby(['year', 'plot_type']).size().nlargest(3)
    ```

# Lecture 3 - Exploratory visualizations


1. Quantitative vs categorical

    ```
    1. # The split parameter has to be removed since it is violin plot specific
    sns.factorplot(x='weight', y='plot_type', hue='sex', data=surveys_common, 
                      col='species', col_wrap=2, kind='box', aspect=1.4)
    2. sns.factorplot(x='month', hue='sex', data=surveys_common, col='species',
                      row='plot_type', kind='count', margin_titles=True) 
    ```
This can be overwhelming and it often a good idea to break things down into
smaller chunks. Especially when presenting the data to others, the most
effective communication is usually to break the messaged down into focused
topics.

2. Quantitative vs quantitative

    ```
    1. The sex column is mapped to the hue since we are interested in comparing the difference between males and females within the same species. The mapping would have been the opposite if the goal was to compare the impacts of different species within each sex.
    sns.lmplot(x='weight', y='hindfoot_length', hue='sex',
               col='species', col_wrap=3, data=surveys_hf_wt, size=3.5,
               fit_reg=False, scatter_kws={'s': 12, 'alpha':0.6})
    ```

3. Faceting Part 2


    ```
    1.
    weights_year = (
        surveys
         .groupby('year')['weight']
         .mean()
         .reset_index()
    )

    g = sns.FacetGrid(data=weights_year)
    g.map(plt.plot, 'year', 'weight') 
    ```

    ```
    2.
    weights_species_year = (
        surveys
         .groupby(['year', 'species'])['weight']
         .mean().reset_index()
         .dropna(subset=['weight'])
    )

    g = sns.FacetGrid(hue='species', data=weights_species_year, size=2.5,
                     aspect=1.3, col='species', col_wrap=4)
    g.map(plt.plot, 'year', 'weight')
    ```

# Lecture 4 - Reshaping data frames and advanced vizualization

1.

    ```
    1.
    
    species_year_wide = surveys.pivot_table(index='plot_type', columns='year', values='species', aggfunc='nunique')

    2.
    species_year_long = species_year_wide.reset_index().melt(id_vars='plot_type')
    ```

2. Ask them to share which color they chose! Show how to search online for "matplolib markers" which should take them to this page https://matplotlib.org/api/markers_api.html listing all possible markers.

    ```
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.scatter(x, y, marker='<', color='rebeccapurple')
    ax2.plot(x, y, linestyle='dotted', color='salmon')

    ax1.set_title('Scatter plot')
    ax2.set_title('Line plot')
    fig.tight_layout()

    ```

3. Discussion.

    - Boxplot shows a few distribution statistics (medain, quartiles, whiskers).
    - Violin plot approximates the distribution with a smoothened histogram, which can be more informative for big data set, but misleading for small datasets. 
    - Swarmplot shows every observations which is great for small to medium data set, but becomes clusttered on large data sets.
    - Combinations of these plots can bring the best of both worlds as we will see soon!

4.

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

5. Save one of the previous figures and upload it on that website, or ask the students to share one of their own.
