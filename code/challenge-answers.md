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

3. surveys.info

    ```
    1. DataFrame
    2. rows: 34786,  columns: 13
    3. some values are NAs (they are missing), for example an animal escaped before it was measured, we will go more into NAs later
    ```

4. Subset data frame

    ```
    1. surveys_200 = surveys.iloc[199]
    2. surveys_last = surveys.iloc[surveys.shape[0]-1]
    3. surveys_last <- surveys.iloc[-1]
    ```

5. Filter data frame

    ```
    1. surveys.loc[surveys['year'] < 1995, ['year', 'sex', 'weight']]
    ```

6. Hindfoot half

    ```
    1.
    surveys_hh = surveys.dropna()[['species_id', 'hindfoot_length']]
    surveys_hh['hindfoot_half'] = surveys_hh['hindfoot_length'] / 2 
    surveys_hh = surveys_hh.loc[surveys_hh['hindfoot_half'] < 30]
    ```

2.1. Groupby

```
1. surveys.grouby('species')['hindfoot_length'].agg(['mean', 'min', 'max'])
```


```
2.
max_weight_indices = surveys.groupby("year")['weight'].idxmax()
surveys[['year', 'genus', 'species', 'weight']].iloc[max_weight_indices]
```

2.2. Size

```
1.
surveys.groupby(['plot_type']).size()
```

```
2.
surveys.groupby(['year', 'plot_type']).size().nlargest(3)
```

2.3. Faceting Part 1
```
1.
sns.factorplot(x='month', hue='sex', data=surveys_common, col='species',
               row='plot_type', kind='count', margin_titles=True) 

This can be overwhelming and it often a good idea to break things down into
smaller chunks. Especially when presenting the data to others, the most
effective communication is usually to break the messaged down into focused
topics.

```

2.4. Faceting Part 2


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
