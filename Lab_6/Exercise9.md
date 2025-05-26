---
title: Bivariate Statistics
subtitle: Foundations of Statistical Analysis in Python
abstract: This notebook explores bivariate relationships through linear correlations, highlighting their strengths and limitations. Practical examples and visualizations are provided to help users understand and apply these statistical concepts effectively.
author:
  - name: Karol Flisikowski
    affiliations: 
      - Gdansk University of Technology
      - Chongqing Technology and Business University
    orcid: 0000-0002-4160-1297
    email: karol@ctbu.edu.cn
date: 2025-05-03
---

## Goals of this lecture

There are many ways to *describe* a distribution. 

Here we will discuss:
- Measurement of the relationship between distributions using **linear, rank correlations**.
- Measurement of the relationship between qualitative variables using **contingency**.

## Importing relevant libraries


```python
%pip install seaborn

```


```python
%pip install scipy
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: scipy in c:\users\admin\appdata\roaming\python\python311\site-packages (1.15.3)
    Requirement already satisfied: numpy<2.5,>=1.23.5 in c:\users\admin\appdata\roaming\python\python311\site-packages (from scipy) (1.24.1)
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip available: 22.3.1 -> 25.1.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ### importing seaborn
import pandas as pd
import scipy.stats as ss
```


```python
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
```


```python
import pandas as pd
df_pokemon = pd.read_csv("data/pokemon.csv")
```

## Describing *bivariate* data with correlations

- So far, we've been focusing on *univariate data*: a single distribution.
- What if we want to describe how *two distributions* relate to each other?
   - For today, we'll focus on *continuous distributions*.

### Bivariate relationships: `height`

- A classic example of **continuous bivariate data** is the `height` of a `parent` and `child`.  
- [These data were famously collected by Karl Pearson](https://www.kaggle.com/datasets/abhilash04/fathersandsonheight).


```python
df_height = pd.read_csv("data/wrangling/height.csv")
df_height.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Father</th>
      <th>Son</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>59.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63.3</td>
      <td>63.2</td>
    </tr>
  </tbody>
</table>
</div>



#### Plotting Pearson's height data


```python
sns.scatterplot(data = df_height, x = "Father", y = "Son", alpha = .5)
```


    
![png](output_12_0.png)
    


### Introducing linear correlations

> A **correlation coefficient** is a number between $[–1, 1]$ that describes the relationship between a pair of variables.

Specifically, **Pearson's correlation coefficient** (or Pearson's $r$) describes a (presumed) *linear* relationship.

Two key properties:

- **Sign**: whether a relationship is positive (+) or negative (–).  
- **Magnitude**: the strength of the linear relationship.

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

Where:
- $r$ - Pearson correlation coefficient
- $x_i$, $y_i$ - values of the variables
- $\bar{x}$, $\bar{y}$ - arithmetic means
- $n$ - number of observations

Pearson's correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. Its value ranges from -1 to 1:
- 1 → perfect positive linear correlation
- 0 → no linear correlation
- -1 → perfect negative linear correlation

This coefficient does not tell about nonlinear correlations and is sensitive to outliers.

### Calculating Pearson's $r$ with `scipy`

`scipy.stats` has a function called `pearsonr`, which will calculate this relationship for you.

Returns two numbers:

- $r$: the correlation coefficent.  
- $p$: the **p-value** of this correlation coefficient, i.e., whether it's *significantly different* from `0`.


```python
ss.pearsonr(df_height['Father'], df_height['Son'])
```




    PearsonRResult(statistic=0.5011626808075912, pvalue=1.272927574366214e-69)



#### Check-in

Using `scipy.stats.pearsonr` (here, `ss.pearsonr`), calculate Pearson's $r$ for the relationship between the `Attack` and `Defense` of Pokemon.

- Is this relationship positive or negative?  
- How strong is this relationship?


```python
r, p = ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
print(f"Pearson's r: {r}, p-value: {p}")

# the relationship is positive since higher attack comes with higher def

# the correlation is strong since 0.758 is between 0.7 and 1.0

# the p-value is very small so the relationship is statistically significant.
```

    Pearson's r: 0.4386870551184896, p-value: 5.858479864289521e-39
    

#### Solution


```python
ss.pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
```




    PearsonRResult(statistic=0.4386870551184896, pvalue=5.858479864289521e-39)



#### Check-in

Pearson'r $r$ measures the *linear correlation* between two variables. Can anyone think of potential limitations to this approach?

### Limitations of Pearson's $r$

- Pearson's $r$ *presumes* a linear relationship and tries to quantify its strength and direction.  
- But many relationships are **non-linear**!  
- Unless we visualize our data, relying only on Pearson'r $r$ could mislead us.

#### Non-linear data where $r = 0$


```python
x = np.arange(1, 40)
y = np.sin(x)
p = sns.lineplot(x = x, y = y)
```


    
![png](output_25_0.png)
    



```python
### r is close to 0, despite there being a clear relationship!
ss.pearsonr(x, y)
```




    (-0.04067793461845844, 0.8057827185936633)



#### When $r$ is invariant to the real relationship

All these datasets have roughly the same **correlation coefficient**.


```python
df_anscombe = sns.load_dataset("anscombe")
sns.relplot(data = df_anscombe, x = "x", y = "y", col = "dataset");
```


    
![png](output_28_0.png)
    



```python
# Compute correlation matrix
corr = df_pokemon.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](output_29_0.png)
    


## Rank Correlations

Rank correlations are measures of the strength and direction of a monotonic (increasing or decreasing) relationship between two variables. Instead of numerical values, they use ranks, i.e., positions in an ordered set.

They are less sensitive to outliers and do not require linearity (unlike Pearson's correlation).

### Types of Rank Correlations

1. $ρ$ (rho) **Spearman's**
- Based on the ranks of the data.
- Value: from –1 to 1.
- Works well for monotonic but non-linear relationships.

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
- $d_i$ – differences between the ranks of observations,
- $n$ – number of observations.

2. $τ$ (tau) **Kendall's**
- Measures the number of concordant vs. discordant pairs.
- More conservative than Spearman's – often yields smaller values.
- Also ranges from –1 to 1.

$$
\tau = \frac{(C - D)}{\frac{1}{2}n(n - 1)}
$$

Where:
- $τ$ — Kendall's correlation coefficient,
- $C$ — number of concordant pairs,
- $D$ — number of discordant pairs,
- $n$ — number of observations,
- $\frac{1}{2}n(n - 1)$ — total number of possible pairs of observations.

What are concordant and discordant pairs?
- Concordant pair: if $x_i$ < $x_j$ and $y_i$ < $y_j$, or $x_i$ > $x_j$ and $y_i$ > $y_j$.
- Discordant pair: if $x_i$ < $x_j$ and $y_i$ > $y_j$, or $x_i$ > $x_j$ and $y_i$ < $y_j$.

### When to use rank correlations?
- When the data are not normally distributed.
- When you suspect a non-linear but monotonic relationship.
- When you have rank correlations, such as grades, ranking, preference level.

| Correlation type | Description | When to use |
|------------------|-----------------------------------------------------|----------------------------------------|
| Spearman's (ρ) | Monotonic correlation, based on ranks | When data are nonlinear or have outliers |
| Kendall's (τ) | Counts the proportion of congruent and incongruent pairs | When robustness to ties is important |

### Interpretation of correlation values

| Range of values | Correlation interpretation |
|------------------|----------------------------------|
| 0.8 - 1.0 | very strong positive |
| 0.6 - 0.8 | strong positive |
| 0.4 - 0.6 | moderate positive |
| 0.2 - 0.4 | weak positive |
| 0.0 - 0.2 | very weak or no correlation |
| < 0 | similarly - negative correlation |


```python
# Compute Kendall rank correlation
corr_kendall = df_pokemon.corr(method='kendall', numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr, 
            annot=True,         # Show correlation coefficients
            fmt=".2f",          # Format for coefficients
            cmap="coolwarm",    # Color palette
            vmin=-1, vmax=1,    # Fixed scale
            square=True,        # Make cells square
            linewidths=0.5,     # Line width between cells
            cbar_kws={"shrink": .75})  # Colorbar shrink

# Title and layout
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()

# Show plot
plt.show()
```


    
![png](output_34_0.png)
    


### Comparison of Correlation Coefficients

| Property                | Pearson (r)                   | Spearman (ρ)                        | Kendall (τ)                          |
|-------------------------|-------------------------------|--------------------------------------|---------------------------------------|
| What it measures?       | Linear relationship           | Monotonic relationship (based on ranks) | Monotonic relationship (based on pairs) |
| Data type               | Quantitative, normal distribution | Ranks or ordinal/quantitative data  | Ranks or ordinal/quantitative data   |
| Sensitivity to outliers | High                          | Lower                               | Low                                   |
| Value range             | –1 to 1                       | –1 to 1                             | –1 to 1                               |
| Requires linearity      | Yes                           | No                                  | No                                    |
| Robustness to ties      | Low                           | Medium                              | High                                  |
| Interpretation          | Strength and direction of linear relationship | Strength and direction of monotonic relationship | Proportion of concordant vs discordant pairs |
| Significance test       | Yes (`scipy.stats.pearsonr`)  | Yes (`spearmanr`)                   | Yes (`kendalltau`)                   |

Brief summary:
- Pearson - best when the data are normal and the relationship is linear.
- Spearman - works better for non-linear monotonic relationships.
- Kendall - more conservative, often used in social research, less sensitive to small changes in data.

### Your Turn

For the Pokemon dataset, find the pairs of variables that are most appropriate for using one of the quantitative correlation measures. Calculate them, then visualize them.


```python
from scipy.stats import pearsonr, spearmanr, kendalltau

# Compute Pearson correlation matrix
corr_pearson = df_pokemon.corr(method='pearson', numeric_only=True)

# 1. attack vs def pearson
r, p_r = pearsonr(df_pokemon['Attack'], df_pokemon['Defense'])
print(f"Pearson r (Attack vs Defense): {r}, p-value: {p_r}")

sns.scatterplot(data=df_pokemon, x='Attack', y='Defense')
plt.title("Attack vs Defense (Pearson)")
plt.show()

# 2. hp vs total — use spearmann
rho, p_rho = spearmanr(df_pokemon['HP'], df_pokemon['Total'])
print(f"Spearman rho (HP vs Total): {rho}, p-value: {p_rho}")

sns.scatterplot(data=df_pokemon, x='HP', y='Total')
plt.title("HP vs Total (Spearman)")
plt.show()

# 3. speed vs sp. atk — kendall
tau, p_tau = kendalltau(df_pokemon['Speed'], df_pokemon['Sp. Atk'])
print(f"Kendall tau (Speed vs Sp. Atk): {tau:.2f}, p-value: {p_tau:.4f}")

sns.scatterplot(data=df_pokemon, x='Speed', y='Sp. Atk')
plt.title("Speed vs Sp. Atk (Kendall)")
plt.show()
```

    Pearson r (Attack vs Defense): 0.4386870551184896, p-value: 5.858479864289521e-39
    


    
![png](plots/output_38_1.png)
    


    Spearman rho (HP vs Total): 0.712736221497133, p-value: 4.942180397408517e-125
    


    
![png](plots/output_38_3.png)
    


    Kendall tau (Speed vs Sp. Atk): 0.33, p-value: 0.0000
    


    
![png](plots/output_38_5.png)
    


## Correlation of Qualitative Variables

A categorical variable is one that takes descriptive values ​​that represent categories—e.g. Pokémon type (Fire, Water, Grass), gender, status (Legendary vs. Normal), etc.

Such variables cannot be analyzed directly using correlation methods for numbers (Pearson, Spearman, Kendall). Other techniques are used instead.

### Contingency Table

A contingency table is a special cross-tabulation table that shows the frequency (i.e., the number of cases) for all possible combinations of two categorical variables.

It is a fundamental tool for analyzing relationships between qualitative features.

#### Chi-Square Test of Independence

The Chi-Square test checks whether there is a statistically significant relationship between two categorical variables.

Concept:

We compare:
- observed values (from the contingency table),
- with expected values, assuming the variables are independent.

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Where:
- $O_{ij}$ – observed count in cell ($i$, $j$),
- $E_{ij}$ – expected count in cell ($i$, $j$), assuming independence.

### Example: Calculating Expected Values and Chi-Square Statistic in Python

Here’s how you can calculate the **expected values** and **Chi-Square statistic (χ²)** step by step using Python.

---

#### Step 1: Create the Observed Contingency Table
We will use the Pokémon example:

| Type 1 | Legendary = False | Legendary = True | Total |
|--------|-------------------|------------------|-------|
| Fire   | 18                | 5                | 23    |
| Water  | 25                | 3                | 28    |
| Grass  | 20                | 2                | 22    |
| Total  | 63                | 10               | 73    |


```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Observed values (contingency table)
observed = np.array([
    [18, 5],  # Fire
    [25, 3],  # Water
    [20, 2]   # Grass
])

# Convert to DataFrame for better visualization
observed_df = pd.DataFrame(
    observed,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("Observed Table:")
print(observed_df)
```

    Observed Table:
           Legendary = False  Legendary = True
    Fire                  18                 5
    Water                 25                 3
    Grass                 20                 2
    

Step 2: Calculate Expected Values
The expected values are calculated using the formula:

$$ E_{ij} = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}} $$

You can calculate this manually or use scipy.stats.chi2_contingency, which automatically computes the expected values.


```python
# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(observed)

# Convert expected values to DataFrame for better visualization
expected_df = pd.DataFrame(
    expected,
    columns=["Legendary = False", "Legendary = True"],
    index=["Fire", "Water", "Grass"]
)
print("\nExpected Table:")
print(expected_df)
```

    
    Expected Table:
           Legendary = False  Legendary = True
    Fire           19.849315          3.150685
    Water          24.164384          3.835616
    Grass          18.986301          3.013699
    

Step 3: Calculate the Chi-Square Statistic
The Chi-Square statistic is calculated using the formula:

$$ \chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

This is done automatically by scipy.stats.chi2_contingency, but you can also calculate it manually:


```python
# Manual calculation of Chi-Square statistic
chi2_manual = np.sum((observed - expected) ** 2 / expected)
print(f"\nChi-Square Statistic (manual): {chi2_manual:.4f}")
```

    
    Chi-Square Statistic (manual): 1.8638
    

Step 4: Interpret the Results
The chi2_contingency function also returns:

p-value: The probability of observing the data if the null hypothesis (independence) is true.
Degrees of Freedom (dof): Calculated as (rows - 1) * (columns - 1).


```python
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
```

    
    Chi-Square Statistic: 1.8638
    p-value: 0.3938
    Degrees of Freedom: 2
    

**Interpretation of the Chi-Square Test Result:**

| Value               | Meaning                                         |
|---------------------|-------------------------------------------------|
| High χ² value       | Large difference between observed and expected values |
| Low p-value         | Strong basis to reject the null hypothesis of independence |
| p < 0.05            | Statistically significant relationship between variables |

### Qualitative Correlations

#### Cramér's V

**Cramér's V** is a measure of the strength of association between two categorical variables. It is based on the Chi-Square test but scaled to a range of 0–1, making it easier to interpret the strength of the relationship.

$$
V = \sqrt{ \frac{\chi^2}{n \cdot (k - 1)} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows/columns) in the contingency table.

---

#### Phi Coefficient ($φ$)

Application:
- Both variables must be dichotomous (e.g., Yes/No, 0/1), meaning the table must have the smallest size of **2×2**.
- Ideal for analyzing relationships like gender vs purchase, type vs legendary.

$$
\phi = \sqrt{ \frac{\chi^2}{n} }
$$

Where:
- $\chi^2$ – Chi-Square test statistic for a 2×2 table,
- $n$ – number of observations.

---

#### Tschuprow’s T

**Tschuprow’s T** is a measure of association similar to **Cramér's V**, but it has a different scale. It is mainly used when the number of categories in the two variables differs. This is a more advanced measure applicable to a broader range of contingency tables.

$$
T = \sqrt{\frac{\chi^2}{n \cdot (k - 1)}}
$$

Where:
- $\chi^2$ – Chi-Square test statistic,
- $n$ – number of observations,
- $k$ – the smaller number of categories (rows or columns) in the contingency table.

Application: Tschuprow’s T is useful when dealing with contingency tables with varying numbers of categories in rows and columns.

---

### Summary - Qualitative Correlations

| Measure            | What it measures                                       | Application                     | Value Range     | Strength Interpretation       |
|--------------------|--------------------------------------------------------|---------------------------------|------------------|-------------------------------|
| **Cramér's V**     | Strength of association between nominal variables      | Any categories                  | 0 – 1           | 0.1–weak, 0.3–moderate, >0.5–strong |
| **Phi ($φ$)**      | Strength of association in a **2×2** table             | Two binary variables            | -1 – 1          | Similar to correlation        |
| **Tschuprow’s T**  | Strength of association, alternative to Cramér's V     | Tables with similar category counts | 0 – 1      | Less commonly used            |
| **Chi² ($χ²$)**    | Statistical test of independence                       | All categorical variables       | 0 – ∞           | Higher values indicate stronger differences |

### Example

Let's investigate whether the Pokémon's type (type_1) is affected by whether the Pokémon is legendary.

We'll use the **scipy** library.

This library already has built-in functions for calculating various qualitative correlation measures.


```python
from scipy.stats.contingency import association

# Contingency table:
ct = pd.crosstab(df_pokemon["Type 1"], df_pokemon["Legendary"])

# Calculating Cramér's V measure
V = association(ct, method="cramer") # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html#association

print(f"Cramer's V: {V}") # interpret!

```

    Cramer's V: 0.3361928228447545
    

### Your turn

What visualization would be most appropriate for presenting a quantitative, ranked, and qualitative relationship?

Try to think about which pairs of variables could have which type of analysis based on the Pokemon data.

---


```python
%pip install statsmodels
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting statsmodels
      Downloading statsmodels-0.14.4-cp311-cp311-win_amd64.whl (9.9 MB)
         ---------------------------------------- 9.9/9.9 MB 2.6 MB/s eta 0:00:00
    Requirement already satisfied: numpy<3,>=1.22.3 in c:\users\admin\appdata\roaming\python\python311\site-packages (from statsmodels) (1.24.1)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in c:\users\admin\appdata\roaming\python\python311\site-packages (from statsmodels) (1.15.3)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in c:\users\admin\appdata\roaming\python\python311\site-packages (from statsmodels) (1.5.3)
    Collecting patsy>=0.5.6
      Downloading patsy-1.0.1-py2.py3-none-any.whl (232 kB)
         -------------------------------------- 232.9/232.9 kB 7.0 MB/s eta 0:00:00
    Requirement already satisfied: packaging>=21.3 in c:\users\admin\appdata\roaming\python\python311\site-packages (from statsmodels) (23.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\admin\appdata\roaming\python\python311\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\admin\appdata\roaming\python\python311\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2022.7.1)
    Requirement already satisfied: six>=1.5 in c:\users\admin\appdata\roaming\python\python311\site-packages (from python-dateutil>=2.8.1->pandas!=2.1.0,>=1.4->statsmodels) (1.16.0)
    Installing collected packages: patsy, statsmodels
    Successfully installed patsy-1.0.1 statsmodels-0.14.4
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip available: 22.3.1 -> 25.1.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
# for presenting a quantitative relationship the approprate visualization is for example a scatterplot or a heatmap>
# quantitative relationship in pokemon dataset could be observed between, for example, Attack and Defense, or Attack and Speed.
sns.scatterplot(data=df_pokemon, x='Attack', y='Defense')
plt.title("Scatterplot: Attack vs Defense (Quantitative)")
plt.show()

# for presentig a qualitative relationship the appropriate visualization is for example a mozaik plot
# a qualitative relationship in pokemon dataset could be observed between Type 1 and Type 2.
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
mosaic(df_pokemon, ['Type 1', 'Type 2'])
plt.title("Mosaic Plot: Type 1 vs Type 2 (Qualitative)")
plt.show()

# ranked / ordinal relationships involve variables that have a meaningful order but not necessarily equal intervals
# a good visualization is a boxplot or violin plot showing distribution across rank categories.
# generation vs total stats.
sns.boxplot(data=df_pokemon, x='Generation', y='Total')
plt.title("Boxplot: Total Stats by Generation (Ranked/Ordinal)")
plt.show()
```


    
![png](plots/output_55_0.png)
    



    <Figure size 1200x800 with 0 Axes>



    
![png](plots/output_55_2.png)
    



    
![png](plots/output_55_3.png)
    


## Heatmaps for qualitative correlations


```python
# git clone https://github.com/ayanatherate/dfcorrs.git
# cd dfcorrs 
# pip install -r requirements.txt

from dfcorrs.cramersvcorr import Cramers
cram=Cramers()
# cram.corr(df_pokemon)
cram.corr(df_pokemon, plot_htmp=True)

```



## Your turn!

Load the "sales" dataset and perform the bivariate analysis together with necessary plots. Remember about to run data preprocessing before the analysis.


```python
%pip install openpyxl
```


```python
df_sales = pd.read_excel("data/sales.xlsx")
df_sales.head(5)
print(df_sales.info())
print(df_sales.describe())

# DATA PREPROCESSING

# check missing values
print(df_sales.isna().sum())

# simple imputation or drop rows with missing data
df_sales_clean = df_sales.dropna(subset=['No_of_Customers', 'Sales', 'Product_Quality'])

# convert categorical variables to strings 
df_sales_clean.loc[:, 'Store_Type'] = df_sales_clean['Store_Type'].astype(str)
df_sales_clean.loc[:, 'City_Type'] = df_sales_clean['City_Type'].astype(str)
df_sales_clean.loc[:, 'Product_Quality'] = df_sales_clean['Product_Quality'].astype(str)

# check again

print(df_sales_clean.info())

# BIVARIATE ANALYSIS

# Quantitative vs Quantitative: Sales vs No_of_Customers
sns.scatterplot(data=df_sales_clean, x='No_of_Customers', y='Sales')
plt.title("Sales vs Number of Customers")
plt.show()

# CORRELATION
r, p = pearsonr(df_sales_clean['No_of_Customers'], df_sales_clean['Sales'])
print(f"Pearson correlation between Sales and No_of_Customers: r={r}, p={p}")

# Quantitative vs Categorical: Sales by Store_Type
sns.boxplot(data=df_sales_clean, x='Store_Type', y='Sales')
plt.title("Sales Distribution by Store Type")
plt.show()

# Categorical vs Categorical: Store_Type vs Product_Quality
type_quality_counts = pd.crosstab(df_sales_clean['Store_Type'], df_sales_clean['Product_Quality'])

type_quality_counts.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Set2')
plt.title("Product Quality Distribution by Store Type")
plt.ylabel("Count")
plt.xlabel("Store Type")
plt.legend(title='Product Quality')
plt.tight_layout()
plt.show()

# Quantitative vs Ranked: Sales by City_TypE
sns.boxplot(data=df_sales_clean, x='City_Type', y='Sales')
plt.title("Sales Distribution by City Type")
plt.show()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12 entries, 0 to 11
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   Date             12 non-null     datetime64[ns]
     1   Store_Type       12 non-null     int64         
     2   City_Type        12 non-null     int64         
     3   Day_Temp         9 non-null      float64       
     4   No_of_Customers  9 non-null      float64       
     5   Sales            9 non-null      float64       
     6   Product_Quality  10 non-null     object        
    dtypes: datetime64[ns](1), float64(3), int64(2), object(1)
    memory usage: 800.0+ bytes
    None
           Store_Type  City_Type   Day_Temp  No_of_Customers        Sales
    count   12.000000  12.000000   9.000000         9.000000     9.000000
    mean     1.833333   1.833333  28.111111        99.444444  2910.222222
    std      0.834847   0.834847   3.887301         7.875772  1098.388112
    min      1.000000   1.000000  22.000000        90.000000  1254.000000
    25%      1.000000   1.000000  26.000000        94.000000  2356.000000
    50%      2.000000   2.000000  29.000000       100.000000  3112.000000
    75%      2.250000   2.250000  31.000000       104.000000  3682.000000
    max      3.000000   3.000000  33.000000       115.000000  4232.000000
    Date               0
    Store_Type         0
    City_Type          0
    Day_Temp           3
    No_of_Customers    3
    Sales              3
    Product_Quality    2
    dtype: int64
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7 entries, 0 to 11
    Data columns (total 7 columns):
     #   Column           Non-Null Count  Dtype         
    ---  ------           --------------  -----         
     0   Date             7 non-null      datetime64[ns]
     1   Store_Type       7 non-null      object        
     2   City_Type        7 non-null      object        
     3   Day_Temp         7 non-null      float64       
     4   No_of_Customers  7 non-null      float64       
     5   Sales            7 non-null      float64       
     6   Product_Quality  7 non-null      object        
    dtypes: datetime64[ns](1), float64(3), object(3)
    memory usage: 448.0+ bytes
    None
    

    C:\Users\Admin\AppData\Local\Temp\ipykernel_9500\2580223973.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_sales_clean.loc[:, 'Store_Type'] = df_sales_clean['Store_Type'].astype(str)
    C:\Users\Admin\AppData\Local\Temp\ipykernel_9500\2580223973.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_sales_clean.loc[:, 'City_Type'] = df_sales_clean['City_Type'].astype(str)
    C:\Users\Admin\AppData\Local\Temp\ipykernel_9500\2580223973.py:17: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_sales_clean.loc[:, 'Product_Quality'] = df_sales_clean['Product_Quality'].astype(str)
    


    
![png](plots/output_60_2.png)
    


    Pearson correlation between Sales and No_of_Customers: r=-0.15025811144508489, p=0.7477837320667933
    


    
![png](plots/output_60_4.png)
    



    
![png](plots/output_60_5.png)
    



    
![png](plots/output_60_6.png)
    


# Summary

There are many ways to *describe* our data:

- Measure **central tendency**.

- Measure its **variability**; **skewness** and **kurtosis**.

- Measure what **correlations** our data have.

All of these are **useful** and all of them are also **exploratory data analysis** (EDA).
