# Data Mining Project

## Project Overview

For financial institutions, credit risk assessment is a crucial task that enables them to distinguish between "good" and "bad" customers—in other words, to determine whether an applicant is eligible for credit. The development of machine learning techniques in recent years has significantly improved this selection process.

The aim of this project is twofold. Firstly, it involves describing and performing exploratory data analysis on the 'German Credit Data' dataset to maximize insight into the dataset and its structure. Secondly, using the knowledge gained from this analysis, different data mining techniques will be employed to classify bank customers as 'good' or 'bad.' These classification models will be compared to select the best one, with the goal of increasing the efficiency of predicting whether a given customer is suitable for credit.

## Data Cleaning 

To ensure the dataset was suitable for analysis and modeling, several preprocessing and cleaning steps were undertaken:

1. **Data Translation:**
   - Translated categorical variables for increased readability. For example, values initially represented as `a1`, `a2`, etc., were converted to more descriptive terms such as `yes`, `no`, etc.

2. **Data Inspection:**
   - **Dataset Size:** Examined the size of the dataset to understand its structure.
   - **Missing Values:** Checked for and confirmed the absence of missing values in the dataset.
   - **Duplicates:** Identified and removed any duplicate entries to maintain data integrity.
   - **Data Types:** Reviewed and adjusted data types of columns to ensure consistency and compatibility with subsequent analysis steps.

3. **Categorical Feature Review and Consolidation:**
   - **Review of Categories:** Analyzed categorical features to understand their categories and distribution.
   - **Merging Groups:** Merged similar categories to reduce granularity and improve interpretability.

## Exploratory Data Analysis (EDA)

In this section, Exploratory Data Analysis (EDA) was performed to gain insights into the dataset and prepare it for classification algorithms. The EDA was divided into two main parts:

1. **Univariate Analysis:**
   - **Categorical Variables:** Visualized the distribution of categorical variables using bar charts to understand the frequency of each category.
   - **Numerical Variables:** 
     - **Summary Statistics:** Generated summary statistics such as mean, median, and standard deviation to understand the characteristics of numerical variables.
     - **Histogram:** Created histograms to visualize the distribution of numerical data.
     - **Boxplot:** Used boxplots to detect outliers and examine the spread and central tendency of numerical variables.

2. **Bivariate Analysis:**
   - **Numerical Variables:**
     - **Summary Tables:** Compiled summary tables to compare numerical variables across different categories of the target variable (e.g., 'good' vs. 'bad').
     - **Scatter Plot:** Plotted scatter plots to explore relationships and correlations between numerical features.
     - **Correlation Matrix:** Developed a correlation matrix to identify correlations between numerical features.
     - **Boxplots:** Compared boxplots for different categories of the target variable to understand variations in numerical features.
   - **Categorical Variables:**
     - **Mosaic Plots:** Utilized mosaic plots to visualize relationships between categorical variables and the target variable.
     - **Bar Plots:** Created bar plots to compare the distribution of categorical variables across target variable categories.

The primary objective of this EDA was to prepare the dataset for classification algorithms by identifying key features that are crucial for classification. This analysis helped in pinpointing the most significant features and relationships, providing valuable insights for improving classification performance.
