# Data Mining Project

## Project Overview

For financial institutions, credit risk assessment is a crucial task that enables them to distinguish between "good" and "bad" customers—in other words, to determine whether an applicant is eligible for credit. The development of machine learning techniques in recent years has significantly improved this selection process.

The aim of this project is twofold. Firstly, it involves describing and performing exploratory data analysis on the 'German Credit Data' dataset to maximize insight into the dataset and its structure. Secondly, using the knowledge gained from this analysis, different data mining techniques will be employed to classify bank customers as 'good' or 'bad.' These classification models will be compared to select the best one, with the goal of increasing the efficiency of predicting whether a given customer is suitable for credit. Additionaly 

## Setup

To get started with this project, you need to install the required Python libraries. All the necessary libraries are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Dataset Overview

The dataset contains 1,000 records, each described by 20 features and one response variable. The features include personal, demographic, and financial factors of potential loan applicants. The response variable indicates the risk score: `1` for a good risk score and `2` for a bad risk score.

### Feature Descriptions

- **Attribute 1:** Status of existing checking account (ordinal)
  - `A11`: < 0 DM
  - `A12`: 0 ≤ ... < 200 DM
  - `A13`: ≥ 200 DM / salary assignments for at least 1 year
  - `A14`: No checking account

- **Attribute 2:** Duration in months (numerical/continuous)

- **Attribute 3:** Credit history (nominal)
  - `A30`: No credits taken/all credits paid back duly
  - `A31`: All credits at this bank paid back duly
  - `A32`: Existing credits paid back duly till now
  - `A33`: Delay in paying off in the past
  - `A34`: Critical account/other credits existing (not at this bank)

- **Attribute 4:** Purpose (nominal)
  - `A40`: Car (new)
  - `A41`: Car (used)
  - `A42`: Furniture/equipment
  - `A43`: Radio/television
  - `A44`: Domestic appliances
  - `A45`: Repairs
  - `A46`: Education
  - `A47`: Vacation
  - `A48`: Retraining
  - `A49`: Business
  - `A410`: Others

- **Attribute 5:** Credit amount (numerical/continuous)

- **Attribute 6:** Savings account/bonds (ordinal)
  - `A61`: < 100 DM
  - `A62`: 100 ≤ ... < 500 DM
  - `A63`: 500 ≤ ... < 1000 DM
  - `A64`: ≥ 1000 DM
  - `A65`: Unknown/no savings account

- **Attribute 7:** Present employment since (ordinal)
  - `A71`: Unemployed
  - `A72`: < 1 year
  - `A73`: 1 ≤ ... < 4 years
  - `A74`: 4 ≤ ... < 7 years
  - `A75`: ≥ 7 years

- **Attribute 8:** Installment rate in percentage of disposable income (ordinal)

- **Attribute 9:** Personal status and sex (nominal)
  - `A91`: Male: divorced/separated
  - `A92`: Female: divorced/separated/married
  - `A93`: Male: single
  - `A94`: Male: married/widowed
  - `A95`: Female: single

- **Attribute 10:** Other debtors/guarantors (nominal)
  - `A101`: None
  - `A102`: Co-applicant
  - `A103`: Guarantor

- **Attribute 11:** Present residence since (ordinal)

- **Attribute 12:** Property (nominal)
  - `A121`: Real estate
  - `A122`: Building society savings agreement/life insurance
  - `A123`: Car or other (not in attribute 6)
  - `A124`: Unknown/no property

- **Attribute 13:** Age (numerical/continuous)

- **Attribute 14:** Other installment plans (nominal)
  - `A141`: Bank
  - `A142`: Stores
  - `A143`: None

- **Attribute 15:** Housing (nominal)
  - `A151`: Rent
  - `A152`: Own
  - `A153`: Free

- **Attribute 16:** Number of existing credits at this bank (ordinal)

- **Attribute 17:** Job (ordinal)
  - `A171`: Unemployed/unskilled - non-resident
  - `A172`: Unskilled - resident
  - `A173`: Skilled employee/official
  - `A174`: Management/self-employed/highly qualified employee/officer

- **Attribute 18:** Number of people being liable to provide maintenance for (ordinal)

- **Attribute 19:** Telephone (binary)
  - `A191`: None
  - `A192`: Yes, registered under the customer's name

- **Attribute 20:** Foreign worker (binary)
  - `A201`: Yes
  - `A202`: No

- **Classification:** 
  - `1`: Good
  - `2`: Bad

### Cost Matrix

Additionally the data set requires a cost matrix, since it is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).

The cost matrix for the classification is as follows:

| Predicted | 1   | 2   |
|-----------|-----|-----|
| Actual 1  | 0   | 1   |
| Actual 2  | 5   | 0   |


For a detailed description of each attribute, please refer this [article](https://www1.beuth-hochschule.de/FB_II/reports/Report-2019-004.pdf).

## Data Cleaning 

To ensure the dataset was suitable for analysis and modeling, several preprocessing and cleaning steps were undertaken:

1. **Data Translation:**
   - Translated categorical variables for increased readability. 

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
   - **Categorical Variables:** Bar charts were used to show category distributions.
   - **Numerical Variables:** 
     - **Summary Statistics:** Included mean, median, and standard deviation.
     - **Histogram:** Showed the distribution of numerical data.
     - **Boxplot:** Revealed outliers and data spread.

2. **Bivariate Analysis:**
   - **Numerical Variables:**
     - **Summary Tables:** Compared numerical features across target categories.
     - **Scatter Plot:** Examined feature relationships and correlations.
     - **Correlation Matrix:** Highlighted feature correlations.
     - **Boxplots:** Compared feature distributions by target category.
   - **Categorical Variables:**
     - **Mosaic Plots:** Illustrated relationships with the target variable.
     - **Bar Plots:** Compared categorical distributions across target categories.

The primary objective of this EDA was to prepare the dataset for classification algorithms by identifying key features that are crucial for classification. This analysis helped in pinpointing the most significant features and relationships, providing valuable insights for improving classification performance.

## Feature Selection

In this phase, feature selection was conducted to identify the most relevant features for the classification model. The process involved statistical tests for both continuous and categorical variables.

1. **Feature Selection for Continuous Variables:**
   - **ANOVA Test:** The ANOVA test was used to evaluate the relationship between continuous features and the target variable. The results showed that all p-values were below the 0.05 significance level, indicating that each continuous feature is significantly correlated with the target variable.
   - **Decision:** All continuous features were retained based on the ANOVA test results. The selected continuous features are:
     - `Age`
     - `Duration`
     - `Credit Amount`

2. **Feature Selection for Categorical Variables:**
   - **Chi-Square Test:** The Chi-square test was applied to assess the association between categorical features and the target variable. Features with p-values above the threshold were deemed less relevant.
   - **Rejected Features:** The following categorical features were rejected:
     - `Installment Rate`
     - `Residence Since`
     - `Existing Credits`
     - `Job`
     - `People Liable`
     - `Telephone`
   - **Selected Features:** The features retained for further analysis include:
     - `Existing Checking`
     - `Credit History`
     - `Purpose`
     - `Savings`
     - `Employment Since`
     - `Status Sex`
     - `Other Debtors`
     - `Property`
     - `Other Installment Plans`
     - `Housing`
     - `Foreign Worker`

## Encoding

To prepare the data for machine learning, categorical features were encoded as follows:

1. **Ordinal Features:** Mapped ordinal features to numeric values.
2. **Nominal Features:** Used dummy encoding to represent nominal features with N-1 binary variables.
3. **Target Variable:** Transformed the ‘Classification’ column into binary values, with ‘Good’ as 0 and ‘Bad’ as 1 to highlight the minority class.

The resulting dataset contains 38 numeric features and a class label.

## Preprocessing and Class Imbalance Handling

In this section, we detail the preprocessing steps and methods used to handle class imbalance:

1. **Data Preparation:**
   - **Splitting and Standardization:** The dataset was divided into training (70%) and test (30%) sets to prevent information leakage. Standardization was performed on the training set to ensure features have a mean of zero and a variance of one, improving model performance. This process was done only on the training data to keep the test set unbiased.

2. **Handling Class Imbalance:**
   - **ADASYN Sampling:** The ADASYN (Adaptive Synthetic Sampling) technique was used to address class imbalance. It generates synthetic samples for the minority class, balancing the dataset without duplicating existing samples. This technique was applied only to the training set to maintain the test set's original class distribution for accurate performance evaluation.


## Methodology

To evaluate the performance of different classification approaches on the German Credit dataset, we performed a comprehensive comparison involving multiple steps:

### 1. Feature Selection and Subsets

We applied different feature selection techniques tailored to each classification method to create distinct subsets of features. This involved the following strategies:

- **Logistic Regression:**
  - **Feature Selection Approach:** Selected features based on the model’s coefficients, which indicate the change in the log odds for a one-unit change in the predictor variable. Larger absolute values suggest a stronger relationship with the target variable.
  - **Standardization:** Features were standardized to ensure that coefficient magnitudes accurately reflect feature importance, preventing features with larger values from appearing more important due to scale differences.

- **K-Nearest Neighbors (KNN):**
  - **Feature Selection Approach:** Utilized LassoCV (Lasso regularization) to determine feature importance. Lasso penalizes irrelevant features by setting their coefficients to zero, effectively removing them from consideration.

- **Decision Trees:**
  - **Feature Selection Approach:** Calculated feature importance based on weighted Gini indices. Feature importance scores were averaged over 10 iterations to reduce the effect of randomness.

### 2. Addressing Class Imbalance

To handle the class imbalance in the dataset, we incorporated the ADASYN (Adaptive Synthetic Sampling) technique. This method generates synthetic samples to balance the class distribution. Each classification method was tested both with and without ADASYN to assess its impact on model performance.

### 3. Combinations Evaluated

We conducted a thorough comparison involving the following combinations:

- **Logistic Regression:**

  1. Logistic Regression with features selected based on coefficients
  2. Logistic Regression with all features
  3. Logistic Regression with features selected based on coefficients and ADASYN
  4. Logistic Regression with all features and ADASYN

- **K-Nearest Neighbors (KNN):**

  5. KNN with features selected using LassoCV
  6. KNN with all features
  7. KNN with features selected using LassoCV and ADASYN
  8. KNN with all features and ADASYN

- **Decision Trees:**

  9. Decision Trees with features selected based on Gini indices
  10. Decision Trees with all features
  11. Decision Trees with features selected based on Gini indices and ADASYN
  12. Decision Trees with all features and ADASYN


### 4. Performance Metrics

Given the specific cost matrix associated with the German Credit dataset, we focused on metrics that account for both precision and recall. This included:

- **Fβ Score:** A generalization of the F1 measure, with β set to 2 to give more weight to recall.
  - Formula: \( Fβ = \frac{(1 + β^2) \times \text{Precision} \times \text{Recall}}{β^2 (\text{Precision} + \text{Recall})} \)

To evaluate model performance, we prepared a data frame containing the following metrics:
- **F2 Score**
- **F1 Score**
- **Accuracy**
- **Cost:** Based on the given cost matrix
- **Sensitivity**
- **Specificity**

To better understand the results, we included several visualizations:
- **Confusion Matrices:** Plotted for each classification method and feature subset combination.
- **ROC Curves:** Compared ROC curves for chosen features, all features, and with or without ADASYN.


