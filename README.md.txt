# Customer Lifetime Value (CLTV) Prediction
--------------------------------------------

##  Project Overview

This project focuses on predicting **Customer Lifetime Value (CLTV)** using machine learning and customer-level insurance-related data.

Customer Lifetime Value is an important business metric that estimates the potential long-term value of a customer to a business.

The objective of this project is to analyze customer characteristics, preprocess the data, engineer meaningful features, build a machine learning regression model, evaluate its performance, and generate CLTV predictions for unseen customers.

This project was developed as part of a **Customer Lifetime Value Prediction Hackathon**.

---

##  Business Problem

Businesses need to understand which customers are likely to generate higher long-term value.

Predicting CLTV can help businesses:

- Identify potentially high-value customers
- Improve customer retention strategies
- Prioritize marketing efforts
- Allocate marketing resources more effectively
- Develop customer segmentation strategies
- Personalize customer campaigns
- Support data-driven customer relationship decisions

The goal of this project is to develop a machine learning model that can estimate CLTV using historical customer information.

---

##  Project Objectives

The main objectives of this project are:

1. Understand and explore the customer dataset.
2. Clean and preprocess the data.
3. Handle missing and inconsistent values.
4. Transform the target variable appropriately.
5. Engineer meaningful customer-level features.
6. Encode categorical variables.
7. Prepare training and validation datasets.
8. Build a machine learning regression model.
9. Evaluate model performance using R².
10. Train the final model on the complete training dataset.
11. Generate CLTV predictions for unseen test customers.
12. Create a final prediction/submission file.
13. Interpret the results from a business perspective.

---

## Dataset

The project uses a customer dataset provided as part of the CLTV prediction hackathon.

The dataset contains customer-level information related to financial and insurance characteristics.

### Important variables include:

| Feature | Description |
|---|---|
| `id` | Unique customer identifier |
| `income` | Customer income |
| `claim_amount` | Customer claim amount |
| `vintage` | Customer relationship duration/vintage |
| `num_policies` | Number of policies held by the customer |
| `cltv` | Customer Lifetime Value target |

The dataset also contains categorical variables that were encoded before model training.

### Dataset Size

The training dataset contains approximately:

**89,392 records**

The test dataset contains approximately:

**59,595 records**

> The original dataset is not included in this repository if redistribution is restricted by the original competition or dataset provider.

---

# Project Workflow

The complete machine learning workflow is:

```text
Customer Dataset
       ↓
Data Loading
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Feature Engineering
       ↓
Target Transformation
       ↓
Categorical Encoding
       ↓
Train / Validation Split
       ↓
LightGBM Regression
       ↓
Model Evaluation
       ↓
Full Dataset Training
       ↓
Test Prediction
       ↓
Inverse Transformation
       ↓
Submission File
       ↓
Business Interpretation

---

## Data Preprocessing

Several preprocessing steps were performed before training the machine learning model.

1. Column Name Cleaning

Column names were cleaned by removing unnecessary whitespace and converting them to lowercase.

------------------------------------------------------
train.columns = train.columns.str.strip().str.lower()
test.columns = test.columns.str.strip().str.lower()
------------------------------------------------------

This provides consistent feature names throughout the workflow.

2. ID Handling

The customer ID was saved separately for the final prediction file and removed from the model features.
----------------------------------------------
test_ids = test["id"]
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)
----------------------------------------------

The ID is not used as a predictive feature because it is simply an identifier.

3. Numerical Data Conversion

The following numerical features were converted to numeric format:
---------------
income
claim_amount
vintage
num_policies
--------------

Invalid values were converted to missing values using:
-------------------------------------
pd.to_numeric(..., errors="coerce")
-------------------------------------

4. Missing Value Handling

Missing numerical values were replaced with zero where appropriate.
---------------------------------------------
train[num_cols] = train[num_cols].fillna(0)
test[num_cols] = test[num_cols].fillna(0)
--------------------------------------------

Additional missing values generated during preprocessing were also handled before model training.


---


## Feature Engineering

Feature engineering was explored to create additional variables that may help the model capture relationships between customer characteristics.

The project defines the following engineered features:

# Income-to-Claim Ratio
------------------------------------------------
income_to_claim = income / (claim_amount + 1)
------------------------------------------------
This represents the relationship between customer income and claim amount.

# Claim-to-Income Ratio
-----------------------------------------------
claim_to_income = claim_amount / (income + 1)
-----------------------------------------------

This provides another perspective on the relationship between claims and income.

# Premium per Policy
--------------------------------------------------
premium_per_policy = income / (num_policies + 1)
--------------------------------------------------

This relates income to the number of policies.

 # Policy per Year
-----------------------------------------------
policy_per_year = num_policies / (vintage + 1)
-----------------------------------------------

This relates the number of policies to customer vintage.

# Log-Transformed Features

The project also defines logarithmic transformations for selected numerical variables.
--------------
income_log
claim_log
policies_log
--------------

These transformations can help reduce the effect of highly skewed numerical distributions.

# Additional Engineered Features

Additional features defined in the project include:
-----------------
income_per_year
claim_per_policy
risk_score
-----------------

Implementation Note: These engineered features are currently defined in the notebook and should be explicitly applied to the training and test datasets before final model training if they are to be included in the final model.


---


## Target Transformation

The CLTV target was transformed using a logarithmic transformation:
----------------
y = np.log1p(y)
----------------

This transformation can help reduce the effect of a highly skewed target distribution.

After generating predictions, the transformation was reversed using:
-----------
np.expm1()
-----------

This converts predictions back to the original CLTV scale.

---

## Categorical Data Encoding
Categorical variables were converted into numerical representations before training the machine learning model.

# The project uses:
----------------------
Pandas get_dummies()
LabelEncoder
----------------------

The training and test datasets were also aligned to ensure they contain compatible feature columns.

---


## Machine Learning Model

# LightGBM Regressor

The main machine learning algorithm used in this project is:

# LightGBM Regressor

LightGBM is a gradient boosting framework designed for efficient learning on structured and tabular datasets.
It was selected because it can model nonlinear relationships and interactions between features effectively.
# Model Configuration
--------------------------
LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03
)
--------------------------
# Training Strategy

The model was initially trained using:
80% training data
20% validation data
After evaluation, the final model was trained using the complete training dataset before generating test predictions.

---


## Model Evaluation

The primary evaluation metric used in this project is:

# R² Score

R², or the coefficient of determination, measures how much of the variation in the target variable is explained by the model.
The validation score was calculated using:
----------------------------
r2_score(y_val, val_preds)
----------------------------
# Validation Result

Metric         Result
----------------------
Validation R²  0.3134

# Interpretation

The validation R² score of 0.3134 indicates that the model explains approximately 31.34% of the variation in the log-transformed CLTV target on the validation dataset.
The score should be interpreted within the context of the target transformation and the competition's evaluation methodology.

---


## Model Results

The trained LightGBM model generated CLTV predictions for the unseen test dataset.
The prediction pipeline was:
-------------------------
Test Features
      ↓
Trained LightGBM Model
      ↓
Log-Scale Predictions
      ↓
Inverse Transformation
      ↓
Original CLTV Scale
--------------------------
The predictions were converted back to the original scale using:
---------------------------------
test_preds = np.expm1(test_preds)
---------------------------------

---

## Prediction Output

The final prediction file contains:

Column       Description
----------------------------------------------
id           Customer identifier
----------------------------------------------
cltv         Predicted Customer Lifetime Value

# Example Output

id       cltv
---------------------
89393    76562.62
89394    107484.00
89395    81115.48
89396    79036.95
89397    84711.74
                              ---------------
The final output is saved as: submission.csv
                              ---------------
---

## Feature Importance

Feature importance was extracted from the trained LightGBM model:
-----------------------------------------
importance = model.feature_importances_
features = X.columns
-----------------------------------------
A horizontal bar chart was created to visualize the relative importance of the model features.
Feature importance can help identify which customer characteristics contribute most strongly to the model's predictions.
The final visualization can be stored in the images/ directory.

---


## Business Insights

The model can be used to support customer-level value analysis.

1. Identification of High-Value Customers

Customers with higher predicted CLTV can potentially be considered high-value customer segments.

# Business Opportunity

Businesses can prioritize these customers for:
----------------------------
Retention campaigns
Loyalty programs
Personalized offers
Premium customer services
----------------------------

2. Customer Segmentation

Predicted CLTV can be used as an additional variable for customer segmentation.

For example:
---------------------------------------
High CLTV
    ↓
High-value customers

Medium CLTV
    ↓
Growth / engagement customers

Low CLTV
    ↓
Potential development opportunities
---------------------------------------
This allows businesses to design different strategies for different customer groups.

3. Marketing Prioritization

CLTV predictions can help businesses determine where marketing resources may have greater long-term impact.
Instead of treating every customer identically, marketing campaigns can be prioritized based on predicted customer value.

4. Customer Retention

Customers with high predicted lifetime value may be especially important to retain.
Businesses could use CLTV predictions alongside customer behavior data to identify customers who deserve additional retention attention.
Important: These are potential business applications of CLTV prediction. Final project-specific insights should be updated after analyzing the actual feature importance and customer prediction distributions.

---


## Business Recommendations

Based on the purpose of the CLTV prediction model, businesses can consider the following strategies:

1. Prioritize High-Value Customers

Use predicted CLTV to identify customers who may generate greater long-term value.

2. Develop Personalized Retention Strategies

Provide targeted offers, loyalty benefits, and personalized experiences to valuable customers.

3. Segment Customers by Expected Value

Create customer segments based on predicted CLTV and develop different strategies for each segment.

4. Optimize Marketing Allocation

Use CLTV predictions as one factor when deciding how marketing resources should be distributed.

5. Monitor Customer Value Over Time

Retrain or update the model as new customer data becomes available.

---


## Potential Business Applications

A CLTV prediction system can potentially support:
--------------------------------------
Customer segmentation
Customer retention
Marketing optimization
Personalized campaigns
Cross-selling
Upselling
Customer relationship management
Marketing budget allocation
High-value customer identification
--------------------------------------

---

## Technologies Used

# Programming Language :- Python

# Data Analysis :- Pandas,NumPy

# Machine Learning :- Scikit-learn,LightGBM

# Data Visualization :- Matplotlib

# Development Environment

Jupyter Notebook
Kaggle
VS Code

# Version Control

Git
GitHub

---

## Repository Structure

cltv-prediction-hackathon/
│
├── README.md
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── cltv_prediction.ipynb
│
├── images/
│   ├── feature_importance.png
│   ├── model_performance.png
│   └── eda.png
│
├── reports/
│   └── CLTV_Analysis_Report.pdf
│
├── outputs/
│   └── submission.csv
│
├── requirements.txt
│
└── .gitignore

# Folder Description

Folder/File             Purpose
---------------------------------------------------------------
README.md          Project documentation
data/              Dataset information or permitted data files
notebooks/         Jupyter Notebook containing the analysis
images/            Important charts and visualizations
reports/           Professional PDF project report
outputs/           Model prediction/submission files
requirements.txt   Python dependencies
.gitignore         Files excluded from Git


The original dataset may be excluded when redistribution is not permitted.

---

## How to Run the Project

1. Clone the Repository
---------------------------------------------------------------------------
git clone https://github.com/tamanna-dev01/cltv-prediction-hackathon.git
---------------------------------------------------------------------------

2. Navigate to the Project Directory
--------------------------------
cd cltv-prediction-hackathon
--------------------------------

3. Create a Virtual Environment
---------------------
python -m venv venv
---------------------

4. Activate the Environment
----------------------------
Windows
venv\Scripts\activate
----------------------------
----------------------------
macOS/Linux
source venv/bin/activate
----------------------------

5. Install Dependencies
---------------------------------
pip install -r requirements.txt
---------------------------------

6. Open the Notebook

Open:
--------------------------------
notebooks/cltv_prediction.ipynb
--------------------------------
You can use:
VS Code
Jupyter Notebook
JupyterLab

Run the notebook cells sequentially.

Note: The original notebook was developed in the Kaggle environment. Kaggle-specific paths such as /kaggle/input/ need to be replaced with local dataset paths when running the project outside Kaggle.

---

## Requirements

The project uses Python and the following major libraries:
-------------
numpy
pandas
scikit-learn
lightgbm
matplotlib
-------------

The complete dependency list is provided in:
------------------
requirements.txt
------------------


Install all dependencies using:
--------------------------------
pip install -r requirements.txt
--------------------------------
---

## Notebook Contents

The main notebook contains the technical implementation of the project.

It includes:
------------------------------
Library imports
Dataset loading
Dataset inspection
Column cleaning
ID handling
Numerical preprocessing
Missing-value handling
Target preparation
Feature engineering
Categorical encoding
Feature alignment
Train-validation split
LightGBM model training
Validation
R² evaluation
Full-data model training
Test prediction
Inverse target transformation
Feature importance
Submission file creation
---------------------------------
---

## Visualizations

Important visualizations from the analysis can be stored in the images/ folder.

These may include:

# Feature Importance

Shows which features contribute most strongly to the LightGBM model.

# Model Performance

Visualizes relevant model evaluation results.

# Exploratory Data Analysis

Shows important patterns and distributions in the customer data.

The visualizations are provided separately so visitors can understand the analysis without running the notebook.

## Project Report

A professional client-friendly report can be included in:
----------------------------------
reports/CLTV_Analysis_Report.pdf
----------------------------------
The report can summarize the project in a business-friendly format.

It can include:
--------------------------
Project objective
Business problem
Dataset overview
Data preparation
Feature engineering
Model methodology
Model performance
Visualizations
Key findings
Business recommendations
Conclusion
--------------------------
---

## Project Highlights
-----------------------------------------------------------------
Built an end-to-end CLTV prediction pipeline.
Processed approximately 89,392 training records.
Generated predictions for approximately 59,595 test records.
Performed numerical preprocessing and missing-value handling.
Explored customer-level feature engineering.
Applied log transformation to the CLTV target.
Encoded categorical variables.
Built a LightGBM regression model.
Achieved a validation R² score of 0.3134.
Retrained the model on the full training dataset.
Generated customer-level CLTV predictions.
Created a final submission.csv file.
Analyzed feature importance for model interpretation.
-----------------------------------------------------------------

---

## Limitations

The current solution has several areas that could be improved.

1. Feature Engineering

Additional domain-specific features could potentially improve predictive performance.

2. Hyperparameter Optimization

The current LightGBM model uses manually selected parameters.
Hyperparameter tuning could be performed using:
------------------------
Grid Search
Random Search
Bayesian optimization
Cross-validation
------------------------
3. Model Comparison

Additional algorithms could be tested, such as:
-------------------
XGBoost
CatBoost
Random Forest
Gradient Boosting
-------------------

4. Model Interpretability

Advanced explainability techniques such as SHAP could be used to provide more detailed explanations of individual predictions.

5. Cross-Validation

Cross-validation could provide a more robust estimate of model performance than a single train-validation split.

---

## Future Improvements

Future versions of this project could include:
----------------------------------------
Advanced feature engineering
Hyperparameter tuning
Cross-validation
Model comparison
SHAP explainability
Customer segmentation
Interactive dashboards
Automated prediction pipelines
Model monitoring
Deployment using Streamlit or FastAPI
----------------------------------------

---

## Key Takeaways

This project demonstrates practical experience in:
-------------------------------------
Data preprocessing
Exploratory data analysis
Feature engineering
Categorical encoding
Regression modeling
LightGBM
Model evaluation
Prediction generation
Feature importance analysis
Business-oriented interpretation
------------------------------------
The project demonstrates how machine learning can be applied to transform customer data into actionable customer-value predictions.

---

## Project Links
--------------------------------------------------------------------------------------
GitHub Repository :- https://github.com/tamanna-dev01/cltv-prediction-hackathon⁠�
GitHub Profile :- https://github.com/tamanna-dev01⁠�
--------------------------------------------------------------------------------------

---

## Author

Tamanna Yadav

Data Analyst | Python | SQL | Machine Learning

# Technical Skills
---------------------
Python
SQL
Pandas
NumPy
Data Analysis
Data Visualization
Machine Learning
Scikit-learn
LightGBM
--------------------

## Disclaimer

This project was developed as part of a machine learning hackathon and is intended for educational, portfolio, and demonstration purposes.

Dataset usage and redistribution are subject to the terms and conditions of the original competition or dataset provider.

