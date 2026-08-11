# Dataset Information

## Overview

This project uses customer-level insurance data provided for a **Customer Lifetime Value (CLTV) Prediction Hackathon**.

The dataset was used to develop a machine learning model for predicting Customer Lifetime Value (CLTV).

---

## Dataset Files

The original dataset contained two files:

- Training dataset
- Test dataset

The training dataset contains the target variable `cltv`, which was used to train and evaluate the machine learning model.

The test dataset does not contain the target variable and was used to generate final CLTV predictions.

---

## Important Features

The dataset contains customer-level information such as:

| Feature | Description |
|---|---|
| `id` | Unique customer identifier |
| `income` | Customer income |
| `claim_amount` | Customer claim amount |
| `vintage` | Customer relationship duration |
| `num_policies` | Number of policies held by the customer |
| `cltv` | Customer Lifetime Value target |

The dataset also contains categorical variables that were processed and encoded during the machine learning workflow.

---

##  Dataset Size

### Training Dataset

Approximately **89,392 customer records**.

### Test Dataset

Approximately **59,595 customer records**.

---

##  Target Variable

The target variable is:

```text
cltv

CLTV represents Customer Lifetime Value, which estimates the potential value of a customer over their relationship with a business.

## Dataset Availability

The original CSV files are not included in this repository because the dataset was obtained through a hackathon/competition and may be subject to the original dataset provider's usage and redistribution terms.
The notebook therefore requires the dataset to be obtained separately and placed in the appropriate local data directory before running the project.

## Expected Local Structure

When running the project locally, the data can be organized as:
----------------
data/
├── train.csv
└── test.csv
----------------
The exact filenames may differ depending on the downloaded dataset files.

## Data Processing

Before model training, the dataset goes through several preprocessing steps:

1.Column name cleaning
2.ID separation
3.Numerical data conversion
4.Missing value handling
5.Target transformation
6.Feature engineering
7.Categorical encoding
8.Feature alignment
9.Train-validation splitting

## Important Note

The dataset should be obtained from the original hackathon/competition source and used according to its applicable terms and conditions.
The repository contains the analysis workflow and code rather than redistributing the original dataset.
-------------------------------------
### So your `data` folder is now:

```text
data/
└── README.md
-------------------------------------
