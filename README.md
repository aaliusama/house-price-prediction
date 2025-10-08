# House Prices Prediction Project

This repository contains a data science project focused on predicting house prices using advanced regression techniques.  
The work was originally developed as part of an academic exercise and has been cleaned and refactored for inclusion in a professional portfolio.  
It demonstrates a full workflow, including exploratory data analysis, extensive feature engineering, model selection, hyper‑parameter tuning and ensemble methods.

## Overview

The goal of the project is to predict the `SalePrice` of houses in Ames, Iowa based on 79 numeric and categorical features describing various aspects of each house (e.g. overall quality, living area, neighborhood, year built).  
The dataset comes from the [House Prices – Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on Kaggle.  

A variety of regression algorithms were explored, including regularized linear models and gradient boosting methods.  
Careful preprocessing and feature engineering were crucial to handling missing values, encoding categorical variables, and extracting informative features from existing ones.  
The final solution uses an ensemble of gradient boosting models tuned with cross‑validation.

## Data

* **Train/Test Data** – The Kaggle dataset is split into training and test sets.  The training set contains the target variable `SalePrice`; the test set does not.  These files are not included in this repository due to licensing restrictions.  They can be downloaded directly from the Kaggle competition page and placed in a `data/` directory as `train.csv` and `test.csv`.
* **Feature‑engineered datasets** – This repository includes a pair of feature‑engineered CSV files used during experimentation.  `df_test_train_fe.csv` contains the concatenated training and test data after preprocessing and feature engineering, while `df_prediction_fe.csv` holds the corresponding model predictions.
* **Submission files** – Sample submissions (`final_submission_4.csv` and `final_submission_6.csv`) illustrate how predictions were formatted for the competition.

## Methodology

### 1. Exploratory Data Analysis and Cleaning

The initial phase involved exploring the distribution of each feature, identifying missing values and potential outliers.  
Features with more than 80 % missing values were dropped.  Remaining missing values were imputed using context‑appropriate strategies (e.g. filling categorical features with the mode and numerical features with the median).  
Special care was taken to distinguish between values that were truly missing (`NA`) and those indicating the absence of a particular feature (`None`), such as the lack of a basement or fireplace.

### 2. Feature Engineering and Encoding

* **Rare category grouping** – For nominal categorical features, categories that appeared in less than 1 % of the dataset were grouped into an `Other` category.  This reduces sparsity when one‑hot encoding.
* **Ordinal encoding** – Ordinal features (such as quality scores) were mapped to numerical scales reflecting their inherent order.
* **New features** – Additional features were derived from existing ones, such as the age of the house (`AgeOfHouse`), years since a remodel (`YearsSinceRemodel`), and garage age (`GarageAge`).  These transformations capture time‑dependent effects more intuitively.
* **One‑hot encoding** – Nominal categorical features were one‑hot encoded after rare category grouping.  Highly imbalanced features that contributed little to the model were dropped to reduce dimensionality.

### 3. Modeling and Evaluation

Multiple regression models were trained and evaluated using repeated cross‑validation to estimate out‑of‑sample performance.  Models considered include:

* **Ridge Regression** – A regularized linear model that serves as a baseline and meta‑learner in stacked ensembles.
* **XGBoost Regressor** – Gradient boosting trees with careful tuning of depth, learning rate and regularization parameters.
* **LightGBM Regressor** – An efficient gradient boosting framework capable of handling large feature sets.
* **CatBoost Regressor** – A gradient boosting model that handles categorical variables natively.

Hyper‑parameter tuning was performed with `Optuna` and `GridSearchCV`, optimizing the root mean squared error (RMSE).  
Several ensemble strategies were explored, including weighted averaging and stacking.  The best performance on the cross‑validation set was obtained with a weighted ensemble of XGBoost and LightGBM models.

### 4. Results

The weighted ensemble achieved an RMSE of approximately **17,099** on the validation data (lower is better).  
SHAP analysis identified features related to overall quality, total living area, and kitchen quality as the most influential.  
These results demonstrate the benefits of thorough preprocessing and model tuning in achieving competitive performance on tabular regression tasks.

## Repository Structure

```
.
├── data/                     # Feature‑engineered datasets and sample submissions
│   ├── df_prediction_fe.csv
│   ├── df_test_train_fe.csv
│   ├── final_submission_4.csv
│   └── final_submission_6.csv
├── src/
│   └── model.py             # Cleaned and refactored analysis script
├── README.md                # Project overview (this file)
└── LICENSE                  # License information (MIT)
```

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your‑username>/house‑prices‑prediction.git
   cd house‑prices‑prediction
   ```

2. **Install dependencies** – The code relies on common data science libraries such as NumPy, pandas, scikit‑learn, XGBoost, LightGBM and CatBoost.  You can install them via:

   ```bash
   pip install -r requirements.txt
   ```

   Note: A `requirements.txt` file is not included here, but you can generate one using `pipreqs` or by inspecting the imports in `src/model.py`.

3. **Add the Kaggle data** – Download `train.csv` and `test.csv` from the Kaggle competition and place them in a `data/` directory.  Update the file paths in `src/model.py` if necessary.

4. **Run the script** – Execute the analysis script to reproduce the preprocessing, modeling and evaluation steps:

   ```bash
   python src/model.py
   ```

## License

This project is licensed under the MIT License.  See the `LICENSE` file for details.