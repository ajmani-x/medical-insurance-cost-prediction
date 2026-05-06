# Medical Insurance Cost Prediction

This project predicts medical insurance costs using regression models on the [Medical Insurance Dataset](https://www.kaggle.com/datasets/rahulvyasm/medical-insurance-cost-prediction) (age, sex, BMI, children, smoker, region). It demonstrates end-to-end workflow in Python including data preprocessing, feature encoding, model training, and evaluation.

---

## Dataset

- The dataset contains the following columns:
  - `age`: Age of the individual
  - `sex`: Gender (`male` or `female`)
  - `bmi`: Body Mass Index
  - `children`: Number of children
  - `smoker`: Smoking status (`yes` or `no`)
  - `region`: Geographic region (`southwest`, `southeast`, `northwest`, `northeast`)
  - `charges`: Medical insurance cost (target variable)

---

## Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Matplotlib
- Scikit-learn (`LinearRegression`, `Lasso`, `StandardScaler`, `train_test_split`, `cross_val_score`, `cross_val_predict`)

---

## Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Checked dataset shape, missing values, duplicates
   - Generated summary statistics
   - Visualized distributions and relationships between features (optional)

2. **Data Preprocessing**
   - Handled categorical variables using **one-hot encoding**
   - Standardized numeric features using `StandardScaler`

3. **Model Training**
   - Trained **Linear Regression** on scaled data
   - Tried **Lasso Regression** for regularization and feature selection
   - Evaluated models using R² score

4. **Model Evaluation**
   - Used **train/test split** and **cross-validation**
   - Plotted **Actual vs Predicted** to visualize model performance

---

## Results

- **Linear Regression R² (train/test split):** 0.746  
- **Linear Regression R² (5-fold cross-validation):** 0.750  
- **Lasso Regression R²:** Slightly lower than Linear Regression, no significant feature reduction  

> Note: This is a regression problem with noisy data, so R² around 0.75 is expected.

---

## Insights

- **Smoker status** has the largest impact on insurance charges.  
- Linear Regression performs well with scaled features and one-hot encoding.  
- Lasso did not improve performance significantly in this case.  
- Cross-validation gives a more reliable estimate of model performance than a single train/test split.  

---
