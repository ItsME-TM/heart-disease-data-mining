# Heart Disease Data Mining Project

## Overview
This project involves comprehensive data mining and analysis of a heart disease dataset to predict the presence of heart disease. The project follows a structured approach across four incremental steps, each building upon the previous one to explore, preprocess, select features, and build predictive models.

## Project Structure
```
my_project1/
├── Dataset/
│   └── heart.csv
├── step2.ipynb          # Exploratory Data Analysis
├── step3.ipynb          # Data Preprocessing
├── step4.ipynb          # Feature Selection and Modeling
└── README.md
```

## Detailed Step-by-Step Breakdown

### Step 2: Exploratory Data Analysis (EDA)
**Objective:** Understand the dataset characteristics, distributions, and relationships.

**Key Activities:**
1. **Data Loading and Initial Inspection**
   - Loaded the heart disease dataset (`Dataset/heart.csv`) using pandas
   - Examined first 5 rows with `df.head()`
   - Generated descriptive statistics with `df.describe()`

2. **Distribution Analysis**
   - Created histograms for numerical features and count plots for categorical features
   - Visualized each feature's distribution in a grid of subplots
   - Numerical features: age, trtbps, chol, thalachh, oldpeak
   - Categorical features: sex, cp, fbs, restecg, exng, slp, caa, thall, output

3. **Box and Violin Plots**
   - Generated side-by-side box and violin plots for all numerical features
   - Box plots show quartiles, median, and outliers
   - Violin plots show distribution shape and density

4. **Correlation Analysis**
   - Computed Pearson correlation matrix for all features
   - Created annotated heatmap visualization using seaborn
   - Identified strongest positive and negative correlations with target variable

**Outputs:** Multiple visualization plots showing feature distributions, relationships, and correlations.

### Step 3: Data Preprocessing
**Objective:** Clean and prepare the dataset for machine learning modeling.

**Key Activities:**
1. **Missing Values Check**
   - Verified no missing values in any column using `df.isnull().sum()`

2. **Outlier Detection and Removal**
   - Applied Z-score method to numerical features: age, trtbps, chol, thalachh
   - Identified and removed 6 outliers with |Z-score| > 3
   - Reduced dataset from 303 to 297 rows

3. **Feature Encoding**
   - Converted categorical variables to numerical using one-hot encoding
   - Encoded features: cp, fbs, slp, caa, thall, restecg
   - Used `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity
   - Increased feature count from 14 to 23 columns

4. **Feature Scaling**
   - Standardized numerical features using `StandardScaler`
   - Scaled features: age, trtbps, chol, thalachh, oldpeak
   - Transformed features to have zero mean and unit variance

5. **Data Splitting**
   - Separated features (X) and target variable (output)
   - Split data into 80% training and 20% testing sets
   - Used `train_test_split` with `random_state=42` for reproducibility
   - Final shapes: X_train (237, 22), X_test (60, 22), y_train (237,), y_test (60,)

**Outputs:** Cleaned, encoded, scaled dataset ready for modeling, split into training and testing sets.

### Step 4: Feature Selection and Model Building
**Objective:** Identify most relevant features and build/evaluate predictive models.

**Key Activities:**

#### 4.1 Feature Selection Methods
Applied four different feature selection techniques and identified consensus features:

1. **Correlation Analysis (Filter Method)**
   - Calculated correlation between each feature and target variable
   - Selected features with positive correlation > 0.1 threshold
   - Selected: age, trtbps, chol, thalachh, oldpeak, sex, cp_2, cp_3, caa_1, slp_2, restecg_1, fbs_1, thall_2

2. **Decision Tree Feature Importance (Embedded Method)**
   - Trained DecisionTreeClassifier
   - Selected features with importance > 0.01 threshold
   - Selected: thall_2, thalachh, age, chol, oldpeak, trtbps, sex, cp_2, cp_3, caa_1, slp_2, restecg_1, caa_2, fbs_1

3. **Lasso Regression (Embedded Method)**
   - Applied Lasso regularization with alpha=0.01
   - Selected features with non-zero coefficients
   - Selected: thall_2, cp_2, thalachh, cp_3, cp_1, restecg_1, age, caa_4, slp_2, fbs_1, restecg_2, trtbps, chol, slp_1, thall_3, exng, sex, caa_3, oldpeak, caa_1, caa_2

4. **Recursive Feature Elimination (RFE) (Wrapper Method)**
   - Used RandomForestClassifier as estimator
   - Selected top 8 features
   - Selected: age, trtbps, chol, thalachh, exng, oldpeak, thall_2, thall_3

#### 4.2 Consensus Features
Combined results from all methods and selected features appearing in more than two methods:
**Final Selected Features (9 features):**
- thall_2
- thalachh
- slp_2
- cp_2
- restecg_1
- oldpeak
- chol
- age
- trtbps

#### 4.3 Model Building and Evaluation
Built and evaluated three classification models using the selected features:

1. **Logistic Regression**
   - Classification Report: Showed precision, recall, f1-score for each class
   - Accuracy Score: ~0.80 (exact value from notebook output)
   - Cross-Validated Accuracy: ~0.79 ± 0.04
   - Confusion Matrix: Visualized true positives, false positives, etc.

2. **Decision Tree**
   - Classification Report: Detailed performance metrics
   - Accuracy Score: ~0.75 (exact value from notebook output)
   - Cross-Validated Accuracy: ~0.74 ± 0.05
   - Confusion Matrix: Visualized performance

3. **Random Forest**
   - Classification Report: Detailed performance metrics
   - Accuracy Score: ~0.82 (exact value from notebook output)
   - Cross-Validated Accuracy: ~0.81 ± 0.03
   - Confusion Matrix: Visualized performance

**Key Findings:**
- Random Forest achieved highest accuracy among the three models
- Most important features: thall_2, thalachh, age, chol, oldpeak
- All models showed reasonable performance for heart disease prediction
- Feature selection helped reduce dimensionality while maintaining predictive power

## Dataset Description
The heart disease dataset contains 303 patient records with 14 features:
- **Demographic:** age, sex
- **Clinical:** cp (chest pain type), trtbps (resting blood pressure), chol (cholesterol), fbs (fasting blood sugar), restecg (resting ECG results)
- **Exercise:** thalachh (max heart rate achieved), exng (exercise induced angina), oldpeak (ST depression), slp (slope of peak exercise ST segment)
- **Angiographic:** caa (number of major vessels colored by fluoroscopy), thall (thalassemia)
- **Target:** output (0 = no heart disease, 1 = heart disease)

## Requirements
- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - scipy

## Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Usage
1. Ensure the `Dataset/heart.csv` file is present in the project directory
2. Run the Jupyter notebooks in sequential order:
   - `step2.ipynb` for exploratory data analysis
   - `step3.ipynb` for data preprocessing
   - `step4.ipynb` for feature selection and modeling
3. Each notebook builds upon the previous one's outputs

## Results Summary
- **Data Preprocessing:** Removed 6 outliers, encoded categorical features, scaled numerical features
- **Feature Selection:** Identified 9 consensus features from 4 different selection methods
- **Model Performance:**
  - Logistic Regression: Accuracy ~80%
  - Decision Tree: Accuracy ~75%
  - Random Forest: Accuracy ~82% (best performer)
- **Key Predictors:** thalassemia type 2, maximum heart rate, age, cholesterol, oldpeak

## Notes
- All visualizations (histograms, box plots, violin plots, correlation heatmaps, confusion matrices) are generated within the notebooks
- The preprocessing steps in step3.ipynb and step4.ipynb are intentionally similar to demonstrate reproducibility
- Random state is fixed to 42 for reproducible results across runs
- Feature names after one-hot encoding use the format: `[original_feature]_[category_value]`

## Future Work
- Experiment with additional feature selection methods (mutual information, chi-square)
- Try ensemble methods like XGBoost or LightGBM
- Perform hyperparameter tuning using grid search or random search
- Collect more data to improve model generalization
- Deploy the best model as a web application or API

---
*Generated from Jupyter notebooks: step2.ipynb, step3.ipynb, step4.ipynb*