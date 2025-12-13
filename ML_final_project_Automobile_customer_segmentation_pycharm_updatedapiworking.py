#     --------Customer Segmentation for Automobile Market Expansion-----------

# --- Project Objective ---
# To identify the most reliable machine learning model ,train and test a customer segmentation/classification model using existing market data,



# 1. Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Import the Data and view info

df = pd.read_csv("Train.csv")

# Taking backup of the dataset
df_raw = df.copy()

# Rename Var_1 to Category
df.rename(columns={'Var_1': 'Category'}, inplace=True)

df.head()

print(df.head())

print("-----------------------------------------------------------------\n")

# Exploration - EDA

df.info()

print(df.info())
print('-*-'*25)

print(df.describe())
print('-*-'*25)



# Unique values in each column

print("Unique values in each column:\n")
for col in ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Category', 'Segmentation']:
    print(f"{col:15} :", df[col].nunique(), "unique values")
    print("-" * 30)
    print(f"Values: {df[col].unique()}")
    print("-*"*25)



# Visualisation

# Visualization of Null values in the columns
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='plasma')
plt.title('Missing Values Heatmap')
plt.show()


# cmap='plasma'     # bright blue → yellow
# cmap='inferno'    # Black → red → yellow
# cmap='magma'      # Black → purple → white
# cmap='cividis'    # Blue → yellow (colorblind-friendly)
# cmap='viridis'    #purple ->yellow


# Dropping off 'ID' column -Training dataset, as it is not necessary

df = df.drop('ID', axis= 1)

# to check the columns after dropping 'ID' in the Training dataset

for col in df.columns:
    print(col)


# Count Null Values
na= df.isna().sum()

print(na)

print('-*-'*25)


# Deal with null values with low frequencies which cannot be recovered

# Count rows before dropping
before = df.shape[0]
print("Number of rows before dropping:\n",before)

# dropping of rows with null values in the below specific columns
df.dropna(subset=['Ever_Married', 'Graduated', 'Category', 'Profession'], inplace=True)
print("----Checking Null values----\n ")
print(df.isna().sum())


# Count rows after dropping
after = df.shape[0]
print("Number of rows after dropping:\n",after)


# total number of rows were dropped
dropped = before - after
print(f"Rows before: {before}\n")
print(f"Rows after : {after}\n")
print(f"Rows dropped: {dropped}\n")

print('-*-'*25)


# Try to find a relation to recover missing data
print(df['Family_Size'].value_counts())
print('-'*50)
print(df['Work_Experience'].value_counts())


# use median to fill missing values to avoid bias from outliers as the values are spread out

df.fillna(
    {'Work_Experience': df['Work_Experience'].median(),
     'Family_Size': df['Family_Size'].median()},
    inplace=True
)

print("----Checking Null values----\n ",df.isna().sum())


# Importing and Data processing for the Test Dataset

# Import the Data and view info

df_test = pd.read_csv("Test.csv")

# Taking backup of the dataset
df_test_raw = df.copy()

# Rename Var_1 to Category
df_test.rename(columns={'Var_1': 'Category'}, inplace=True)

print(df_test.head())


# Unique values in each column

print("*** Unique values in each column- Test data ***\n")
for col in ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Category', 'Segmentation']:
    print(f"{col:15} :", df[col].nunique(), "unique values")
    print("-" * 50)
    print(f"Values: {df[col].unique()}")
    print("\n")


# Dropping off 'ID' column - Test dataset , as it is not necessary

df_test = df_test.drop('ID', axis= 1)

# to check columns after dropping 'ID' in the Test dataset

for col_test in df_test.columns:
    print(col_test)


# Count Null Values
na_test= df_test.isna().sum()

print(na_test)

print('-*-'*25)


# Deal with null values with low frequencies which cannot be recovered

# Count rows before dropping
before_test = df_test.shape[0]
print("Test dataset_Number of rows before dropping:\n",before_test)

# dropping of rows with null values in the below specific columns
df_test.dropna(subset=['Ever_Married', 'Graduated', 'Category', 'Profession'], inplace=True)
df_test.isna().sum()

# Count rows after dropping - Test Dataset
after_test = df.shape[0]
print("Test dataset_Number of rows before dropping:",after_test)


# total number of rows were dropped
dropped_test = before - after
print(f"\nRows before: {before_test}\n")
print(f"Rows after : {after_test}\n")
print(f"Rows dropped: {dropped_test}\n")


# Try to find a relation to recover missing data
print(df_test['Family_Size'].value_counts())
print('-'*50)
print(df_test['Work_Experience'].value_counts())


# use median to fill missing values to avoid bias from outliers as the values are spread out

df_test.fillna(
    {'Work_Experience': df_test['Work_Experience'].median(),
     'Family_Size': df_test['Family_Size'].median()},
    inplace=True
)

print("----Test Dataset_Checking Null values----\n ",df_test.isna().sum())



# VISUALISATION - Pair plot,Heat Map

# to list columns in the dataset

for col in df.columns:
    print(col)

# Pair Plot without hue - summary of relationships and distributions across multiple features at once.

sns.pairplot(data= df)
plt.suptitle("Distributions and Correlations Across Features", y=1.0)  # y adjusts title position
plt.show()


# Pair Plot with hue – Customer Relationships Across Features
sns.pairplot(df, hue='Segmentation')
plt.suptitle("Pairwise Relationships of Customer Attributes", y=1.0)  # y adjusts title position
plt.show()



# Correlation Matrix (HEAT MAP) - Select only numeric columns for correlation

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(data=df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title('Heat Map of Numeric Features')
plt.show()


# BOX PLOT - Effect of Family size on Spending for each gender

sns.boxplot(data=df, x='Spending_Score', y='Family_Size', hue='Gender')
plt.title('Effect of Family Size on Spending for Each Gender')
plt.show()



# ----feature‑wise visualization dashboard----
# Visualisation automation: The visualization for every column except for Segmentation column
# Histogram - Feature distribution - feature is skewed, has outliers, or is balanced.
# #Boxplot - Feature relation against segmentation feature
# Pie chart -quick sense of categorical distribution


for col in df.columns:              # Loop through all columns in the DataFrame
    if col != 'Segmentation':       # Skip the 'Segmentation' column itself
        plt.figure(figsize=(8, 4))  # Create a new figure for each column, size 8x4

        # 1️. Histogram (Distribution of the column)
        plt.subplot(1, 3, 1)        # First subplot in a 1-row, 3-column layout
        sns.histplot(data=df[col], color='salmon', kde=True)
        plt.title(col)              # Title = column name

        # 2. Boxplot (Relation with Segmentation, grouped by Gender)
        plt.subplot(1, 3, 2)        # Second subplot
        sns.boxplot(data=df, x=col, y='Segmentation', hue='Gender')
        plt.title('Segmentation')

        # 3️. Pie chart (Proportion of values in the column)
        plt.subplot(1, 3, 3)        # Third subplot
        df[col].value_counts().plot.pie(
            autopct='%1.1f%%',      # Show percentages with 1 decimal
            startangle=90,          # Rotate start angle for better look
            cmap='Blues'            # Color map
        )
        plt.title(f'Pie Chart of \n{col}')
        plt.ylabel('')              # Remove y-label for cleaner look

        plt.tight_layout()          # Adjust spacing so plots don’t overlap
        plt.show()                  # Display the figure



# 3. Machine Learning Kick starts here ...
#  Data preprocessing - Label encoder -Transforming categorical to numerical


# Creating an empty dictionary to store LabelEncoder objects for each categorical column
le = dict()  # store encoders per column

lbld_train_data = df.copy()  # shallow copy to avoid override of original dataset

cat_cols = df.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    le[col] = LabelEncoder()
    lbld_train_data[col] = le[col].fit_transform(df[col])

print("*-"*20)
print("Label encoded columns\n", lbld_train_data.head())

import joblib

#  Save the encoders
joblib.dump(le, "label_encoders.pkl")

#  Create a dictionary to store mappings for each column
label_mappings = {}

for col in cat_cols:
    classes = le[col].classes_
    codes = le[col].transform(classes)
    label_mappings[col] = dict(zip(codes, classes))

#  Optional: print mappings for documentation
for col, mapping in label_mappings.items():
    print(f"\nMapping for {col}:")
    for code, label in mapping.items():
        print(f"  {code} → {label}")

# Save mappings for use in Flask
joblib.dump(label_mappings, "label_mappings.pkl")

#  Load encoders and mappings (AFTER saving)
le = joblib.load("label_encoders.pkl")
label_mappings = joblib.load("label_mappings.pkl")
segment_map = label_mappings["Segmentation"]



# Feature Selection for model training

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

# Disable sklearn warnings
warnings.filterwarnings('ignore')

# Numerical Features selection

from sklearn.feature_selection import f_classif    #imports ANOVA F-Test for numeric feature selection

x = lbld_train_data[numeric_columns]    # numeric features

y = lbld_train_data['Segmentation']     # target variable

# to understand the dataset split
x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size= 0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_cv.shape)
print(y_cv.shape)




# Run ANOVA F-test to Obtain Scores for feature selection

f_value, p_value = f_classif(x, y)

# Threshold
alpha = 0.05

# Lists to collect results
relevant_features = []
irrelevant_features = []

# Loop through features
for i in range(len(f_value)):
    feature = numeric_columns[i]
    f_val = f_value[i]
    p_val = p_value[i]

    if p_val < alpha:
        relevant_features.append((feature, f_val, p_val))
    else:
        irrelevant_features.append((feature, f_val, p_val))

# Print consolidated results
print("---- Relevant Features (strong relation with target): ----")
if relevant_features:
    for feat, fval, pval in relevant_features:
        print(f"- {feat}: F-value = {fval:.3f}, P-value = {pval:.3e}")
else:
    print("0")

print("\n---- Irrelevant Features (not statistically suitable): ----")
if irrelevant_features:
    for feat, fval, pval in irrelevant_features:
        print(f"- {feat}: F-value = {fval:.3f}, P-value = {pval:.3e}")
else:
    print("0")

# Final conclusion
if not irrelevant_features:
    print("\n Final conclusion — all features are statistically significant.\n")
else:
    print("\n️ Final conclusion — some features are not statistically significant and may be excluded.\n")




## ----Categorical Features Selection - with chi square test----

from sklearn.feature_selection import chi2

x = lbld_train_data[cat_cols].drop('Segmentation', axis= 1)
y = lbld_train_data['Segmentation']

# 🔍 Debug checks (add here)
print("Categorical columns:", cat_cols)
print("Shape of x:", x.shape)
print("Unique values in Gender:", x['Gender'].unique())
print("Unique values in Ever_Married:", x['Ever_Married'].unique())



# # Run Chi-square test
score, p_value = chi2(x, y)

# Threshold
alpha = 0.05

# Lists to collect results
relevant_features = []
irrelevant_features = []

# Loop through categorical features
for i in range(len(score)):
    feature = cat_cols[i]
    chi_val = score[i]
    p_val = p_value[i]

    if p_val < alpha:
        relevant_features.append((feature, chi_val, p_val))
    else:
        irrelevant_features.append((feature, chi_val, p_val))

# Print consolidated results

print("---- Relevant Categorical Features (strong relation with target): ----\n")
if relevant_features:
    for feat, chi_val, pval in relevant_features:
        print(f"- {feat}: Chi2 = {chi_val:.3f}, P-value = {pval:.3e}")
else:
    print("0")

print("\n---- Irrelevant Categorical Features (not statistically suitable): ----\n")
if irrelevant_features:
    for feat, chi_val, pval in irrelevant_features:
        print(f"- {feat}: Chi2 = {chi_val:.3f}, P-value = {pval:.3e}")
else:
    print("0")

# Final conclusion

if not irrelevant_features:
    print("\n Final conclusion — all categorical features are statistically significant.\n")
else:
    print("\n Final conclusion — some categorical features are not statistically significant and may be excluded if required.\n")





# 4.Building Models

# --Scaling & Splitting--

# Splitting the Data

x = lbld_train_data.drop('Segmentation', axis= 1)
y = lbld_train_data['Segmentation']

x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_cv shape: {x_cv.shape}")
print(f"y_cv shape: {y_cv.shape}")


#Scaling Numeric Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train[numeric_columns] = scaler.fit_transform(x_train[numeric_columns])
x_cv[numeric_columns] = scaler.transform(x_cv[numeric_columns])


# I. Decision Tree Model

print("\n----- Decision Tree Model-----\n")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


# Training the model
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)


# Train Test
# train_pred = dtree.predict(x_train)
# print("Training predit output:\n",train_pred)


# Predict on training set
train_pred = dtree.predict(x_train)

# Convert numeric codes to original labels
train_labels = [segment_map[int(code)] for code in train_pred]

print("\n--- Decision Tree model prediction ---\n")
print("\n Raw prediction codes:\n", train_pred[:20])
print("\n Mapped segment labels:\n", train_labels[:20])



# Evaluating performance
print("\n Training Classification Report :\n",classification_report(y_train, train_pred))


# Evaluating the model on the Cross-Validation Set

cv_pred = dtree.predict(x_cv)

print("\n-- CV predict:Decision Tree -- \n",cv_pred)
print(" *-" * 20)


#CV (validation) report
print("\n CV Validation Report of testdata :\n",classification_report(y_cv, cv_pred))
print(" *-" * 20)


#Confusion matrix - Model performance comparing true labels vs predicted labels.

cm = confusion_matrix(y_cv, cv_pred)
print("\n confusion matrix - Decision Tree on CV Set :\n",cm)
print(" *-" * 20)



# Heat Map
sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues')
plt.title("Confusion Matrix on CV Set - Decision Tree", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

print("*-" * 20)




## II. Bagging - ML technique to reduce variance, prevent overfitting


# -----Random Forest-----

print("\n----- Random Forest Model-----\n")

from sklearn.ensemble import RandomForestClassifier

# Training the model
rf_clf = RandomForestClassifier(n_estimators= 5)
rf_clf.fit(x_train, y_train)


# Testing on the training set
train_pred = rf_clf.predict(x_train)
# print("Training predit output:\n",train_pred)

# Convert numeric codes to original labels
train_labels = [segment_map[int(code)] for code in train_pred]

print("\n--- Random forest model prediction ---\n")
print("\n Raw prediction codes:\n", train_pred[:20])
print("\n Mapped segment labels:\n", train_labels[:20])

# Evaluating performance
print("\n Training Classification Report :\n", classification_report(y_train, train_pred))


# Evaluating the model on the Cross-Validation Set

cv_pred = rf_clf.predict(x_cv)

print("\n-- CV predict:Random forest -- \n",cv_pred)


#CV (validation) report

print("\n CV Validation Report of testdata :\n",classification_report(y_cv, cv_pred))


# Confusion matrix
cm = confusion_matrix(y_cv, cv_pred)
print("\n-- Confusion Matrix:Random forest -- \n",cm)


#Heat Map
sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues')
plt.title("Confusion Matrix on CV Set - Random Forest", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

print("*-" * 20)



# Bagging Classifier Model


print("\n----- Bagging Classifier Model-----\n")

from sklearn.ensemble import BaggingClassifier

# Training the model

bg_clf = BaggingClassifier(estimator= DecisionTreeClassifier(), n_estimators= 50)
bg_clf.fit(x_train, y_train)


# Testing on the training set

train_pred = bg_clf.predict(x_train)
# print("Training predit output:\n",train_pred)

# Convert numeric codes to original labels
train_labels = [segment_map[int(code)] for code in train_pred]

print("\n--- Bagging classifier model prediction ---\n")
print("\n Raw prediction codes:\n", train_pred[:20])
print("\n Mapped segment labels:\n", train_labels[:20])


# Evaluating performance

print("\n Classification Report :\n", classification_report(y_train, train_pred))


#Evaluating the model on the Cross-Validation set

cv_pred = bg_clf.predict(x_cv)
print("\n-- CV predict:Bagging Classifier -- \n",cv_pred)


#CV (validation) report

print("\n CV Validation Report of testdata :\n",classification_report(y_cv, cv_pred))


# Confusion matrix
cm = confusion_matrix(y_cv, cv_pred)
print("\n-- Confusion Matrix:Bagging Classifier -- \n",cm)


#Heat Map
sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues')
plt.title("Confusion Matrix on CV Set - Bagging classifier", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

print("*-" * 20)


# Check the features importance in the first tree

# Access the first decision tree from BaggingClassifier
dtree_fitted = bg_clf.estimators_[0]

# Get feature importance values
feature_importance = dtree_fitted.feature_importances_

features = x_train.columns   # Match feature names with importance

#Create and sort the dictionary
importance_dict = dict(zip(features, feature_importance))

# Sort by importance (descending)
sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))


# Explicitly print in PyCharm

print("\n--- Features Importance ---")
for feature, importance in sorted_importance.items():
    print(f"{feature}: {importance:.4f}")

print("*-" * 20)


# Bar chart - horizontal

plt.figure(figsize=(8, 5))
plt.barh(list(sorted_importance.keys()), list(sorted_importance.values()), color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance from One Decision Tree in BaggingClassifier", fontsize=14)
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()


# ---- Boosting ----

print("\n----BOOSTING----\n")

### AdaBoosting

print("\n----ADA BOOSTING----\n")

from sklearn.ensemble import AdaBoostClassifier

# Training the model
ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate= 1,  random_state=42)
ada_clf.fit(x_train, y_train)

# Testing on the training set

train_pred = ada_clf.predict(x_train)
#print("Training predict output:\n",train_pred)

# Convert numeric codes to original labels
train_labels = [segment_map[int(code)] for code in train_pred]

print("\n--- ADA BOOSTING model prediction ---\n")
print("\n Raw prediction codes:\n", train_pred[:20])
print("\n Mapped segment labels:\n", train_labels[:20])


# Evaluating performance
print("\n Training Classification Report :\n", classification_report(y_train, train_pred))


# Evaluating the model on the Cross-Validation Set

cv_pred = ada_clf.predict(x_cv)

print("\n-- CV predict:ADA BOOSTING -- \n",cv_pred)


#CV (validation) report

print("\n CV Validation Report of testdata :\n",classification_report(y_cv, cv_pred))


# Confusion matrix
cm = confusion_matrix(y_cv, cv_pred)
print("\n-- Confusion Matrix:ADA BOOSTING -- \n",cm)


#Heat Map
sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues')
plt.title("Confusion Matrix on CV Set - ADA BOOSTING ", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

print("*-" * 20)



### Gradient Boosting

print("\n----Gradient Boosting----\n")

from sklearn.ensemble import GradientBoostingClassifier

# Training the model
gb_clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, random_state=42)
gb_clf.fit(x_train, y_train)

# Testing on the training set

train_pred = gb_clf.predict(x_train)
# print("Training predict output:\n",train_pred)

# Convert numeric codes to original labels
train_labels = [segment_map[int(code)] for code in train_pred]

print("\n--- Gradient Boosting model prediction ---\n")
print("\n Raw prediction codes:\n", train_pred[:20])
print("\n Mapped segment labels:\n", train_labels[:20])


# Evaluating performance
print("\n Training Classification Report :\n", classification_report(y_train, train_pred))


# Evaluating the model on the Cross-Validation Set

cv_pred = gb_clf.predict(x_cv)

print("\n-- CV predict:Gradient Boosting -- \n",cv_pred)


#CV (validation) report

print("\n CV Validation Report of testdata :\n",classification_report(y_cv, cv_pred))


# Confusion matrix
cm = confusion_matrix(y_cv, cv_pred)
print("\n-- Confusion Matrix:Gradient Boosting -- \n",cm)


#Heat Map
sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues')
plt.title("Confusion Matrix on CV Set - Gradient Boosting ", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

print("*-" * 20)



## XGBoost

print("\n---- XGBoost ----\n")

from xgboost import XGBClassifier


# Training the model
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=.1, random_state=42)
xgb_clf.fit(x_train, y_train)

# Testing on the training set

train_pred = xgb_clf.predict(x_train)
# print("Training predict output:\n",train_pred)

# Convert numeric codes to original labels
train_labels = [segment_map[int(code)] for code in train_pred]

print("\n--- XG BOOST model prediction ---\n")
print("\n Raw prediction codes:\n", train_pred[:20])
print("\n Mapped segment labels:\n", train_labels[:20])


# Evaluating performance
print("\n Training Classification Report :\n", classification_report(y_train, train_pred))


# Evaluating the model on the Cross-Validation Set

cv_pred = xgb_clf.predict(x_cv)

print("\n-- CV predict:XGBoost -- \n",cv_pred)


#CV (validation) report

print("\n CV Validation Report of testdata :\n",classification_report(y_cv, cv_pred))


# Confusion matrix
cm = confusion_matrix(y_cv, cv_pred)
print("\n-- Confusion Matrix: XGBoost -- \n",cm)


#Heat Map
sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues')
plt.title("Confusion Matrix on CV Set - XGBoost ", fontsize=14)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

print("*-" * 20)

# Grid Search - Hyper Parameter Tuning

print("\n ---- Grid Search - Hyper Parameter Tuning ---- \n ")

from sklearn.model_selection import GridSearchCV

### multi-model hyperparameter tuning

# Define Model lists

models = [
    {"name": "Decision Tree", "model": DecisionTreeClassifier(), "parameters": {"max_depth": [None, 10, 20, 30]}},
    {"name": "Random Forest", "model": RandomForestClassifier(), "parameters": {"n_estimators": [10, 50, 100]}},
    {"name": "Bagging Classifier", "model": BaggingClassifier(), "parameters": {"n_estimators": [10, 50, 100]}},
    {"name": "AdaBoost", "model": AdaBoostClassifier(), "parameters": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}},
    {"name": "Gradient Boosting", "model": GradientBoostingClassifier(), "parameters": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}},
    {"name": "XGBoost", "model": XGBClassifier(), "parameters": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}}
]

#  Run GridSearchCV for Each Model

for entry in models:
    print(f"\n--- Grid Search: {entry['name']} ---")

    grid = GridSearchCV(estimator=entry['model'],
                        param_grid=entry['parameters'],
                        cv=5,
                        scoring='accuracy',
                        n_jobs=-1)

    grid.fit(x_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print(f"Best CV Score: {grid.best_score_:.4f}")


print("*-" * 25)


# SCATTER PLOT FOR GRID SEARCH

# Function to plot grid search results

def plot_grid_search(cv_results, param_name, model_name):
    plt.title(f"Grid Search Scores for {model_name}", fontsize=16)
    plt.xlabel(param_name, fontsize=14)
    plt.ylabel("Mean Test Score", fontsize=14)
    plt.grid()

    param_values = cv_results['param_' + param_name].data
    mean_test_scores = cv_results['mean_test_score']

    plt.scatter(param_values, mean_test_scores, marker='o')

best_models = {} # Initialize Dictionary - to store the best parameters


# Loop Through Models and Perform grid search and plot results for each model

for model_info in models:
    model_name = model_info["name"]
    model = model_info["model"]
    param_grid = model_info["parameters"]

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    plt.figure(figsize=(15, 5))
    idx = 0
    for param_name in param_grid.keys():
        plt.subplot(1, 2, idx % 2 + 1)
        plot_grid_search(grid_search.cv_results_, param_name, model_name)
        idx += 1

    plt.show()

 # Save Best Parameters
    best_models[model_name] = grid_search.best_params_

#  Print once after all models are processed
print("\n--- Final Best Parameters for All Models ---")
for name, params in best_models.items():
    print(f"{name}: {params}")

print("*-" * 25)


## 3. Evaluation on the test dataset

print("\n----Evaluation on the test dataset----\n")

## Pre-Processing

# Encode the Categorical Columns in the Test Data
lbld_test = df_test.copy()
for col in cat_cols:
    lbld_test[col] = le[col].transform(df_test[col])

# Scaling the Numeric Columns
lbld_test[numeric_columns] = scaler.transform(df_test[numeric_columns])

lbld_test.head()

print("--- Preprocessed data ---\n",lbld_test.head())

print("*-" *25)


# Separate features and target from the labeled test set

x_eval = lbld_test.drop('Segmentation', axis= 1)
y_eval = lbld_test['Segmentation']


## Getting the Results of Best Models


# model registry: a list of dictionaries with name,model,parameter

models = [
    {"name": "Decision Tree", "model": DecisionTreeClassifier, "parameters": {"max_depth": [None, 10, 20, 30]}},
    {"name": "Random Forest", "model": RandomForestClassifier, "parameters": {"n_estimators": [10, 50, 100]}},
    {"name": "Bagging Classifier", "model": BaggingClassifier, "parameters": {"n_estimators": [10, 50, 100]}},
    {"name": "AdaBoost", "model": AdaBoostClassifier, "parameters": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}},
    {"name": "Gradient Boosting", "model": GradientBoostingClassifier, "parameters": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}},
    {"name": "XGBoost", "model": XGBClassifier, "parameters": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}}
]

# Loop Through Models and Evaluate

for model_info in models:
    params = best_models[model_info['name']]
    model = model_info['model'](**params)

    model.fit(x_train, y_train)  # train on training data
    y_pred = model.predict(x_eval)  # Evaluate on testdata

    print(f"Model: #----{model_info['name']}----#")
    print(classification_report(y_eval, y_pred))


## Model Performance Summary and Best Overall Model

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import joblib

# Import all candidate model classes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Dictionary mapping model names to their classes
model_classes = {
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Bagging Classifier": BaggingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "XGBoost": XGBClassifier
}

results = []
best_models = {}

# Evaluate all models and capture metrics
for model_info in models:
    model_name = model_info["name"]
    model_class = model_info["model"]      #  # this is the class, e.g. DecisionTreeClassifier
    model = model_class()   ## instantiate it
    param_grid = model_info["parameters"]

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Get classification report as dictionary
    y_pred = grid_search.predict(x_eval)
    report = classification_report(y_eval, y_pred, output_dict=True)

    # Extract metrics dynamically
    results.append({
        "Model": model_name,
        "Accuracy": report['accuracy'],
        "Weighted F1": report['weighted avg']['f1-score'],
        "Macro F1": report['macro avg']['f1-score'],
        "Class 3 F1": report['3']['f1-score']  # assumes class label "3"
    })

    # Save best parameters for this model
    best_models[model_name] = grid_search.best_params_

# Create DataFrame
df_results = pd.DataFrame(results)

print("\n--- Model Performance Summary ---")
print(df_results.to_string(index=False))

# Compute overall score
metrics = ["Accuracy", "Weighted F1", "Macro F1", "Class 3 F1"]
df_results["Score"] = df_results[metrics].sum(axis=1)

# Identify best model dynamically
best_row = df_results.loc[df_results["Score"].idxmax()]
best_model_name = best_row["Model"]
best_params = best_models[best_model_name]

BestModelClass = model_classes[best_model_name]
best_model = BestModelClass(**best_params)      # # best_params unpacks the dictionary into keyword arguments Example: {'learning_rate': 0.05, 'n_estimators': 150} of the best model
best_model.fit(x_train, y_train)

# Save the trained best model for later use
joblib.dump(best_model, "best_model.pkl")           ## to reuse later // model = joblib.load("best_model.pkl")


# # Identify best model dynamically
# best_row = df_results.loc[df_results["Score"].idxmax()]
# best_model_name = best_row["Model"]
# best_params = best_models[best_model_name]

print("Model names in df_results:", df_results["Model"].tolist())
print("Best model name:", repr(best_model_name))
print("Best model detected:", best_model_name)
print("Best row:\n", best_row)


### Visualisation - Bar chart to highlight best model

fig, ax = plt.subplots(figsize=(10,6))
df_results.set_index("Model")[metrics].plot(kind="bar", ax=ax)

bars = ax.patches
tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
highlight_index = tick_labels.index(best_model_name)

num_models = len(tick_labels)
num_metrics = len(metrics)

for i in range(num_metrics):
    bar_index = i * num_models + highlight_index
    bars[bar_index].set_color("orange")

plt.title("Model Performance Comparison", fontsize=16)
plt.ylabel("Score", fontsize=14)
plt.xticks(rotation=30, ha="right")
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


# Print best model with reason
reason = (
    f"{best_model_name} is chosen because it achieves the highest overall score "
    f"across Accuracy ({best_row['Accuracy']:.2f}), Weighted F1 ({best_row['Weighted F1']:.2f}), "
    f"Macro F1 ({best_row['Macro F1']:.2f}), and Class 3 F1 ({best_row['Class 3 F1']:.2f})."
)
print(f"\n Best Overall Model: {best_model_name}")
print(f"Reason: {reason}")
