"""
Predicting Credit Card Approvals from the 'Credit Card Approval dataset' from UCI Machine Learning Repository.

IPython Notebook can be found as predicting_credit_card_approvals.ipynb
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load dataset
cc_apps = pd.read_csv('datasets/cc_approvals.data', header=None)

# Inspect data (Uncomment for information)
# print(cc_apps.head())
'''
=OUTPUT=
  0      1      2  3  4  5  6     7  8  9   10 11 12     13   14 15
0  b  30.83  0.000  u  g  w  v  1.25  t  t   1  f  g  00202    0  +
1  a  58.67  4.460  u  g  q  h  3.04  t  t   6  f  g  00043  560  +
2  a  24.50  0.500  u  g  q  h  1.50  t  f   0  f  g  00280  824  +
3  b  27.83  1.540  u  g  w  v  3.75  t  t   5  t  g  00100    3  +
4  b  20.17  5.625  u  g  w  v  1.71  t  f   0  f  s  00120    0  +
'''

# Print summary statistics (Uncomment for information)
# cc_apps_description = cc_apps.describe()
# print(cc_apps_description)
# print("\n")

# Print DataFrame information (Uncomment for information)
# cc_apps_info = cc_apps.info()
# print(cc_apps_info)
# print("\n")

# Inspect missing values in the dataset (Uncomment for information)
# print(cc_apps.tail())

'''
=OUTPUT=
               2           7          10             14
count  690.000000  690.000000  690.00000     690.000000
mean     4.758725    2.223406    2.40000    1017.385507
std      4.978163    3.346513    4.86294    5210.102598
min      0.000000    0.000000    0.00000       0.000000
25%      1.000000    0.165000    0.00000       0.000000
50%      2.750000    1.000000    0.00000       5.000000
75%      7.207500    2.625000    3.00000     395.500000
max     28.000000   28.500000   67.00000  100000.000000

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 690 entries, 0 to 689
Data columns (total 16 columns):
0     690 non-null object
1     690 non-null object
2     690 non-null float64
3     690 non-null object
4     690 non-null object
5     690 non-null object
6     690 non-null object
7     690 non-null float64
8     690 non-null object
9     690 non-null object
10    690 non-null int64
11    690 non-null object
12    690 non-null object
13    690 non-null object
14    690 non-null int64
15    690 non-null object
dtypes: float64(2), int64(2), object(12)
memory usage: 86.3+ KB
None

    0      1       2  3  4   5   6     7  8  9   10 11 12     13   14 15
685  b  21.08  10.085  y  p   e   h  1.25  f  f   0  f  g  00260    0  -
686  a  22.67   0.750  u  g   c   v  2.00  f  t   2  t  g  00200  394  -
687  a  25.25  13.500  y  p  ff  ff  2.00  f  t   1  t  g  00200    1  -
688  b  17.92   0.205  u  g  aa   v  0.04  f  f   0  f  g  00280  750  -
689  b  35.00   3.375  u  g   c   h  8.29  f  f   0  t  g  00000    0  -
'''

'''
=Handling the Missing Values=
There are missing values in the dataset, but they are marked with '?'.
Instead of the standard NaN value. Below we will replace with NaN's 

Replace the NaN's with the mean() of the observations using fillna().
'''
# Inspect missing values in the dataset (Uncomment for information)
# print(cc_apps.tail(17))

# Replace the '?'s with NaN
cc_apps = cc_apps.replace({'?': np.nan})

# Inspect the missing values again (Uncomment for information)
# print(cc_apps.tail(17))

# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset to verify (Uncomment for information)
# print(cc_apps.isnull().sum())

'''
For object type columns, fillna() with mean() will not work.

We will replace the rest with the most frequent observation. 
'''
# Iterate over each column of cc_apps
for col in cc_apps.columns:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify (Uncomment for information)
# print(cc_apps.isnull().sum())
'''
Now the columns have zero NaN values. 
=OUTPUT=
0     0
1     0
2     0
3     0
4     0
5     0
6     0
7     0
8     0
9     0
10    0
11    0
12    0
13    0
14    0
15    0
dtype: int64
'''

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtypes == 'object':
        # Use LabelEncoder to do the numeric transformation
        cc_apps[col] = le.fit_transform(cc_apps[col])

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values

# Segregate features and labels into separate variables
X, y = cc_apps[:, 0:13], cc_apps[:, 13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it (Uncomment for information)
# print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model (Uncomment for information)
# print(confusion_matrix(y_test, y_pred))
'''
=OUTPUT=
Accuracy of logistic regression classifier:  0.8377192982456141
[[92 11]
 [26 99]]
'''

# Use grid search to find optimal values.
# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict({'tol': tol, 'max_iter': max_iter})

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv= 5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score = grid_model_result.best_score_
best_params = grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
'''
=OUTPUT=
Best: 0.853623 using {'max_iter': 100, 'tol': 0.01}
'''
