# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "scikit-learn"]
# ///

"""
DAT200 CA3 Classification of wine
Group 12: Kiki Julia, Sathusan and Durga

Training of our best classifier and computation of the prediction submitted to Kaggle.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# The data files are read from the same folder as this script
DATA_DIR = Path(".")

# Read the data
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Remove outliers from the training data: drop a row if any feature is more than 4 sd from the mean
features = train.drop(columns=["quality", "Id"]).columns
z = (train[features] - train[features].mean()) / train[features].std()
train = train[~(z.abs() > 4).any(axis=1)].reset_index(drop=True)

# Standardise train and test with mean and sd from the (cleaned) training data
sc = StandardScaler().set_output(transform="pandas")
sc.fit(train[features])
train_sc = sc.transform(train[features])
test_sc = sc.transform(test[features])


# Best classifier: rbf-SVM trained on all cleaned training data, using the four selected features
selected = ["alcohol", "volatile acidity", "total sulfur dioxide", "sulphates"]
best_model = SVC(kernel="rbf", C=1, gamma=0.3)
best_model.fit(train_sc[selected], train["quality"])

# Predict the quality of the test wines
y_test_pred = best_model.predict(test_sc[selected])

# Kaggle expects the column "Id" with the row number of test.csv (same as ID in sample_submission)
submission = pd.DataFrame({"Id": sample_sub["ID"], "quality": y_test_pred})
submission.to_csv(DATA_DIR / "submission.csv", index=False)
print("submission.csv written:", len(submission), "rows")
