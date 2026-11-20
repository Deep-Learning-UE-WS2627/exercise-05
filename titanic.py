from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Read the Titanic data from the CSV file included in this repository.
file_path = Path(__file__).resolve().parent / "data" / "train.csv"
df = pd.read_csv(file_path)

# ## Step 3: Data preparation
# 1. Remove columns that are irrelevant to the prediction.
df = df.drop(columns=["Name", "PassengerId"])

# 2. Remove rows with missing values before encoding. This is important
#    because dummy encoding would otherwise hide missing categorical values.
df = df.dropna()

# 3. Convert non-numeric categorical columns to numeric (0/1) features.
df = pd.get_dummies(
    df,
    columns=["Sex", "Cabin", "Ticket", "Embarked"],
    dtype=int,
)

# ## Step 4: Machine learning
# 1. Separate the target (Survived) from the input features.
x = df.drop(columns=["Survived"])
y = df["Survived"]

# 2. Split the data into training data (80%) and test data (20%).
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train a logistic regression model using the required liblinear solver.
model = LogisticRegression(solver="liblinear", max_iter=1000)
model.fit(x_train, y_train)

# 4. Predict the labels and calculate precision, recall and F1 score.
y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
