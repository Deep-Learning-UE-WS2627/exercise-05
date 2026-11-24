import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split


# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(columns=["Name", "PassengerId"])
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
for col in ["Sex", "Cabin", "Ticket", "Embarked"]:
    df[col] = df[col].astype("category").cat.codes
# 3. Remove all rows that contain missing values
df = df.dropna()

print(df.shape)
print(df.dtypes)
# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
X = df.drop(columns="Survived")
y = df["Survived"]
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
clf = LogisticRegression(solver="liblinear")
clf.fit(X_train, y_train)
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
y_pred = clf.predict(X_test)
precision, recall, fscore, _ = precision_recall_fscore_support(
    y_test, y_pred, average="binary"
)
print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F-score: {fscore:.3f}")
print(classification_report(y_test, y_pred, target_names=["drowned", "survived"]))
