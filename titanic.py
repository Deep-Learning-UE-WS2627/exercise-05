import pandas
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split



# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")

df = df.drop(columns=["Name","PassengerId"])

# drops all rows with NULL values from the DataFrame
df = df.dropna()

non_numeric_cols = ["Sex","Cabin","Ticket","Embarked"]
# get_Dummies ersetzt jede nicht numerische Spalte durch die binären Werte 0|1 ersetzt.
df = pd.get_dummies(df, columns=non_numeric_cols, dtype=int)

x = df.drop(columns=["Survived"])
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42

)

model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


 
# read the data from a CSV file (included in the repository)
df = pandas.read_csv("data/train.csv")
df.dropna()
# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
# 3. Remove all rows that contain missing values
df.drop(columns=["Name","PassengerId"])
df.get_dummies("Sex","Cabin","Ticket","Embarked")

# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
x = df.drop(columns=["Survived"], axis=1)
y = df["Survived"]

# splitting input features from training labels,
# just means separating known variables from the target variable (here: the survivors),
# i.e the one we want to predict
