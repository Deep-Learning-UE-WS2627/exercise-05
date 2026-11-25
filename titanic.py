import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("C:/Users/Yuima/exercise-05/data/train.csv")


# ## Step 3
## 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
cldf = df.drop(columns= ['Name', 'PassengerId'])

## 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".

#print(cldf.dtypes)
#print("\nConverted:\n")
#cldf = cldf.convert_dtypes()
cldf = pd.get_dummies(cldf, columns=["Sex","Cabin","Ticket","Embarked"], drop_first=True)

#print("\n")
#print(cldf.dtypes)

## 3. Remove all rows that contain missing values
cldf = cldf.dropna(axis = 'index')
print(cldf)


print(cldf)


# ## Step 4
## 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.drop(columns=["Survived"])
y = df["Survived"]

## 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

## 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

## 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("precision:", precision)
print("recall:", recall)
print("F1-Score:", f1)
