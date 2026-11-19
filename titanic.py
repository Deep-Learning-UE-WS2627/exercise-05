import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# read the data from a CSV file (included in the repository)
df = pd.read_csv("data/train.csv")


# ## Step 3
# 1. Remove the columns "Name" and "PassengerId" (because we know they are irrelevant for our problem).
df = df.drop(columns=["Name", "PassengerId"])

# 2. Convert all non-numeric columns into numeric ones. The non-numeric columns are "Sex", "Cabin", "Ticket" and "Embarked".
df = pd.get_dummies(df, columns=["Sex", "Cabin", "Ticket", "Embarked"], drop_first=True)

# 3. Remove all rows that contain missing values
df = df.dropna()


# ## Step 4
# 1. As a next step, we need to split the input features from the training labels. This can be done easily with `pandas`.
x = df.drop(columns=["Survived"])
y = df["Survived"]

# 2. Secondly, we need to split training and test data. This can be done with the function [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) from the `scikit-learn` library.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Finally, initialize a LogisticRegression object with a `liblinear` solver, and fit it to the training data.
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Prediction: ", y_pred) #Prediction zur Testdata, die anzeigt, ob überlebt (1/0) 

# 4. Lastly, calculate precision/recall/f-score on the test data using the appropriate functions from `scikit-learn`.
accuracy = model.score(x_test, y_test) #Anteil aller korrekten Vorhersagen (als zusätzliche Information)
precision = precision_score(y_test, y_pred) #Anteil der vorhergesagten positives, die tatsächliche positives sind (true positives)
recall = recall_score(y_test, y_pred) #Anteil aller nicht-vorhergesagten positives, die als true positives vorhergesagt wurden
f_score = f1_score(y_test, y_pred) #Mittelwert aus precision und recall

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-Score: ", f_score)

