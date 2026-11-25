
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Step 3: Prepare the passenger data
df = pd.read_csv("data/train.csv")

df = df.drop(columns=["Name", "PassengerId"])
df = df.dropna()

df = pd.get_dummies(
    df,
    columns=["Sex", "Cabin", "Ticket", "Embarked"],
    dtype=int
)

# Step 4: Separate information from survival results
x = df.drop(columns=["Survived"])
y = df["Survived"]

# Use 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)

# Predict survival for the test passengers
predictions = model.predict(x_test)

# Evaluate the predictions
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1-Score:", f1_score(y_test, predictions))
