Columns:
PassengerId: Unique ID for each passenger.
Survived: Target variable (0 = No, 1 = Yes).
Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Passenger's name.
Sex: Gender of the passenger.
Age: Age of the passenger.
SibSp: Number of siblings/spouses aboard.
Parch: Number of parents/children aboard.
Ticket: Ticket number.
Fare: Fare paid for the ticket.
Cabin: Cabin number.
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
Steps to Follow
1. Set up the Environment
To get started, you need to have the following installed:

Python 3.x
Libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
You can install the required libraries using the following command:

bash
نسخ الكود
pip install numpy pandas matplotlib seaborn scikit-learn
2. Load and Inspect the Data
Load the Titanic dataset using Pandas.
Display the first few rows and check for missing values.
python
نسخ الكود
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
train.info()
3. Data Preprocessing
Preprocessing includes handling missing values, feature engineering, and encoding categorical variables.

Steps:
Impute Missing Values: For features like Age, you can impute missing values based on Pclass using a custom imputation function, like the impute_age function.
Drop Irrelevant Columns: Drop columns like Name, Ticket, Cabin, and PassengerId as they may not contribute to the prediction.
Convert Categorical Variables: Convert Sex and Embarked into numeric values using label encoding or one-hot encoding.
python
نسخ الكود
# Imputing missing 'Age' based on Pclass
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# Dropping irrelevant columns
train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Converting categorical variables
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)
train['Embarked'] = pd.get_dummies(train['Embarked'], drop_first=True)

train.info()
4. Feature Selection
Identify the most important features for the model. In this case, likely features include:

Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
5. Splitting the Data
Split the dataset into training and testing sets using train_test_split from sklearn.

python
نسخ الكود
from sklearn.model_selection import train_test_split

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
6. Train the Logistic Regression Model
Use the LogisticRegression model from sklearn to train the model on the training data.

python
نسخ الكود
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=200)
logmodel.fit(X_train, y_train)
7. Make Predictions
After training the model, make predictions on the test set.

python
نسخ الكود
predictions = logmodel.predict(X_test)
8. Evaluate the Model
Evaluate the model’s performance using metrics such as accuracy, precision, recall, and F1-score.

python
نسخ الكود
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
9. Submit Predictions
After evaluating, you can generate predictions on the test data and submit them to Kaggle.

python
نسخ الكود
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis=1)
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)
test['Embarked'] = pd.get_dummies(test['Embarked'], drop_first=True)

X_submission = test.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
predictions = logmodel.predict(X_submission)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
Conclusion
This project demonstrates the steps to build a Logistic Regression model to predict Titanic survival. The process involves data preprocessing, model building, and evaluation. Logistic regression is a simple and effective approach for binary classification problems like this one.
