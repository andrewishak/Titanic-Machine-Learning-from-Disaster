import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


#maxmize the size of output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#reading data 
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_test_data = [train_data, test_data]
#print(train_data.head())
#print(train_data.isnull().sum())
#print(test_data.head())
#print(test_data.isnull().sum())

#fun to analyze features
def analyze_data(feature):
    return train_data[[feature, 'Survived']].groupby([feature], as_index=False).mean().sort_values(by='Survived', ascending=False)

# drop passengerid
train_data.drop('PassengerId', axis=1, inplace=True)

#for feature in ['Sex','Pclass','SibSp','Parch','Embarked'] :
#    print(analyze_data(feature))

 
#replace name feature with title feature
for data in train_test_data:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
#print(train_data['Title'].value_counts())

for data in train_test_data:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    

title_map = {"Mr": 0, "Miss": 1, "Mrs": 2,  "Master": 3, "Rare": 4 }
for data in train_test_data:
    data['Title'] = data['Title'].map(title_map)

train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)

#print(train_data.head())

#map sex feature
sex_map = {"male": 0, "female": 1}
for data in train_test_data:
    data['Sex'] = data['Sex'].map(sex_map)

#print(train_data.head())
    
#find nan and map age feature

train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)
test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)

#print(train_data.isnull().sum())
#print(test_data.isnull().sum())

for data in train_test_data:    
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']

#print(train_data.head())

#find nan and map age feature
    
embarked_miss = train_data['Embarked'].value_counts().idxmax()
for data in train_test_data:
    data['Embarked'] = data['Embarked'].fillna(embarked_miss)
    
#print(data.isnull().sum())
embarked_map = {"S": 0, "C": 1, "Q": 2}
for data in train_test_data:
    data['Embarked'] = data['Embarked'].map(embarked_map)

#find nan and map fare feature
    
train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
   
for data in train_test_data:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    
    data['Fare'] = data['Fare'].astype(int)

#print(train_data.head())

#replace sibsp and parch features with alone

train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1

for data in train_test_data:
    data['Alone'] = 0
    data.loc[data['FamilySize'] == 1, 'Alone'] = 1


train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

#print(train_data.head())

# drop ticket and cabin
train_data.drop(['Cabin','Ticket'], axis=1, inplace=True)
test_data.drop(['Cabin','Ticket'], axis=1, inplace=True)

#print(train_data.head())
#print(test_data.head())

#Modling

train_x = train_data.drop("Survived", axis=1)
train_y = train_data["Survived"]
val_x = test_data.drop("PassengerId", axis=1).copy()
gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
val_y = gaussian.predict(val_x)

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": val_y
    })
submission.to_csv('submission.csv', index=False)
