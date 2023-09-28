import pandas as pd
from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.metrics import *

#Load the csv
df = pd.read_csv("./titanic_dataset(1).csv")

x=df.drop("Survived", axis=1)
y=df["Survived"].to_frame()


#Lets use x and y to create train and test sets of data
#IMPORTANT define the random_state seed 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=420)

#Create an instance of a Decision Tree Classifier DTC
clf = DecisionTreeClassifier(random_state=420)

#Lets drop the categorical data from the datasets so SciKitLearn can handle them
#Instead of dropping them we could use techniques such as Label or one-hot encoding to
#fix this issue


x_train = x_train.drop(["Name", "Sex", "Age", "Ticket", "Cabin", "Embarked"], axis=1)
x_test = x_test.drop(["Name", "Sex", "Age", "Ticket", "Cabin", "Embarked"], axis=1)

#training the model
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
print(predictions)

#Confusion Matrix
print(confusion_matrix(y_test, predictions))

print(accuracy_score(y_test,predictions))
print(precision_score(y_test,predictions))

print(recall_score(y_test, predictions))
