import pandas as pd
from sklearn.tree import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

#import Datasets
CallsData = pd.read_excel("./CallsData.xls")
ContractData = pd.read_csv("./ContractData.csv")

#Merge datasets by AreaCode and Phone
MergedDF = pd.merge(CallsData, ContractData, "inner", ["Area Code", "Phone"])
#Transform Churn atribute from ordinal to nominal
MergedDF["Churn"].replace([0,1],["Permaneceu na op.", "Mudou de op."], inplace=True)

x=MergedDF.drop("Churn", axis=1)
y = MergedDF["Churn"].to_frame()


#split the data into training and testing sets
#we will be using a 75/25 split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=840)

#Define the Decision tree classifier
dt = DecisionTreeClassifier(random_state=840)

#dropping nominal data
x_train = x_train.drop(["Phone", "State"], axis=1)
x_test= x_test.drop(["Phone", "State"], axis=1)

#training the model
dt.fit(x_train,y_train)

#training the model
dt.fit(x_train, y_train)

predictions = dt.predict(x_test)

#Checking the models accuracy
print(accuracy_score(y_test,predictions))

#f1 score is 2*(precision*recall)/(precision+recall)
#wich gives us a harmonic mean of the precision and recall
#the closer to 1 the better
print(f1_score(y_test, predictions, average="macro"))

#TODO: 10 fold cross validation