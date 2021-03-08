# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:36:42 2021

@author: sergei
"""

# =============================================================================
# Proyect about titanic data set from Kaggle
# =============================================================================

# =============================================================================
# Goal = use machine learning to predict whether or not a person 
# on the titanic ship survived.
# =============================================================================

# =============================================================================
# Methodology = Superviced learning with classification.
# =============================================================================

# import packages 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# import packages for machine learning

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

# Configuration

pd.set_option("display.max_columns", 12)
plt.style.use("ggplot")

# load train dataset using pandas 

titanic = pd.read_csv("train.csv")

# =============================================================================
# Meaning of features 
# =============================================================================

# Survived = Whether or not a person survive (0=no, 1=yes)

# Pclass = Ticket class (1 = 1st, 2= 2nd, 3= 3rd)

# SibSp = siblings/spouses

# Parch = #parents/ children aboard the boat

# Fare = Tarifa

# Cabin = Cabin number 

# Embarked = Where the person embarked


# =============================================================================
# EDA (Exploratory Data Analysis)
# =============================================================================

titanic.head

titanic.shape

# See types of data

titanic.dtypes

# See if there are missing values 

titanic.isnull().sum()

# Due to the fact that almost every cabin value is null, I'm going to drop it

titanic.drop("Cabin", axis=1, inplace=True)

# Convert types to the rights ones

columns= ["PassengerId", "Survived", "Pclass", "Sex", "Embarked"]

for i in columns:
    titanic[i]=titanic[i].astype("category")

# See a summary statistic

titanic.describe()

# See diferents plots describing numeric variables

pd.plotting.scatter_matrix(titanic)

titanic[["Age", "Fare", "SibSp", "Parch"]].plot.box()

# See a correlation matrix

titanic.corr()


## Eliminating the nan values 

# In the case of the column embarked im going to replace it with the most commun value

titanic["Embarked"].fillna(titanic["Embarked"].value_counts().index[0],
                           inplace=True)

# In the case of the column Age and Fare im going to replaced with the mean of the column

titanic["Age"].fillna(np.mean(titanic["Age"]), inplace=True)

titanic["Fare"].fillna(np.mean(titanic["Fare"]), inplace=True)


# Another describe

titanic.describe()


# im going to convert the categorical variables to numerical to be able 
# of doing correlations


labelencoder= LabelEncoder()

titanic["SexEncoded"] = labelencoder.fit_transform(titanic["Sex"])
titanic["SurvivedEncoded"] = labelencoder.fit_transform(titanic["Survived"])
titanic["PclassEncoded"] = labelencoder.fit_transform(titanic["Pclass"])
titanic["EmbarkedEncoded"] = labelencoder.fit_transform(titanic["Embarked"])


# =============================================================================
# Dealing with the name feature 
# This is a very cool way of dealing with names and probably finding
# good information.
# This is why it is important to deal with every feature and not just drop it.
# =============================================================================

# Here what we are doing is extracting the title of the person

titlename= titanic.Name.str.split(".").str.get(0).str.split(",").str.get(1)
print(titlename.value_counts())

# there are some titles with little values so we are going to put it
# in another category


titlename.replace(["Dr", "Rev", "Major", "Col", "Capt"], "Officer", inplace=True,
              regex=True)

titlename.replace(to_replace = ["Dona", "Jonkheer", "Countess", 
                                "Sir", "Lady", "Don"], value = "Aristocrat", 
                  inplace = True,regex=True)

titlename.replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}, 
                  inplace = True,regex=True)

titlename.replace({"the Aristocrat":"Aristocrat"}, inplace = True,regex=True)

print(titlename.value_counts())

### now we are going to append this data to our table and drop the name column

titanic["Titlename"] = titlename
titanic.drop("Name", axis=1, inplace=True)

titanic.head()

# now we are going to convert this new column to category

titanic["Titlename"] = titanic["Titlename"].astype("category")

# and we are going to pass it to numerical 

titanic["TitleEnconded"] = labelencoder.fit_transform(titanic["Titlename"])


# =============================================================================
# Dealing with SibSP and Parch 
# Here this two features basically represents the size of the family
# so we are going to summarized and then create a new feature called size family.
# =============================================================================

titanic["FamilySize"] = titanic["SibSp"]+titanic["Parch"] + 1

titanic["FamilySize"].value_counts()

# we are going to replace the number with categories

titanic["FamilySize"].replace(1, "single", inplace = True, regex=True)
titanic["FamilySize"].replace([2,3], "small", inplace = True, regex=True)
titanic["FamilySize"].replace([4,5,6], "medium", inplace = True, regex=True)
titanic["FamilySize"].replace([7,11,8], "large", inplace = True, regex=True)

# now we are going to convert this new column to category

titanic["FamilySize"] = titanic["FamilySize"].astype("category")

# and we are going to pass it to numerical 

titanic["FamilySizeEncoded"] = labelencoder.fit_transform(titanic["FamilySize"])

# =============================================================================
# Dealing with Ticked
# =============================================================================

titanic.Ticket.head()

# here we are taking the first digit of the ticket 
# then if the first digit is a number we are putting an N and otherwise we
# are putting the digit

first_digit_ticket = titanic.Ticket.str.split(" ").str.get(0).str.get(0)

titanic["first_digit_ticket"] = np.where(titanic.Ticket.str.isdigit(), "N", first_digit_ticket)

# now we are going to convert this new column to category

titanic["first_digit_ticket"] = titanic["first_digit_ticket"].astype("category")

# and we are going to pass it to numerical 

titanic["first_digit_ticket_encoded"] = labelencoder.fit_transform(titanic["first_digit_ticket"])


# =============================================================================
# At whis point we have done feature enginnering to all features 
# =============================================================================

# lets go the correlation again

corr = titanic.corr()

# lets plot it 

sns.heatmap(corr, vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

# =============================================================================
# Findings: It seens that there is a stron negative correlation between 
# Survived and Sex
# It seens that there are a positive correlation between survived and Fare
# It seens that there are a positive correlation between survived and Pclass
# =============================================================================

# Comparing between sex and surviving


ax = sns.barplot(x="Sex", y="SurvivedEncoded", data=titanic)

ax.set_xlabel("Sex")
ax.set_ylabel("Survival")

# =============================================================================
# Findings: It seem like women survived a lot more than man, as seen by the 
# chart an by the correlation
# =============================================================================

ax = sns.boxplot(x=titanic.SurvivedEncoded, y= titanic.Fare, data= titanic)

ax.set_xlabel("Survived")
ax.set_ylabel("Fare")

ax = sns.boxplot(x=titanic.SexEncoded, y= titanic.Fare, data= titanic)
ax.set_xlabel("Sex")
ax.set_ylabel("Fare")

sns.histplot(x = titanic["Fare"], hue="SurvivedEncoded", data = titanic)

# =============================================================================
# Findings: It seem like people how pay more survived more
# Women tends to pay more
# =============================================================================

ax = sns.barplot(x=titanic.SurvivedEncoded, y= titanic.PclassEncoded, data= titanic)

ax.set_xlabel("Survived")
ax.set_ylabel("Pclass")

ax = sns.barplot(x=titanic.SexEncoded, y= titanic.PclassEncoded, data= titanic)

ax.set_xlabel("Sex")
ax.set_ylabel("Pclass")

# =============================================================================
# Findings: It seem like people of lower classes survived more
# Women tends to be of lower classes
# =============================================================================

ax = sns.barplot(x=titanic.SurvivedEncoded, y= titanic.FamilySizeEncoded, data= titanic)

ax.set_xlabel("Survived")
ax.set_ylabel("familySize")

# =============================================================================
# Findings: People with big families tends to survive more.
# =============================================================================

ax = sns.boxplot(x=titanic.SexEncoded, y= titanic.Age, data= titanic)
ax.set_xlabel("Sex")
ax.set_ylabel("Age")

sns.histplot(x = titanic["Age"], hue="SurvivedEncoded", data = titanic)

# =============================================================================
# Findings: There was a lot of babies who survided.
# =============================================================================

# =============================================================================
# =============================================================================
# =============================================================================
# Now we are ready to start doing some machine learning 
# =============================================================================
# =============================================================================
# =============================================================================

# First lets drop all the features that we dont need

titanic.columns

titanic.drop(["Survived", "Pclass", "Sex", "SibSp", "Parch", "Ticket",
              "Embarked", "Titlename", "FamilySize", 
              "first_digit_ticket"], axis=1, inplace= True)


titanic.dtypes

# we have to convert all data types again to numbers 

columns= ["SexEncoded", "SurvivedEncoded", "PclassEncoded", "EmbarkedEncoded",
          "TitleEnconded", "FamilySizeEncoded", "first_digit_ticket_encoded"]

for i in columns:
    titanic[i]=titanic[i].astype("int")
    
titanic.dtypes

# lets go the correlation again

corr= titanic.corr()

# lets plot it 

sns.heatmap(corr, vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)


# =============================================================================
# =============================================================================
# # Generating our X and y data for testing the model before doing the submission.
# =============================================================================
# =============================================================================

X = titanic.drop("SurvivedEncoded", axis=1)

y = titanic["SurvivedEncoded"]

#### splitting our data in train and test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=30, 
                                                    stratify=y)


# =============================================================================
# =============================================================================
# We are going to start using a KKN classifier with 11 neighbors
# =============================================================================
# =============================================================================

knn = KNeighborsClassifier(n_neighbors=11)

# fit the model 

knn.fit(X_train, y_train)

# predict 

y_pred = knn.predict(X_test)

# get score 

knn.score(X_train, y_train)


# now we are going to try with diferent n_neighbors values

neighbors = np.arange(1,20)   
accuracy=[]

for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i) 
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy.append(knn.score(X_test, y_test))
    

# Generate plot to see the best accuracy

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()    

print(accuracy)

# =============================================================================
# Here we manage to have a model with 80% accuracy on the train data and 68%
# on the test data
# =============================================================================

# =============================================================================
# =============================================================================
# =============================================================================
# # # Testing multiple models.
# =============================================================================
# =============================================================================
# =============================================================================

seed = 40

# Logistic Regression

lr = LogisticRegression()

#2.Support Vector Machines

svc = SVC(gamma = "auto")

#3.Random Forest Classifier

rf = RandomForestClassifier(random_state = seed, n_estimators = 100)

#4.KNN

knn = KNeighborsClassifier(n_neighbors=14)

#5.Gaussian Naive Bayes

gnb = GaussianNB()

#6.Decision Tree Classifier

dt = DecisionTreeClassifier(random_state = seed)

#7.Gradient Boosting Classifier

gbc = GradientBoostingClassifier(random_state = seed)

#8.Adaboost Classifier

abc = AdaBoostClassifier(random_state = seed)

#9.ExtraTrees Classifier

etc = ExtraTreesClassifier(random_state = seed)

#List of all the models and indices

modelNames = ["LR", "SVC", "RF", "KNN", "GNB", "DT", "GBC", "ABC", "ETC"]
models = [lr, svc, rf, knn, gnb, dt, gbc, abc, etc]


# creating a function to get the accuracies of all models with train and test data

def calculateTrainAccuracy(model):
    
    model.fit(X_train, y_train)
    trainAccuracy = model.score(X_train, y_train)
    return trainAccuracy

def calculateTestAccuracy(model):
    
    model.fit(X_train, y_train)
    testAccuracy = model.score(X_test, y_test)
    return testAccuracy

# applying map to get all the scores with train data

modelScoresTrain = list(map(calculateTrainAccuracy, models))


trainAccuracy = pd.DataFrame(modelScoresTrain, columns = ["trainAccuracy"], 
                             index=modelNames)

trainAccuracySorted = trainAccuracy.sort_values(by="trainAccuracy", ascending=False)

trainAccuracySorted

# applying map to get all the scores with test data

modelScoresTest = list(map( calculateTestAccuracy, models))


testAccuracy = pd.DataFrame(modelScoresTest, columns = ["testAccuracy"], 
                             index=modelNames)

testnAccuracySorted = testAccuracy .sort_values(by="testAccuracy", ascending=False)

testnAccuracySorted

### creating the dataframe of test and train to visualize better 

accuraciesvalues = pd.DataFrame()

accuraciesvalues["testAccuracy"]= testAccuracy["testAccuracy"]

accuraciesvalues["trainAccuracy"]= trainAccuracy["trainAccuracy"]

accuraciesvalues.sort_values("testAccuracy", ascending=False)

# =============================================================================
# With this information we can see that it seems that RF and ETC are the best
# models.
# =============================================================================

# =============================================================================
# To confirm this we have to do some crossvalidation 
# =============================================================================

def CrossValSCore(model):
    value = cross_val_score(model, X_train, y_train, cv=10).mean()
    return value


modelscoresCross = list(map(CrossValSCore, models))


xCrossScores = pd.DataFrame(modelscoresCross, columns = ["xCrossScores"],
                          index=modelNames)

xCrossScores.sort_values("xCrossScores", ascending=False)

# =============================================================================
# With this information we can see that it seems that RF, ETC, GBC and LR
# are the bestmodels.
# =============================================================================

# =============================================================================
# Lets do the hyperparameter tunning using grid search.
# =============================================================================

# First me have to define all the hyperparameter that we want to optimice.

# For logistic regression

lrParams = {"penalty":["l1", "l2"],
            "C": np.logspace(0, 4, 10),
            "max_iter":[5000]}

# For Gradient Boosting Classifier

gbcParams = {"learning_rate": [0.01, 0.02, 0.05, 0.01],
              "max_depth": [4, 6, 8],
              "max_features": [1.0, 0.3, 0.1], 
              "min_samples_split": [ 2, 3, 4],
              "random_state":[seed]}

# For Support Vector Machines

svcParams = {"C": np.arange(6,13), 
              "kernel": ["linear","rbf"],
              "gamma": [0.5, 0.2, 0.1, 0.001, 0.0001]}

# For Decision Tree Classifier
dtParams = {"max_features": ["auto", "sqrt", "log2"],
             "min_samples_split": np.arange(2,16), 
             "min_samples_leaf":np.arange(1,12),
             "random_state":[seed]}

# For Random Forest Classifier
rfParams = {"criterion":["gini","entropy"],
             "n_estimators":[10, 15, 20, 25, 30],
             "min_samples_leaf":[1, 2, 3],
             "min_samples_split":np.arange(3,8), 
             "max_features":["sqrt", "auto", "log2"],
             "random_state":[44]}

# For KNN KNeighborsClassifier
knnParams = {"n_neighbors":np.arange(3,9),
              "leaf_size":[1, 2, 3, 5],
              "weights":["uniform", "distance"],
              "algorithm":["auto", "ball_tree","kd_tree","brute"]}

# Adaboost Classifier
abcParams = {"n_estimators":[1, 5, 10, 15, 20, 25, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300],
              "learning_rate":[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],
              "random_state":[seed]}

# ExtraTrees Classifier
etcParams = {"max_depth":[None],
              "max_features":[1, 3, 10],
              "min_samples_split":[2, 3, 10],
              "min_samples_leaf":[1, 3, 10],
              "bootstrap":[False],
              "n_estimators":[100, 300],
              "criterion":["gini"], 
              "random_state":[seed]}


# creating a function to get the scores with hyperparameter tuning.

def HyperParameterTuning(model, params):
    gridSearch = GridSearchCV(model, params, verbose=0, cv=10,
                              scoring="accuracy", n_jobs = -1)
    
    gridSearch.fit(X_train, y_train)
    
    bestParams, bestScore = gridSearch.best_params_,(gridSearch.best_score_*100, 2)
    
    return bestScore, bestParams

modelstotune = [lr, svc, rf, knn, dt, gbc, abc, etc]

parametersLists = [lrParams, svcParams, rfParams, knnParams,
                   dtParams, gbcParams, abcParams, etcParams]

# Note: Here i apply the function one by one due to computational resources.

HyperParameterTuning(gbc, gbcParams)

#The best models and the best parameters was

# Model ExtraTreesClassifier

((82.64464925755249, 2),
 {'bootstrap': False,
  'criterion': 'gini',
  'max_depth': None,
  'max_features': 3,
  'min_samples_leaf': 3,
  'min_samples_split': 10,
  'n_estimators': 100,
  'random_state': 40})

# Model RandomForest 

((83.12852022529442, 2),
 {'criterion': 'gini',
  'max_features': 'sqrt',
  'min_samples_leaf': 3,
  'min_samples_split': 3,
  'n_estimators': 15,
  'random_state': 44})

# Model Gradient Boosting Classifier.

((83.12852022529442, 2),
 {'learning_rate': 0.01,
  'max_depth': 4,
  'max_features': 1.0,
  'min_samples_split': 2,
  'random_state': 40})

# =============================================================================
# From this point I'm only going to work with this models and parameters.
# =============================================================================

gbc = GradientBoostingClassifier(learning_rate = 0.01,
  max_depth = 4, max_features = 1.0, min_samples_split = 2, 
  random_state = 40)
                                 
                                 
rf = RandomForestClassifier(criterion= "gini", max_features="sqrt",
  min_samples_leaf= 3, min_samples_split = 3, n_estimators= 15,
  random_state = 40)


etc = ExtraTreesClassifier(bootstrap = False, criterion = "gini",
  max_depth = None, max_features = 3, min_samples_leaf = 3,
  min_samples_split = 10, n_estimators = 100, random_state = 40)

modelspicked = [gbc, rf, etc]

# lets check the accuracy in the train data and test data.

Trainaccurracy =list(map(calculateTrainAccuracy, modelspicked))

Trainaccurracy

Testaccurracy =list(map(calculateTestAccuracy, modelspicked))

Testaccurracy

# it seens that the best models are RF and ETC.

# =============================================================================
# Getting the feature importances
# =============================================================================

def GetFeatureImportance(model):
    
    featureimportances = pd.DataFrame({"feature": X_train.columns,
                             "importance": model.feature_importances_})
    
    featureimportancesdorted = featureimportances.sort_values("importance", 
                                                          ascending=False)
    
    return featureimportancesdorted


def Importanceplot(model):
        importance = GetFeatureImportance(model)
        sns.barplot(x=importance["importance"],
                    y=importance["feature"], data=importance)
    

Importanceplot(rf)



# =============================================================================
# Preparing the test data 
# =============================================================================

## In this point I repeated everything done with the train data

test = pd.read_csv("test.csv")

test.columns

# dropin Cabin

test.drop("Cabin", axis=1, inplace=True)

# dealing with missing values.


test["Embarked"].fillna(test["Embarked"].value_counts().index[0],
                        inplace=True)


test["Age"].fillna(np.mean(titanic["Age"]), inplace=True)

test["Fare"].fillna(np.mean(test["Fare"]), inplace=True)

# Converting features to numeric.

test["SexEncoded"] = labelencoder.fit_transform(test["Sex"])
test["PclassEncoded"] = labelencoder.fit_transform(test["Pclass"])
test["EmbarkedEncoded"] = labelencoder.fit_transform(test["Embarked"])

# Dealing with the name feature 

titlename= test.Name.str.split(".").str.get(0).str.split(",").str.get(1)

# there are some titles with a little values so we are going to put it
# in another category


titlename.replace(["Dr", "Rev", "Major", "Col", "Capt"], "Officer", inplace=True,
              regex=True)

titlename.replace(to_replace = ["Dona", "Jonkheer", "Countess", 
                                "Sir", "Lady", "Don"], value = "Aristocrat", 
                  inplace = True,regex=True)

titlename.replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}, 
                  inplace = True,regex=True)

titlename.replace({"the Aristocrat":"Aristocrat"}, inplace = True,regex=True)


test["Titlename"] = titlename
test.drop("Name", axis=1, inplace=True)


# now we are going to convert this new column to category

test["Titlename"] = test["Titlename"].astype("category")

# and we are going to pass it to numerical 

test["TitleEnconded"] = labelencoder.fit_transform(test["Titlename"])


# Dealing with SibSP and Parch 


test["FamilySize"] = test["SibSp"]+test["Parch"] + 1

test["FamilySize"].value_counts()

# we are goint to replace the number with categories

test["FamilySize"].replace(1, "single", inplace = True, regex=True)
test["FamilySize"].replace([2,3], "small", inplace = True, regex=True)
test["FamilySize"].replace([4,5,6], "medium", inplace = True, regex=True)
test["FamilySize"].replace([7,11,8], "large", inplace = True, regex=True)

# now we are going to convert this new column to category

test["FamilySize"] = test["FamilySize"].astype("category")

# and we are going to pass it to numerical 

test["FamilySizeEncoded"] = labelencoder.fit_transform(test["FamilySize"])

# Dealing with ticket

first_digit_ticket = test.Ticket.str.split(" ").str.get(0).str.get(0)

test["first_digit_ticket"] = np.where(test.Ticket.str.isdigit(), "N", first_digit_ticket)

# now we are going to convert this new column to category

test["first_digit_ticket"] = test["first_digit_ticket"].astype("category")

# and we are going to pass it to numerical 

test["first_digit_ticket_encoded"] = labelencoder.fit_transform(test["first_digit_ticket"])

# we have to convert all data types again to numbers 

columns= ["SexEncoded", "PclassEncoded", "EmbarkedEncoded",
          "TitleEnconded", "FamilySizeEncoded", "first_digit_ticket_encoded"]

for i in columns:
    test[i]=test[i].astype("int")
    
# and drop the features that we dont need

test.drop(["Pclass", "Sex", "SibSp", "Parch", "Ticket",
              "Embarked", "Titlename", "FamilySize", 
              "first_digit_ticket"], axis=1, inplace= True)


# =============================================================================
# Prediction using Random Forest.
# =============================================================================

                                 
rf = RandomForestClassifier(criterion= "gini", max_features="sqrt",
  min_samples_leaf= 3, min_samples_split = 3, n_estimators= 15,
  random_state = 40)


rf.fit(X, y)


# =============================================================================
# Creating the submission  with RF
# =============================================================================

submisionRF1 = pd.DataFrame({"PassengerId": test["PassengerId"],
                            "Survived":rf.predict(test)})


submisionRF1.to_csv("submisionRF1.csv", index=False)

# =============================================================================
# Prediction using Random Forest.
# =============================================================================

etc = ExtraTreesClassifier(bootstrap = False, criterion = "gini",
  max_depth = None, max_features = 3, min_samples_leaf = 3,
  min_samples_split = 10, n_estimators = 100, random_state = 40)

etc.fit(X, y)

# =============================================================================
# Creating the submission  with ETC
# =============================================================================


submisionETC1 = pd.DataFrame({"PassengerId": test["PassengerId"],
                            "Survived":etc.predict(test)})


submisionETC1.to_csv("submisionETC1.csv", index=False)

# =============================================================================
# Trying to improve the accusary using assembly.
# =============================================================================

