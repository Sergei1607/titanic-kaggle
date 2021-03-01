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
# Methodology = Superviced learning.
# =============================================================================

# import packages 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder

# Configuration

pd.set_option("display.max_columns", 12)
plt.style.use("ggplot")
# load dataset using pandas 

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

# Due to the fact that almost every cabin value is null, im going to drop it

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

# In the case of the column Age im going to replaced with the mean of the column

titanic["Age"].fillna(np.mean(titanic["Age"]), inplace=True)


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
# This is why it is important to deal with every feature.
# =============================================================================

# here what we are doing is extracting the title of the person

titlename= titanic.Name.str.split(".").str.get(0).str.split(",").str.get(1)
print(titlename.value_counts())

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

print(titlename.value_counts())

### now we are going to append this data to our table 
# and drop the name column

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
# so we are going to summarized and then create a new feature called 
# size family.
# =============================================================================


titanic["FamilySize"] = titanic["SibSp"]+titanic["Parch"] + 1

titanic["FamilySize"].value_counts()

# we are goint to replace the number with categories

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

sns.boxplot(titanic.columns)


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
ax.set_ylabel("Fare")

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

titanic.drop("first_digit_ticket", axis=1, inplace= True)

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



## recordar reducir la varianza de age y fare 