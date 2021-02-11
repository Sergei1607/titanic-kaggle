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

columns= ["PassengerId", "Survived", "Pclass", "Sex", "Ticket", "Embarked"]

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

# In the case of the column im going to replace it with the most commun value

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
titanic["TicketEncoded"] = labelencoder.fit_transform(titanic["Ticket"])
titanic["EmbarkedEnconded"] = labelencoder.fit_transform(titanic["Embarked"])


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
# =============================================================================

# Comparing between sex and surviving


ax = sns.barplot(x="Sex", y="SurvivedEncoded", data=titanic)

ax.set_xlabel("Sex")
ax.set_ylabel("Survival")

# =============================================================================
# Findings: It seem like women survived a lot more than man, as seen by the 
# chart an by the correlation
# =============================================================================



sns.histplot(x="Fare", hue="Survived", data=titanic)
