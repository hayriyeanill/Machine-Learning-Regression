# Data Science with Python Machine Learning Regression Project
The purpose of this project to look into different features of individual such as age, physical/family condition and location against their existing medical expense to be used for predicting future medical expenses of individuals that help medical insurance to make decision on charging the premium.

The insurance.csv dataset contains 1338 observations (rows) and 7 features (columns). The dataset contains 4 numerical features (age, bmi, children and charges) and 3 nominal features (sex, smoker and region) that were converted into factors with numerical value designated for each level.
Dataset available on https://www.kaggle.com/mirichoi0218/insurance#insurance.csv

Regression Techniques:
1) Multiple Linear Regression
2) Polynomial Linear Regression
3) Decision Tree
4) Random Forest

Elimination techniques:
1) Including all variables
2) Backward Elimination
3) Forward Selection

Measuring the Performance of Regression Models:
1) Root Mean Squared Error (RMSE)
2) R-Square Score
3) Mean Squared Error (MSE)
4) Mean Absolute Error (MAE)

The project was implemented and tested using Python. Sklearn library and packages of pandas, numpy are used. Matplotlib and seaborn was used for plotting the data.
The code was implemented in Python by creating a class. All classes have a function called __init __ () which is always executed when starting the class. 
The __init __ () function is used to assign values to object properties or other operations that need to be done while creating the object. 
Data import, independent and dependent are divided into x and y, encoder and elimination technique start in init.
Visualization, train_test_split, each regression model and elimination techniques used are written in the class as separate methods.
Most recently, for evaluation and comparison each different elimination and regression techniques has been collected in different run methods.
