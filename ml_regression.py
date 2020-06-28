import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import seaborn as sb

class Regression():
    def __init__(self):
        print("init")
        data = pd.read_csv("insurance.csv")
        self.df = pd.DataFrame(data)

        self.x = self.df.iloc[:, : -1] # dependent variable
        self.y = self.df.iloc[: , 6]  # independent variable

        # sex and smoke column to one hot encoder
        sex = self.x.iloc[:, 1:2]
        smoker = self.x.iloc[:, 4:5]
        ohe = OneHotEncoder()
        smoker = ohe.fit_transform(smoker).toarray()
        sex = ohe.fit_transform(sex).toarray()
        self.x['smoker'] = smoker
        self.x['sex'] = sex

        # region column to label encoder
        region = self.x.iloc[:,5:6].values
        le = LabelEncoder()
        region[:,0]= le.fit_transform(region[:,0])
        self.x['region'] = region[:,0]
        self.x = self.x.astype(float)

        self.X = self.x.values
        self.Y = self.y.values

        # elimination techniques
        self.el_x = np.append(arr=np.ones((1338,1)).astype(float), values=self.x, axis=1) # adding column for x0
        self.x_opt = self.el_x[:, [0,1,2,3,4,5,6]]

        self.sl = 0.05
        self.numVars = len(self.el_x[0])
        self.X_df = pd.DataFrame(self.el_x)

    def visualization(self):
        # heat map
        sb.set(font_scale=1.0)
        corr = self.df.corr()
        sb.heatmap(corr, annot=True, fmt='.2f')
        plt.show()

        # histogram
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(self.df['age'], bins=7)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('#Person')
        plt.show()

        # pie chart
        gender = self.df['sex'].value_counts(normalize=True) * 100
        plt.pie(gender, labels=['Male', 'Female'], autopct='%1.1f%%', colors=['steelblue', 'orangered'])
        plt.title('Distribution of Gender')
        plt.show()

        # Stacked Column Chart BMI
        self.df.loc[self.df['bmi'] <= 18.5, 'BMI'] = 'Below normal weight'
        self.df.loc[(self.df['bmi'] >= 18.5) & (self.df['bmi'] < 25), 'BMI'] = 'Normal weight'
        self.df.loc[(self.df['bmi'] >= 25) & (self.df['bmi'] < 30), 'BMI'] = 'Overweight'
        self.df.loc[(self.df['bmi'] >= 30) & (self.df['bmi'] < 35), 'BMI'] = 'Class I Obesity'
        self.df.loc[(self.df['bmi'] >= 35) & (self.df['bmi'] < 40), 'BMI'] = 'Class II Obesity'
        self.df.loc[(self.df['bmi'] >= 35) & (self.df['bmi'] < 40), 'BMI'] = 'Class II Obesity'
        self.df.loc[self.df['bmi'] >= 40,'BMI'] = 'Class III Obesity'
        var = self.df.groupby(['BMI','sex']).charges.sum()
        var.unstack().plot(kind='bar',stacked=True,  color=['red','teal'], grid=False)
        plt.title("Distribution of Sum of Charges by BMI and Gender")
        plt.show()

        # Stacked Column Chart REGION
        var = self.df.groupby(['region','sex']).charges.sum()
        var.unstack().plot(kind='bar',stacked=True,  color=['g','darkslategrey'], grid=False)
        plt.title("Distribution of Sum of Charges by Region and Gender")
        plt.show()

        # bar chart
        var = self.df.groupby('sex').charges.sum()
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_xlabel('Gender')
        ax1.set_ylabel('Sum of Charges')
        ax1.set_title("Gender wise Sum of Charges")
        var.plot(kind='bar', color=['indianred', 'steelblue'])
        plt.show()

        # scatter
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(self.df['age'],self.df['charges'], color=['goldenrod'])
        ax.set_xlabel('Age')
        ax.set_ylabel('Charges')
        ax.set_title("Distribution of Charges by Age")
        plt.show()

        # categorical plot
        sb.swarmplot(x="smoker", y="age", hue='sex',data=self.df, palette="Set1")
        plt.show()

        # factor plot
        sb.catplot(kind = "point", data = self.df, x = "sex", y = "charges", col = "smoker", row = "region")
        plt.show()

        # box plot
        sb.boxplot(x='region', y ='charges', data=self.df)
        plt.show()

    def backward_elimination(self, x_opt, y, sl):
        x_opt_df = pd.DataFrame(x_opt)
        excluded = list(set(x_opt_df.columns))
        for i in range(0, self.numVars):
            regressor_ols = sm.OLS(y, x_opt).fit()
            maxVar = max(regressor_ols.pvalues)
            if maxVar > sl:
                for j in range(0, self.numVars - i):
                    if (regressor_ols.pvalues[j] == maxVar):
                        x_opt = np.delete(x_opt, j, 1)
                        excluded.remove(j)

        backward = pd.DataFrame(self.el_x)
        backward_df = backward.iloc[:,excluded]
        return backward_df

    def forward_selection(self, x_df, y, sl):
        initial_list = []
        included = list(initial_list)
        while True:
            changed=False
            excluded = list(set(x_df.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                regressor_ols = sm.OLS(y, x_df).fit()
                new_pval[new_column] = regressor_ols.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < sl:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
            if not changed:
                break

        fs_opt = self.el_x[:, []]
        for j in included:
            fs_opt = np.append(fs_opt, self.x_opt[:, [j]], 1)

        forward = pd.DataFrame(self.el_x)
        forward_df = forward.iloc[:, included]
        return forward_df

    def train_test_split(self,x,y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    def linear_regression(self):
        lr = LinearRegression()
        lr.fit(self.x_train, self.y_train)
        l_pred = lr.predict(self.x_test)
        l_df = pd.DataFrame({'Actual': self.y_test, 'Predicted': l_pred})
        print(l_df)
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, l_pred)
        plt.plot([0, 50000], [0, 50000], '--k')
        plt.axis('tight')
        plt.title("linear regression true and predicted value comparison")
        plt.xlabel("y_test")
        plt.ylabel("pred")
        plt.tight_layout()
        plt.show()

        # finding error
        rmse_l_pred = sqrt(mean_squared_error(self.y_test, l_pred))
        print("Linear Regression RMSE", rmse_l_pred)
        lr_score = r2_score(self.y_test, l_pred)
        print("Linear Regression r2 score ", lr_score)
        lr_ms = mean_squared_error(self.y_test, l_pred)
        print("Linear Regression mean squared error ", lr_ms)
        lr_m = mean_absolute_error(self.y_test, l_pred)
        print("Linear Regression mean absolute error ", lr_m)

        return rmse_l_pred

    def polynomial_linear_regression(self, X):
        poly_reg = PolynomialFeatures(degree=2)
        x_poly = poly_reg.fit_transform(X)
        lr2 = LinearRegression()
        lr2.fit(x_poly, self.y_train)
        poly_reg = lr2.predict(poly_reg.fit_transform(self.x_test))
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, poly_reg, color='r')
        plt.plot([0, 60000], [0, 60000], '--k')
        plt.axis('tight')
        plt.title("polynomial linear regression true and predicted value comparison")
        plt.xlabel("y_test")
        plt.ylabel("pred")
        plt.tight_layout()
        plt.show()

        # finding error
        rmse_p_reg = sqrt(mean_squared_error(self.y_test,poly_reg))
        print("Polynomial Regression RMSE", rmse_p_reg)
        pr_score = r2_score(self.y_test, poly_reg)
        print("Polynomial Regression r2 score ", pr_score)
        pr_ms = mean_squared_error(self.y_test, poly_reg)
        print("Polynomial Regression  mean squared error ", pr_ms)
        pr_m = mean_absolute_error(self.y_test, poly_reg)
        print("Polynomial Regression  mean absolute error ", pr_m)

    def decision_tree(self):
        # Decision tree
        r_dt = DecisionTreeRegressor(random_state=0, max_depth=2)
        r_dt.fit( self.x_train,  self.y_train)
        d_pred = r_dt.predict( self.x_test)
        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, d_pred, color='g')
        plt.plot([0, 50000], [0, 50000], '--k')
        plt.axis('tight')
        plt.title("decision tree true and predicted value comparison")
        plt.xlabel("y_test")
        plt.ylabel("pred")
        plt.tight_layout()
        plt.show()

        # finding error
        rmse_d_pred = sqrt(mean_squared_error(self.y_test, d_pred))
        print("Decision Tree RMSE", rmse_d_pred)
        dt_score = r2_score( self.y_test, d_pred)
        print("Decision Tree r2 score ", dt_score)
        dt_ms = mean_squared_error(self.y_test, d_pred)
        print("Decision Tree  mean squared error ", dt_ms)
        dt_m = mean_absolute_error(self.y_test, d_pred)
        print("Decision Tree  mean absolute error ", dt_m)

    def random_forest(self):
        # max depth versus error
        md = 20
        md_errors = np.zeros(md)
        # random forest regression
        for i in range(1, md + 1):
            rf_reg = RandomForestRegressor(n_estimators=100, max_depth=i, random_state=0)
            rf_reg.fit(self.x_train, self.y_train)
            rf_pred = rf_reg.predict(self.x_test)
            # finding error
            md_errors[i - 1] = sqrt(mean_squared_error(self.y_test, rf_pred))

        plt.figure(figsize=(6, 5))
        plt.scatter(self.y_test, rf_pred, color='y')
        plt.plot([0, 50000], [0, 50000], '--k')
        plt.axis('tight')
        plt.title("random forest and predicted value comparison")
        plt.xlabel("y_test")
        plt.ylabel("pred")
        plt.tight_layout()
        plt.show()
        print("Random Forest RMSE ", md_errors[i - 1])
        rf_score = r2_score(self.y_test, rf_pred)
        print("Random Forest r2 score ", rf_score)
        rf_ms = mean_squared_error(self.y_test, rf_pred)
        print("Random Forest mean squared error ", rf_ms)
        rf_m = mean_absolute_error(self.y_test, rf_pred)
        print("Random Forest  mean absolute error ", rf_m)

    def run_all_variables_technique(self):
        self.train_test_split(self.x, self.y)
        self.linear_regression()
        self.polynomial_linear_regression(self.x_train)
        self.decision_tree()
        self.random_forest()

    def run_backward_elimination_technique(self):
        backward_df = self.backward_elimination(self.x_opt, self.y, self.sl)
        self.train_test_split(backward_df, self.y)
        self.linear_regression()
        self.polynomial_linear_regression(self.x_train)
        self.decision_tree()
        self.random_forest()

    def run_forward_elimination_technique(self):
        forward_df = self.forward_selection(self.X_df, self.y,self.sl)
        self.train_test_split(forward_df, self.y)
        self.linear_regression()
        self.polynomial_linear_regression(self.x_train)
        self.decision_tree()
        self.random_forest()

    def run(self):
        print("run")
        self.visualization()
        print("-------ALL VARIABLE TECHNIQUE--------")
        self.run_all_variables_technique()
        print("---------BACKWARD ELIMINATION---------")
        self.run_backward_elimination_technique()
        print("---------FORWARD ELIMINATION----------")
        self.run_forward_elimination_technique()

Regression().run()