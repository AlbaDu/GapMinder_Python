### Libraries
# data libraries
import pandas as pd
import numpy as np

# machine learning libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

def main_sequence(merged, target, columns_for_train, column_for_test, year):
    def general_info (df):
        """
                            ---What it does--
        This function checks the info, columns and shape of the df, printing them. Also it checks the presence of NaNs values on the df and prints them in case it founds them.

                            ---What it needs---
        A df object
        """

        # df columns info
        print('-dtype, length and name of columns-')
        print(df.info())
        print()
        print(df.columns)
        print()
        print(df.shape)
        print()

        # Presence of NaNs in df
        need_to_print =  False
        nulls = df.isnull().any()
        print('-Presence of NaNs in df-')
        print (nulls)
        for e in list(nulls):
            if e == True:
                need_to_print = True
        if need_to_print == True:
            print()
            print('-Number of NaNs in df-')
            print (df.isnull().sum() )

    def year_selector (df, year):
        """
        ---What it does---
        Filters by year inputed by the user.

        ---What it needs---
            - A df object (df)
                * MUST contain either columns with the year searched for or a column named 'year'
        ---What it returns---
        A df object (df_2)
        """

        df_2 = df.copy()

        if year in list(df_2.columns):
            df_2 = df_2[['country', year]]
            df_2.columns = ['country', 'target']
            print(df_2)
            return df_2

        elif 'year' in list(df_2.columns):
            if df_2['year'].dtype != str:
                df_2['year'] = df_2['year'].astype(str)

            df_2 = df_2.loc[df_2['year'] == year]
            print(df_2)
            return df_2

    def creating_linear_model(X_train, y_train):
        """
        ---What it does---
        Creates a linear model for Machine Learning.
        ---What it needs---
            - The X and y_train sets
        ---What it returns---
        The ML model and the accuracy score
        """

        linear = LinearRegression(fit_intercept=False).fit(X, y)
        linear.fit(X_train, y_train)
        print(linear)
        print()

        linear_model = linear.fit(X_train, y_train)

        y_pred = np.round(linear_model.predict(X_test))
        m = confusion_matrix(y_test, y_pred)
        print('- Confusion Matrix')
        print(m)

        acc = accuracy_score(y_pred, y_test)
        print(f'- Accuracy: {acc}')
        # print(print(classification_report(y_test, y_pred)))
        print()

        return linear_model, acc

    def creating_logistic_regression (X_train, y_train):
        """
        ---What it does---
        Creates a logistic regression model for Machine Learning.
        ---What it needs---
            - The X and y_train sets
        ---What it returns---
        The ML model and the accuracy score
        """

        logistic = LogisticRegression(random_state=0).fit(X, y)
        logistic.fit(X_train, y_train)
        print(logistic)
        print()

        logistic_model = logistic.fit(X_train, y_train)
        y_pred = np.round(logistic_model.predict(X_test))

        m = confusion_matrix(y_test, y_pred)
        print('- Confusion Matrix')
        print(m)

        acc = accuracy_score(y_pred, y_test)
        print(f'- Accuracy: {acc}')
        # print(classification_report(y_test, y_pred))
        print()

        return logistic_model, acc

    def creating_random_forest (X_train, y_train):
        """
        ---What it does---
        Creates a random forest model for Machine Learning.
        ---What it needs---
            - The X and y_train sets
        ---What it returns---
        The ML model and the accuracy score
        """
        
        forest = RandomForestClassifier(max_depth=5, random_state=17)
        forest.fit(X_train, y_train)
        print(forest)
        print()

        forest_model = forest.fit(X_train, y_train)
        y_pred = np.round(forest_model.predict(X_test))

        m = confusion_matrix(y_test, y_pred)
        print('- Confusion Matrix')
        print(m)

        acc = accuracy_score(y_pred, y_test)
        print(f'- Accuracy: {acc}')
        # print(classification_report(y_test, y_pred))
        print()

        return forest_model, acc

    def creating_decission_trees (X_train, y_train):
        """
        ---What it does---
        Creates a decission trees model for Machine Learning.
        ---What it needs---
            - The X and y_train sets
        ---What it returns---
        The ML model and the accuracy score
        """
        
        dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=17)
        dt_classifier.fit(X_train, y_train)
        print(dt_classifier)
        print()

        dtc_model = dt_classifier.fit(X_train, y_train)
        y_pred = np.round(dtc_model.predict(X_test))

        m = confusion_matrix(y_test, y_pred)
        print('- Confusion Matrix')
        print(m)

        acc = accuracy_score(y_pred, y_test)
        print(f'- Accuracy: {acc}')
        # print(classification_report(y_test, y_pred))
        print()

        return dtc_model, acc

    def creating_gradient_boost_classifier (X_train, y_train):
        """
        ---What it does---
        Creates a gradient boost classifier model for Machine Learning.
        ---What it needs---
            - The X and y_train sets
        ---What it returns---
        The ML model and the accuracy score
        """
        
        gbc_model = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
        gbc_model.fit(X_train, y_train)
        print(gbc_model)
        print()

        gradientboost_model = gbc_model.fit(X_train, y_train)
        y_pred = np.round(gradientboost_model.predict(X_test))

        m = confusion_matrix(y_test, y_pred)
        print('- Confusion Matrix')
        print(m)

        acc = accuracy_score(y_pred, y_test)
        print(f'- Accuracy: {acc}')
        # print(classification_report(y_test, y_pred))
        print()

        return gradientboost_model, acc

    def creating_xlgboost (X_train, y_train):
        """
        ---What it does---
        Creates a xlgboost model for Machine Learning.
        ---What it needs---
            - The X and y_train sets
        ---What it returns---
        The ML model and the accuracy score
        """

        xgb_model = XGBClassifier(learning_rate=0.5)
        xgb_model.fit(X_train, y_train)
        print(xgb_model)
        print()

        xlg_boost = xgb_model.fit(X_train, y_train)
        y_pred = np.round(xlg_boost.predict(X_test))

        m = confusion_matrix(y_test, y_pred)
        print('- Confusion Matrix')
        print(m)

        acc = accuracy_score(y_pred, y_test)
        print(f'- Accuracy: {acc}')
        # print(classification_report(y_test, y_pred))
        print()

        return xlg_boost, acc


    # Quick recon of the dfs
    general_info(merged)
    print()
    general_info(target)

    # NaN filling
    target = target.fillna(0)
    
    # Selection of year and merging
    train = year_selector(merged, year)
    test = year_selector(target, year)

    whole = train.merge(test, on='country')

    # Creating the splits for Machine Learning
    X = whole[columns_for_train]
    y = whole[column_for_test]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Model creations
    linear_model, linear_acc = creating_linear_model(X_train, y_train)
    logistic_model, logistic_acc = creating_logistic_regression (X_train, y_train)
    forest_model, forest_acc = creating_random_forest (X_train, y_train)
    decission_trees, dt_acc = creating_decission_trees (X_train, y_train)
    gradient_boost_model, gboost_acc = creating_gradient_boost_classifier (X_train, y_train)
    xlg_boost, xlgb_acc = creating_xlgboost (X_train, y_train)




    