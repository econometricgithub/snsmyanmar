#importing necessary library
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template,flash, request
from wtforms import Form, StringField, validators, SubmitField, SelectField
from bioinfokit.analys import get_data, stat
#Importing SQLAlcheny
from flask_sqlalchemy import SQLAlchemy
#from custom_validators import height_validator, weight_validator
from myModules import results_summary_to_dataframe, result_one,results_two, main_fun, reg_metric, linerity,normality,homoscedasticity,multicollinearity
#input are just variables - not as in the short form [ Y, X ]
def model_1(OFEP="",SIU=""):
    df=pd.read_csv("Dataset/new_data.csv")
    plt.scatter(df.OFEP, df.SIU)
    plt.title('{0}  vs. {1}'.format(OFEP,SIU))
    plt.xlabel("{0}".format(SIU))
    plt.ylabel('{0}'.format(OFEP))
    plt.savefig("static/mode1_1.png")
    #define response variable
    y = df[OFEP]
    #define explanatory variable
    x = df[[SIU]]
    #add constant to predictor variables
    x = sm.add_constant(x)
    model="Simple Linear Regression"
    #fit linear regression model
    results = sm.OLS(y, x).fit()
    r1=result_one(model,y,results)
    r2=results_two(results)
    r3=results_summary_to_dataframe(results)
    r4=reg_metric(y,x,results)
    #define figure size
    fig = plt.figure(figsize=(12,8))
    #produce residual plots
    fig = sm.graphics.plot_regress_exog(results, SIU, fig=fig)
    fig.savefig("static/mode1_2.png")
    #define residuals
    res = results.resid
    #create Q-Q plot
    fig = sm.qqplot(res, fit=True, line="45")
    plt.savefig("static/mode1_3.png")
    return r1,r2,r3,r4
#a,b,c,d=model_1(OEP="OEP",SIU="SIU")
e,f,g,h=model_1(OFEP="OFEP",SIU="political_efficacy")


def strager_2():
    import pandas as pd
    from sklearn import datasets
    import statsmodels.api as sm
    from stargazer.stargazer import Stargazer
    from IPython.core.display import HTML

    diabetes = datasets.load_diabetes()
    df = pd.DataFrame(diabetes.data)
    print(df)
    df.columns = ['Age', 'Sex', 'BMI', 'ABP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    df['target'] = diabetes.target

    est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
    est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()

    stargazer = Stargazer([est, est2])
    stargazer=stargazer.render_html()
    return stargazer

def stargazer(dv,idv=[]):
    import pandas as pd
    from sklearn import datasets
    import statsmodels.api as sm
    from stargazer.stargazer import Stargazer
    from IPython.core.display import HTML
    df = pd.read_csv("Dataset/new_data.csv")
    X = df[idv]
    y = df[dv]
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    stargazer = Stargazer([mlr_results])
    with open('templates/summary.html', 'w') as f:
        f.write(stargazer.render_html())