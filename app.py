#importing necessary library
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template,flash, request
from wtforms import Form, StringField, validators, SubmitField, SelectField
from bioinfokit.analys import get_data, stat
#Importing SQLAlcheny
from flask_sqlalchemy import SQLAlchemy
#from custom_validators import height_validator, weight_validator
from myModules import results_summary_to_dataframe, result_one,results_two, main_fun, reg_metric, label_function, gender_pie,location,values,values_two
from task import model_1, stargazer
from imageModules import create_figure, coeff_plot,partial_plot,rsquared_plot
import csv
app = Flask(__name__)
@app.route("/")
def login():

    return  render_template("login.html")
database={'kyawthet':'6925057','phyominkyi':'ct43','ctdep':'ct18'}
@app.route('/home', methods=['POST','GET'])
def home():
    name=request.form['username']
    pwd=request.form['password']
    if name not in database:
        return render_template('login.html', info="Invalid User")
    else:
        if database[name]!=pwd:
            return render_template('login.html', info="Invalid Password")
        else:
            return render_template('main.html')
@app.route('/background')
def background():
    return render_template("background.html")

#this function is for Stat DataAnalysis MLR ***
@app.route('/sda', methods = ['GET', 'POST'])
def sda():
    #upload the CSV to Data Frame
    df = pd.read_csv("Dataset/new_data.csv")
    idv=[] #Indenpendent Variable - Format array
    dv=" " #single string
    model="" #chosen model
    if request.method=='POST':
        idv=request.form.getlist("idv")
        dv=request.form.get("dv")
        #Extracting Coefficients values, std err values and p-values
        if 'informational_use' and 'entertainment_use' and 'SIU' in idv:
            c_1, c_2, c_3, c_4, s_1, s_2, s_3, s_4, p_1, p_2, p_3, p_4 = values_two(dv, idv)
            stargazer(dv,idv)
            model=request.form.get("model")
            X=df[idv]
            y=df[dv]
            idv.append(dv)
            df_data=df[idv]
            result_one, result_two, result_three, reg_metric_output= main_fun(model,y, X)
            # scatter plot for all variables.
            sns.set_palette('colorblind')
            scatter_plot=sns.pairplot(df_data,height=1.5)
            plt.savefig('static/pairplot.png')

            #Coefficient Plot for regression summary
            coeff_plot(y,X)
            #Partial Plot
            partial_plot(y,X)
            return render_template('output.html',table= result_one, tables1=result_two,  tables2=[result_three.to_html(classes='output2')], tables3=[reg_metric_output.to_html(classes='output3')], dv=dv,idv=idv,coef_1=c_1,coef_2=c_2,coef_3=c_3,coef_4=c_4,std_1=s_1,std_2=s_2,std_3=s_3,std_4=s_4,pvalue_1=p_1,pvalue_2=p_2,pvalue_3=p_3,pvalue_4=p_4)
        elif 'three_usage' in idv:
            c_1, c_2, s_1, s_2, p_1, p_2 = values(dv, idv)
            stargazer(dv, idv)
            model = request.form.get("model")
            X = df[idv]
            y = df[dv]
            idv.append(dv)
            df_data = df[idv]
            result_one, result_two, result_three, reg_metric_output = main_fun(model, y, X)
            # scatter plot for all variables.
            sns.set_palette('colorblind')
            scatter_plot = sns.pairplot(df_data, height=1.5)
            plt.savefig('static/pairplot.png')

            # Coefficient Plot for regression summary
            coeff_plot(y, X)
            # Partial Plot
            partial_plot(y, X)
            return render_template('output_2.html', table=result_one, tables1=result_two, tables2=[result_three.to_html(classes='output2')], tables3=[reg_metric_output.to_html(classes='output3')], dv=dv,idv=idv,coef_1=c_1,coef_2=c_2,std_1=s_1,std_2=s_2,pvalue_1=p_1,pvalue_2=p_2)
        else:
            c_1, c_2, s_1, s_2, p_1, p_2 = values(dv, idv)
            stargazer(dv, idv)
            model = request.form.get("model")
            X = df[idv]
            y = df[dv]
            idv.append(dv)
            df_data = df[idv]
            result_one, result_two, result_three, reg_metric_output = main_fun(model, y, X)
            # scatter plot for all variables.
            sns.set_palette('colorblind')
            scatter_plot = sns.pairplot(df_data, height=1.5)
            plt.savefig('static/pairplot.png')
            # Coefficient Plot for regression summary
            coeff_plot(y, X)
            # Partial Plot
            partial_plot(y, X)
            return render_template('output_3.html', table=result_one, tables1=result_two,tables2=[result_three.to_html(classes='output2')],tables3=[reg_metric_output.to_html(classes='output3')], dv=dv,idv=idv,coef_1=c_1,coef_2=c_2,std_1=s_1,std_2=s_2,pvalue_1=p_1,pvalue_2=p_2)
    return render_template("sda.html")
@app.route('/slr', methods = ['GET', 'POST'])
def slr():
    df = pd.read_csv("Dataset/new_data.csv")
    idv =" "
    dv = " "
    if request.method == 'POST':
        idv = request.form.get("IDV")
        dv = request.form.get("DV")
        a,b,c,d=model_1(OFEP=dv,SIU=idv)
        return render_template("slroutput.html",tables=a, tables1=b,tables2=[c.to_html(classes='output2')],tables3=[d.to_html(classes='output3')], independent=idv)
    return render_template("slr.html")
@app.route('/descriptive', methods = ['GET', 'POST'])
def descriptive():
    df = pd.read_csv("Dataset/new_data.csv")
    #Dependent Variables Data Summary
    dv = ['OEP', 'OFEP', 'political_efficacy']
    df_dv = df[dv]
    result = df_dv.describe()
    # Three Usage Data Summary
    threeUsage = ['informational_use', 'entertainment_use', 'SIU']
    dv_2 = df[threeUsage]
    result2=dv_2.describe()
    gender_pie(df)
    return render_template("descriptive.html", table1=[result.to_html(classes='output3')], table2=[result2.to_html(classes='output3')])

if __name__ == '__main__':
    app.run(debug=True)

