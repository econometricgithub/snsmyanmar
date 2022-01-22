import pandas as pd
def create_figure():
    import io
    import random
    from flask import Response
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    df = pd.read_csv("Dataset/new_data.csv")
    b = df['OEP']
    d = df['political_efficacy']
    B, D = np.meshgrid(b, d)
    nu = np.sqrt(1 + (2 * D * B) ** 2) / np.sqrt((1 - B ** 2) ** 2 + (2 * D * B) ** 2)
    fig= plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(B, D, nu)
    plt.xlabel('b')
    plt.ylabel('d')
    #plt.savefig("static/exam.png")
    #plt.close(fig)
    #return fig
    plt.show()
def coeff_plot(dv, idv=[]):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    X = sm.add_constant(idv)
    y = dv
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    coeff = round(mlr_results.params, 4)
    confid = mlr_results.conf_int()
    err_series = coeff - confid[0]
    coef_df = pd.DataFrame({'coef': coeff.values[1:],
                            'err': err_series.values[1:],
                            'varname': err_series.index.values[1:]})
    #fig_size=len(idv)
    #print(fig_size)
    fig, ax = plt.subplots(figsize=(18, 8))
    coef_df.plot(x='varname', y='coef', kind='bar',
                 ax=ax, color='none',
                 yerr='err', legend=False)
    ax.set_ylabel('Dependent Variable',fontsize=20)
    ax.set_xlabel('Independent Variables',fontsize=20)
    ax.scatter(x=pd.np.arange(coef_df.shape[0]),
               marker='s', s=120,
               y=coef_df['coef'], color='black')
    ax.axhline(y=0, linestyle='--', color='blue', linewidth=2)
    ax.xaxis.set_ticks_position('none')
    _ = ax.set_xticklabels(idv, rotation='vertical', fontsize=20)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.01)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('static/coefficient_plt.png',dpi=45)

def partial_plot(dv, idv=[]):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    X = sm.add_constant(idv)
    y = dv
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    fig = sm.graphics.plot_partregress_grid(mlr_results)
    fig.tight_layout(pad=1.0)
    figsize = (18, 8)
    plt.savefig('static/partial_plot.png',dpi=90)


def rsquared_plot(dv,idv=[]):
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    df = pd.read_csv("Dataset/new_data.csv", header=0, sep=",")
    x = df[[idv]]
    x = x.iloc[:, 0]
    y = df[[dv]]
    y = y.iloc[:, 0]
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    r_sqauared = round(r ** 2, 2)

    def myfunc(x):
        return slope * x + intercept
    mymodel = list(map(myfunc, x))
    plt.scatter(x, y)
    plt.plot(x, mymodel)
    if dv=='OEP' and idv=='informational_use':
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=6)
    if dv=='OEP' and idv=='entertainment_use':
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=6)
    if dv == 'OFEP' and idv == 'entertainment_use':
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=6)
    if dv=='OFEP' and idv=='informational_use':
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=6)
    if dv=='OFEP' and idv=='political_efficacy':
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=6)
    if dv=='OEP' and idv=='political_efficacy':
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=6)
    if dv=='political_efficacy' and idv=='informational_use':
        plt.ylim(ymin=0, ymax=6)
        plt.xlim(xmin=0, xmax=6)
    if dv=='political_efficacy' and idv=='entertainment_use':
        plt.ylim(ymin=0, ymax=6)
        plt.xlim(xmin=0, xmax=6)
    else:
        plt.ylim(ymin=0, ymax=1)
        plt.xlim(xmin=0, xmax=1)
    plt.xlabel("{}".format(idv))
    plt.ylabel("{}".format(dv))
    plt.title('R2: ' + str(r_sqauared))
    plt.savefig('static/r_squared.png', dpi=72)





