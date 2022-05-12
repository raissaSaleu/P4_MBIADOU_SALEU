############### PROJET 4 #################

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from scipy import stats
import warnings

def split_words(df, column = None):
  list_words = set()
  for word in df[column].str.split(','):
    if isinstance(word, float):
      continue
    list_words = set().union(word, list_words)
  return list(list_words)

#------------------------------------------

def poucentageValeursManquantes(data, column):
    ''' 
        Calcule le pourcentage moyen de valeurs manquantes
        dans une dataframe donnée par valeur unique d'une colonne donnée
        
        Parameters
        ----------------
        data   : dataframe contenant les données                 
        column : str
                 La colonne à analyser
        
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "column"
            - une colonne "Percent Missing" : contenant le pourcentage de valeur manquante pour chaque valeur de colonne    
    '''
    
    percent_missing = data.isnull().sum(axis=1) * 100 / len(data.columns)
    
    return pd.DataFrame({column: data[column], 'Percent Missing': percent_missing})\
                        .groupby([column])\
                        .agg('mean')
#------------------------------------------


def tauxRemplissage(data):
    ''' 
        Cette fonction prépare les données qui seront affichées pour montrer le taux de valeurs 
        renseignées/non renseignées des colonnes dans un dataframe "data" 
        
        Parameters
        ----------------
        - data : dataframe contenant les données
 
        Returns
        ---------------
        Un dataframe contenant:
            - une colonne "Percent Missing" : pourcentage de données non renseignées
            - une colonne "Percent Filled" : pourcentage de données renseignées
            - une colonne "Total": 100
            
    '''    

    
    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})

    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']

    missing_percent_df['Total'] = 100

    percent_missing = data.isnull().sum() * 100 / len(data.columns)
    
    return missing_percent_df

#------------------------------------------

def plotTauxRemplissage(data, long, larg):
    ''' 
        Trace les proportions de valeurs remplies/manquantes pour chaque colonne
        dans la colonne de data sous forme de graphique à barres horizontales empilées.
        
        Parameters
        ----------------
        data   : un dataframe avec : 
                   - une colonne "Percent Missing" : pourcentage de données non renseignées
                   - une colonne "Percent Filled" : pourcentage de données renseignées
                   - une colonne "Total": 100
                                 
        long   : int 
                 La longueur de la figure
        
        larg   : int
                 La largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''
    
    data_to_plot = tauxRemplissage(data).sort_values("Percent Filled").reset_index()

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    b = sns.barplot(x="Total", y="index", data=data_to_plot,label="non renseignées", color="thistle", alpha=0.3)
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    b.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    c = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,label="renseignées", color="darkviolet")
    c.set_xticklabels(c.get_xticks(), size = TICK_SIZE)
    c.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)

    ax.set(ylabel="Colonnes",xlabel="Pourcentage de valeurs (%)")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x)) + '%'))
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    plt.savefig('missingPercentagePerColumn.png')

    # Display the figure
    plt.show()
    
#------------------------------------------


def plotBarplot(data, col_x, col_y, long, larg, title):
    '''
        Plots a horizontal bar plot
        
        Parameters
        ----------------
        data    : pandas dataframe
                  Working data containing col_x et col_y
        
        col_x   : string
                  Name of the features to use for x
                  
        col_y   : string
                  Name of the features to use for y
                 
        long    : int
                  The length of the figure for the plot
        
        larg    : int
                  The width of the figure for the plot
                                  
        title   : string
                  Title for the plot
                  
        Returns
        ---------------
        -
    '''
    
    LABEL_SIZE = 20
    LABEL_PAD = 15

    f, ax = plt.subplots(figsize=(larg, long))

    plt.title(title,
                  fontweight="bold",
                  fontsize=30, pad=10)

    b = sns.barplot(x=col_x, y=col_y,
                    data=data,
                    label="non renseignées",
                    color="darkviolet")

    b.set_xticklabels(b.get_xticks(), size = 20)
    _, ylabels = plt.yticks()
    _, xlabels = plt.xticks()
    _=b.set_yticklabels(ylabels, size=20)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
            
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

#------------------------------------------

def plot3Regplots(data, col_x1, col_x2, col_x3, col_y, long, larg, title):
    '''
        Plots 3 regplots horizontally in a single figure
        
        Parameters
        ----------------
        data   : pandas dataframe
                 Working data
                 
        col_x1 : string
                 Name of the feature to use for x in the 1st regplot
         
        col_x2 : string
                 Name of the feature to use for x in the 2nd regplot
                 
        col_x3 : string
                Name of the feature to use for x in the 3rd regplot
        
        col_y : string
                Name of the feature to use for y in the regplots
        
        long   : int
                 The length of the figure for the plot
        
        larg   : int
                 The width of the figure for the plot
                
        title   : string
                  The title for the plot
        
        Returns
        ---------------
        -
    '''
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=False, figsize=(larg, long))

    plt.suptitle(title, fontweight="bold", fontsize=25)

    sns.regplot(x=data[col_x1], y=data[col_y], ax=ax1)
    sns.regplot(x=data[col_x2], y=data[col_y], ax=ax2)
    sns.regplot(x=data[col_x3], y=data[col_y], ax=ax3)

#------------------------------------------




def plotBoxPlots(data, long, larg, nb_rows, nb_cols):
    '''
        Affiche un boxplot pour chaque colonne dans data.
        
        Parameters
        ----------------
        data : dataframe contenant des exclusivment des variables quantitatives
                                 
        long : int
               longueur de la figure
        
        larg : int
               largeur de la figure
               
        nb_rows : int
                  Le nombre de lignes dans le subplot
        
        nb_cols : int
                  Le nombre de colonnes dans le subplot
                                  
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 35
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("BOXPLOT DES VARIABLES QUANTITATIVES", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_quant in data.columns.tolist():
        ax = axes[row, column]

        sns.despine(left=True)

        #b = sns.boxplot(x=np.log10(data[ind_quant]), ax=ax, color="darkviolet")
        b = sns.boxplot(x=data[ind_quant], ax=ax, color="darkviolet")


        plt.setp(axes, yticks=[])

        plt.tight_layout()

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        if ind_quant == "salt_100g":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        ly = ax.get_ylabel()
        ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.tick_params(axis='both', which='major', pad=TICK_PAD)

        ax.xaxis.grid(True)
        ax.set(ylabel="")

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
                    
#------------------------------------------



def plotPieChart(data, long, larg, title):
    '''
        Plots a pie chart of the proportion of each modality for groupby_col
        with the dimension (long, larg), with the given title and saved figure
        title.
        
        Parameters
        ----------------
        data           : pandas dataframe
                         Working data, with a "groupby_col" column
        
        groupby_col    : string
                         The name of the quantitative column of which the modality
                         frequency should be plotted.
                                  
        long           : int
                         The length of the figure for the plot
        
        larg           : int
                         The width of the figure for the plot
        
        title          : string
                         title for the plot
        
        title_fig_save : string
                         title under which to save the figure
                 
        Returns
        ---------------
        -
    '''
    
    TITLE_SIZE = 25
    TITLE_PAD = 60

    # Initialize the figure
    f, ax = plt.subplots(figsize=(long, larg))


    # Set figure title
    # Set figure title
    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)
       
    # Put everything in bold
    #plt.rcParams["font.weight"] = "bold"

    # Create pie chart for topics
    a = data.plot(kind='pie', autopct=lambda x:'{:2d}'.format(int(x)) + '%', fontsize =20)
    # Remove y axis label
    ax.set_ylabel('')
    
    # Make pie chart round, not elliptic
    plt.axis('equal')
    
    # Display the figure
    plt.show()

#------------------------------------------

def plotEnergySourcesDistribution(data, criterion, long, larg):
    '''
        Trace les proportions d'utilisation par source d'énergie en % pour
        chaque valeur prise par criterion sous forme de barre horizontale empilée
        
        Parameters
        ----------------
        data   : pandas dataframe avec:
                    - un indice nommé criterion contenant les différentes valeurs des critères
                    - une colonne par source d'énergie contenant
                 
                                 
        long   : int
                 longueur de la figure
        
        larg   : int
                 largeur de la figure
                                
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 30
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 30

    # Reset index to access the Seuil as a column
    data_to_plot = data.reset_index()

    sns.set(style="whitegrid")
    palette = sns.husl_palette(len(data.columns))

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(long, larg))

    plt.title("RÉPARTITION DES SOURCES D'ÉNERGIE",
              fontweight="bold",fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Get the list of topics from the columns of data
    column_list = list(data.columns)

    # Create a barplot with a distinct color for each topic
    for idx, column in enumerate(reversed(column_list)):
        color = palette[idx]
        b = sns.barplot(x=column, y=criterion, data=data_to_plot, label=str(column), orient="h", color=color)
        
        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        _, ylabels = plt.yticks()
        b.set_yticklabels(ylabels, size=TICK_SIZE)

        
    # Add a legend and informative axis label

    ax.legend(bbox_to_anchor=(0,-0.6,1,0.2), loc="lower left", mode="expand",
              borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    ax.set(ylabel="Type du bâtiment principal",xlabel="% des sources d'énergie")

    lx = ax.get_xlabel()
    ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ly = ax.get_ylabel()
    ax.set_ylabel(ly, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:2d}'.format(int(x))))

    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------



def plotQualitativeDist(data, long, larg):
    '''
        Affiche un graphique à barres indiquant la fréquence
        des modalités pour chaque colonne de données.
        
        Parameters
        ----------------
        data : dataframe contenant des données qualitatives
                                 
        long : int
               longueur de l figure
        
        larg : int
               largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 80
    TITLE_PAD = 1.05
    TICK_SIZE = 40
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    nb_rows = 3
    nb_cols = 2

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VARIABLES QUALITATIVES", fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    row = 0
    column = 0

    for ind_qual in data.columns.tolist():
        
        data_to_plot = data.sort_values(by=ind_qual).copy()
        
        ax = axes[row, column]
        
        b = sns.countplot(y=ind_qual,
                          data=data_to_plot,
                          color="darkviolet",
                          ax=ax,
                          order = data_to_plot[ind_qual].value_counts().index)


        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=1.4, hspace=0.2)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        
        ylabels = [item.get_text().upper() for item in ax.get_yticklabels()]
        b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")

        lx = ax.get_xlabel()
        ax.set_xlabel(lx, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
        ly = ax.get_ylabel()
        ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))
        
        ax.xaxis.grid(True)

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
    
#------------------------------------------

def search_componant(df, suffix=None):
  componant = []
  for col in df.columns:
      if suffix in col: 
        componant.append(col)
  return componant
#------------------------------------------



def calculeLorenzGini(data):
    '''
        Calcule la courbe de Lorenz et le coefficient de Gini pour une variable donnée
        
        ----------------
        - data       : data series
        
        Returns
        ---------------
        Un tuple contenant :
        - lorenz_df : une liste contenant les valeurs de la courbe de Lorenz
        - gini_coeff : les coefficients de Gini associés
        
        Source : www.openclassrooms.com
    '''
    
    dep = data.dropna().values
    n = len(dep)
    lorenz = np.cumsum(np.sort(dep)) / dep.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

    #---------------------------------------------------
    # Gini :
    # Surface sous la courbe de Lorenz. Le 1er segment
    # (lorenz[0]) est à moitié en dessous de 0, on le
    # coupe donc en 2, on fait de même pour le dernier
    # segment lorenz[-1] qui est à 1/2 au dessus de 1.
    #---------------------------------------------------

    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    # surface entre la première bissectrice et le courbe de Lorenz
    S = 0.5 - AUC
    gini_coeff = [2*S]
         
    return (lorenz, gini_coeff)
    
#------------------------------------------

def calculeLorenzsGinis(data):
    '''
        Calcule la courbe de Lorenz et les coefficients de Gini
        pour toutes les colonnes d'une dataframe
        
        ----------------
        - data       : dataframe
        
        Returns
        ---------------
        Un tuple contenant :
        - lorenz_df : une dataframne contenant les valeurs de la courbe de Lorenz
                      pour chaque colonne de la dataframe
        - gini_coeff : une dataframe contenant le coeff de Gini associé pour
                       chaque colonne de la dataframe
    '''
    
    ginis_df = pd.DataFrame()
    lorenzs_df = pd.DataFrame()

    for ind_quant in data.columns.unique().tolist():
        lorenz, gini = calculeLorenzGini(data[ind_quant])
        ginis_df[ind_quant] = gini
        lorenzs_df[ind_quant] = lorenz

    n = len(lorenzs_df)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    lorenzs_df["index"]=xaxis[:-1]
    lorenzs_df.set_index("index", inplace=True)
    
    ginis_df = ginis_df.T.rename(columns={0:'Indice Gini'})
    
    return (lorenzs_df, ginis_df)

#------------------------------------------

def plotLorenz(lorenz_df, long, larg):
    '''
        Dessin la courbe de Lorenz
        
        ----------------
        - lorenz_df : une dataframe contenant les valeurs de Lorenz
                      une colonne = valeur Lorenz pour une variable
        - long       : int
                       longueur de  la figure
        
        - larg       : int
                       largeur de la figure
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50


    sns.set(style="whitegrid")
    
    f, ax = plt.subplots(figsize=(long, larg))
    
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.title("VARIABLES QUANTITATIVES - COURBES DE LORENZ",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    sns.set_color_codes("pastel")
    
    b = sns.lineplot(data=lorenz_df, palette="pastel", linewidth=5, dashes=False)
    
    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

    b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)

    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
    
    ax.tick_params(axis='both', which='major', pad=TICK_PAD)

    ax.set_xlabel("")

    # Add a legend and informative axis label
    leg = ax.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, ncol=1, frameon=True,
             fontsize=LEGEND_SIZE)
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)


    # Display the figure
    plt.show()

#------------------------------------------

def plotDistplotsRug(data, long, larg, nb_rows, nb_cols):
    '''
        Trace la distribution de toutes les colonnes dans data
        (doit être des colonnes quantitatives uniquement)
        couplée à un graphique en tapis de la distribution
        
        Parameters
        ----------------
        data : dataframe contenant exclusivement des données quatitatives
                                 
        long : int
               longueur de la figure
        
        larg : int
               largeur de la figure
               
        nb_rows : int
                  nombre de lignes du subplot
        
        nb_cols : int
                  nombre de colonnes du subplot
                                 
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(long, larg))

    f.suptitle("DISTRIBUTION DES VARIABLES QUANTITATIVES", fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for ind_quant in data.columns.tolist():

        sns.despine(left=True)

        ax = axes[row, column]

        b = sns.distplot(data[ind_quant], ax=ax, rug=True)

        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
        b.set_xlabel(ind_quant,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

        if ind_quant in ["saturated-fat_100g", "salt_100g"]:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(float(x))))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:d}'.format(int(x))))

        plt.setp(axes, yticks=[])

        plt.tight_layout()

        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0

#------------------------------------------

def plotCorrelationHeatMap(data,long, larg, title):
    '''
        Plots a heatmap of the correlation coefficients
        between the quantitative columns in data
        
        Parameters
        ----------------
        - data : dataframe
                 Working data
                 
        - corr : string
                 the correlation method ("pearson" or "spearman")
        
        - long : int
                 The length of the figure for the plot
        
        - larg : int
                 The width of the figure for the plot
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 40
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 45
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    f, ax = plt.subplots(figsize=(long, larg))
                
    f.suptitle(title, fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.heatmap(data, mask=np.zeros_like(data, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                    annot=data, annot_kws={"fontsize":20}, fmt=".2f")

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold", rotation=90)
    b.set_xlabel(data.columns.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in ax.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold",rotation=45,va='top')
    b.set_ylabel(data.index.name,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()

#------------------------------------------

def plotTableContingenceKhi2(data, X, Y, long, larg):
    '''
        Affiche une table de contingence de var1 et var2 coloré en 
        fonction du test Khi2
        
        ----------------
        - data : une dataframe contenant les données
        - X,Y : 2 variables qualitatives
        - long : int
                 longueur de la figure
        
        - larg : int
                 largeur de la figure 
        
        Returns
        ---------------
        _
    '''
    
    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 18
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    cont = data[[X,Y]].pivot_table(index=Y,columns=X,aggfunc=len,margins=True,margins_name="Total").fillna(0).copy().astype(int)
    
    
    stat, p, dof, expected = chi2_contingency(cont)

    measure = (cont-expected)**2/expected
    table = measure/stat
    
    
    #tx = cont.loc[:,["Total"]]
    #ty = cont.loc[["Total"],:]
    #n = len(data)
    #indep = tx.dot(ty) / n

    #c = cont.fillna(0) # On remplace les valeurs nulles par 0
    #measure = (c-indep)**2/indep
    #xi_n = measure.sum().sum()
    #table = measure/xi_n

    f, axes = plt.subplots(figsize=(long, larg))

    f.suptitle("TABLEAU DE CONTINGENCE\nAVEC MISE EN LUMIÈRE DES RELATIONS PROBABLES (KHI-2)", fontweight="bold",
                   fontsize=TITLE_SIZE, y=TITLE_PAD)

    b=sns.heatmap(table.iloc[:-1,:-1],annot=cont.iloc[:-1,:-1],annot_kws={"fontsize":20}, fmt="d")

    xlabels = [item.get_text().upper() for item in axes.get_xticklabels()]
    b.set_xticklabels(xlabels, size=TICK_SIZE, weight="bold")
    b.set_xlabel(table.iloc[:-1, :-1].columns.name.upper(),fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    ylabels = [item.get_text() for item in axes.get_yticklabels()]
    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    b.set_ylabel(table.iloc[:-1, :-1].index.name.upper(),fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
    
#------------------------------------------

def testKhi2(data, X, Y):
    '''
        Test Khi2
        
        ----------------
        - data : une dataframe contenant les données
        - X,Y : 2 variables qualitatives
        
        Returns
        ---------------
        - stat         : the result of the Khi-2 test
        - p            : the p-value
        - dof          : degrees of freedom
    '''
    cont = data[[X,Y]].pivot_table(index=Y,columns=X,aggfunc=len,margins=True,margins_name="Total").fillna(0).copy().astype(int)
    stat, p, dof, expected = chi2_contingency(cont)
    
    return (stat, p, dof)
    
#------------------------------------------

def plotQualQuantDist(data, X, Y, long, larg):
    '''
        Affiche un graphique à barres indiquant la fréquence
        des modalités pour chaque colonne de données.
        
        Parameters
        ----------------
        data : dataframe contenant des données qualitatives
        
        X    : variable qualitative
        
        Y    : variable quantitative
                                 
        long : int
               longueur de la figure
        
        larg : int
               largeur de la figure
                                  
        
        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 25
    TITLE_PAD = 1
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 25
    LABEL_PAD = 10
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    
    sous_echantillon = data.copy()

    modalites = sous_echantillon[X].sort_values(ascending=True).unique()
    groupes = []
    for m in modalites:
        groupes.append(sous_echantillon[sous_echantillon[X]==m][Y])
    
    # Propriétés graphiques (pas très importantes)    
    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}

    f, axes = plt.subplots(figsize=(long, larg))

    f.suptitle("BOXPLOT "+ X +"/"+Y, fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)


    b = plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianprops, 
                vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
    axes.set_xlabel(Y.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    axes.set_ylabel(X.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
    #if X == "nutriscore_grade" :
    #    ylabels = [item.get_text().upper() for item in axes.get_yticklabels()]
    #    b.set_yticklabels(ylabels, size=TICK_SIZE, weight="bold")
    plt.show()
    
#------------------------------------------

def eta_squared(data, x_qualit,y_quantit):
    '''
        Calculate the proportion of variance
        in the given quantitative variable for
        the given qualitative variable
        
        ----------------
        - data      : The dataframe containing the data
        - x_quantit : The name of the qualitative variable
        - y_quantit : The name of the quantitative variable
        
        Returns
        ---------------
        Eta_squared
    '''
    
    sous_echantillon = data.copy().dropna(how="any")

    x = sous_echantillon[x_qualit]
    y = sous_echantillon[y_quantit]

    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
#------------------------------------------

def plot2Distplots(col_x1, label_col_x1, col_x2, label_col_x2, long, larg, title):
    '''
        Plots 2 distplots horizontally in a single figure
        
        Parameters
        ----------------
        col_x1          : pandas series
                          The data to use for x in the 1st distplot
                   
        label_col_x1    : string
                          The label for x in the 1st distplot
        
        col_x2          : pandas series
                          The data to use for x in the 2nd distplot
        
        label_col_x2    : string
                          The label for x in the 2nd distplot
        
        long            : int
                          The length of the figure for the plot
        
        larg            : int
                          The width of the figure for the plot
                
        title           : string
                          The title for the plot
        
        Returns
        ---------------
        -
    '''
        
    TITLE_SIZE = 30
    TITLE_PAD = 1.05
    TICK_SIZE = 15
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    sns.set_palette(sns.dark_palette("purple", reverse="True"))

    f, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(long, larg))

    f.suptitle(title, fontweight="bold",
               fontsize=TITLE_SIZE, y=TITLE_PAD)

    sns.despine(left=True)

    b = sns.distplot(col_x1, ax=ax1)

    b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    b.set_xlabel(label_col_x1,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    c = sns.distplot(col_x2, ax=ax2)

    c.set_xticklabels(b.get_xticks(), size = TICK_SIZE)
    c.set_xlabel(label_col_x2,fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.setp((ax1, ax2), yticks=[])
    plt.setp((ax1, ax2), xticks=[])


    plt.tight_layout()
    
    # Display the figure
    plt.show()
    
#------------------------------------------------

def fitPredictPlot(algorithms, xtrain, xtrain_std, ytrain, xtest, xtest_std, ytest):
    '''
        For each given algorithms :
        - Trains it on (xtrain, ytrain)
        - Predicts the values for xtest
        - Calculate the RMSE and R2 score with ytest
        - Getst the calculation time
        
        It does the same for xtrain_std, ytrain, xtest_std, ytest.
        
        The function then plots for each algorithm and each type of input the predicted value
        as a function of the known value, and return the performance data for all models.
        
        Parameters
        ----------------
        algorithms  : dictionary with
                        - names and type of input as keys
                        - instantiated algorithms as values
        
        - xtrain    : pandas dataframe
                      x training data
        - xtrain_std: pandas dataframe
                      standardized x training data
        - ytrain    : pandas dataframe
                      y training data
        - ytrain_std: pandas dataframe
                      standardized y training data
        - xtest     : pandas dataframe
                      x test data
        - xtest_std : pandas dataframe
                      standardized x test data
        - ytest     : pandas dataframe
                      y test data
                
        Returns
        ---------------
        r2_rmse_time   : pandas dataframe containing the RMSE, R2 score and calculation time for
        each algorithm
-
    '''
    
    r2_rmse_mae_time = pd.DataFrame()
    
    # Set up for the plot
    TITLE_SIZE = 45
    SUBTITLE_SIZE = 25
    TITLE_PAD = 1.05
    TICK_SIZE = 25
    TICK_PAD = 20
    LABEL_SIZE = 30
    LABEL_PAD = 30
    LEGEND_SIZE = 30
    LINE_WIDTH = 3.5

    nb_rows = 5
    nb_cols = 2

    f, axes = plt.subplots(nb_rows, nb_cols, figsize=(20, 40))
    f.suptitle("Performances des modèles", fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for algoname, algo in algorithms.items():

        if "std" in algoname:
            X_train = xtrain_std
            X_test = xtest_std
        else:
            X_train = xtrain
            X_test = xtest
        
            algo.fit(X_train, ytrain.values.ravel())
            
            start_time = time.time()

            ypred = algo.predict(X_test)
            
        elapsed_time = time.time() - start_time

        r2_rmse_mae_time.loc[algoname, "RMSE"] = np.sqrt(mean_squared_error(ytest, ypred))
        r2_rmse_mae_time.loc[algoname, "MAE"] = mean_absolute_error(ytest, ypred)
        r2_rmse_mae_time.loc[algoname, "R2"] = r2_score(ytest, ypred)
        r2_rmse_mae_time.loc[algoname, "Time"] = elapsed_time

        # plot
        ax = axes[row, column]

        b = sns.regplot(x=ytest, y=ypred, ax=ax)

        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        ax.set_xlabel('Valeur mesurée', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        ax.set_ylabel('Valeur prédite', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        
        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)
        scores = (r'$R^2={:.2f}$' + '\n' + r'$RMSE={:.2f}$' + '\n' + r'$MAE={:.2f}$').format(r2_score(ytest, ypred), np.sqrt(mean_squared_error(ytest, ypred)), mean_absolute_error(ytest, ypred))
        
        
        ax.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        ax.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")
        
            
        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
        
        # Display the figure
    plt.show()
        
    return r2_rmse_mae_time
 
#------------------------------------------

def predictPlot(algorithm, xtest, ytest, title):
    '''
        - Predicts the values for xtest
        - Calculate the MAE and R2 score with yval
        - Getst the calculation time
        
               
        Parameters
        ----------------
        algorithm  :   dictionary with
                        - names and type of input as keys
        - xtest     : pandas dataframe
         
        - ytest     : pandas dataframe
                               
        Returns
        ---------------
        r2_rmse_mae_time   : pandas dataframe containing the RMSE, MAE, R2 score and calculation time for
        the algorithm
-
    '''
    
    r2_mae_time = pd.DataFrame()
    
    #ytest_log = np.log(ytest)
    
    # Set up for the plot
    TITLE_SIZE = 35
    SUBTITLE_SIZE = 25
    TITLE_PAD = 1.05
    TICK_SIZE = 18
    TICK_PAD = 14
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LEGEND_SIZE = 25
    LINE_WIDTH = 3.5

    nb_rows = 1
    nb_cols = 1

    f, ax = plt.subplots(nb_rows, nb_cols, figsize=(8,8))
    f.suptitle("Performances du modèle (jeu de test)\n Variable {}".format(title), fontweight="bold", fontsize=TITLE_SIZE, y=TITLE_PAD)

    row = 0
    column = 0

    for algoname, algo in algorithm.items():

        start_time = time.time()
        ypred = algo.predict(xtest)    
        elapsed_time = time.time() - start_time
        
        #transformation inverse une fois la prédiction effectuée pour obtenir la variable souhaitée avant de calculer les métriques.
        #ytest_exp = np.exp(ytest)
        #ypred_exp = np.exp(ypred)
        
        
        r2_mae_time.loc[algoname, "MAE"] = mean_absolute_error(ytest, ypred)
        r2_mae_time.loc[algoname, "R2"] = r2_score(ytest, ypred)
        r2_mae_time.loc[algoname, "Time"] = elapsed_time

        # plot
        #ax = axes[row, column]

        b = sns.regplot(x=ytest, y=ypred, ax=ax)

        plt.tight_layout()
        
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.3, hspace=0.4)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        ax.set_xlabel('Valeur réelle', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        ax.set_ylabel('Valeur prédite', fontsize=LABEL_SIZE, labelpad=LABEL_PAD)
        
        b.set_xticklabels(b.get_xticks(), size = TICK_SIZE)

        b.set_yticklabels(b.get_yticks(), size = TICK_SIZE)
        #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))
        #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

        extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                              edgecolor='none', linewidth=0)
#        scores = (r'$R^2={:.2f}$' + '\n' + r'$RMSE={:.2f}$' + '\n' + r'$MAE={:.2f}$').format(r2_score(yval, ypred), np.sqrt(mean_squared_error(yval, ypred)), mean_absolute_error(yval, ypred))
        
        scores = (r'$R^2={:.2f}$' + '\n' + r'$MAE={:.2f}$').format(r2_score(ytest, ypred), mean_absolute_error(ytest, ypred))

        
        ax.legend([extra], [scores], loc='upper left', fontsize=LEGEND_SIZE)
        title = algoname + '\n Évaluation en {:.2f} secondes'.format(elapsed_time)
        ax.set_title(title, fontsize=SUBTITLE_SIZE, fontweight="bold")
        
            
        if column < nb_cols-1:
            column += 1
        else:
            row += 1
            column = 0
        
    
    plt.xticks(rotation=45)
    
    #plt.ylim(0, ytest.max())
    #plt.xlim(0, ytest.max())
    #plt.xlim()
    # Display the figure
    plt.show()
        
    return r2_mae_time
 
#------------------------------------------




def getPerformances(data_dict, y_col, r2_adjusted=None, std=None):
    '''
        Gives the performance in terms of RMSE, R2, adjusted R2
        
        Parameters
        ----------------
        data_dict   : dict with :
                    - name of the input data as keys (string)
                    - input data as values (pandas dataframe)
                 
        y_col       : string
                      The name of the feature to predict
        
        r2_adjusted : bool
                      True if adjusted R2 should be adjusted
                
        Returns
        ---------------
        A tuple containing :
        - r2_rmse_time  : dataframne
                          The RMSE et R2 score values for each model
        - model_df      : dataframe
                          The associated Gini coeff for each column of
                          the given dataframe
    '''
    
    r2_rmse_time = pd.DataFrame(columns=["RMSE", "R2", "Time", "R2 Ajusté"])
    model_df = pd.DataFrame()

    for ajout_name, data_ajout in data_dict.items():
        
        xtrain, xtest, ytrain, ytest = train_test_split(
                                data_ajout.loc[:, data_ajout.columns != y_col],
                                data_ajout[[y_col]],
                                test_size=0.3)
                                
        if std == "standardized":
            # Standardisation des X
            std_scale = preprocessing.StandardScaler().fit(xtrain)
            X_train = std_scale.transform(xtrain)
            X_test = std_scale.transform(xtest)
        else:
            X_train = xtrain
            X_test = xtest
            
        ytrain_log = np.log(ytrain)
        ytest_log = np.log(ytest)

        parameters = [{'n_estimators': [100, 200, 500, 700, 1000]}]

        clf = GridSearchCV(RandomForestRegressor(), parameters)
    
        clf.fit(X_train, ytrain_log.values.ravel())

        model = RandomForestRegressor(n_estimators=clf.best_params_["n_estimators"],
                                   oob_score=True, random_state=4)
        
        start_time = time.time()

        model.fit(X_train, ytrain_log.values.ravel())
        ypred = model.predict(X_test)

        elapsed_time = time.time() - start_time

        model_df[ajout_name] = [model]
    
        r2_rmse_time.loc[ajout_name, "RMSE"] = np.sqrt(mean_squared_error(ytest_log, ypred))
        
        if r2_adjusted==True:
            r2_rmse_time.loc[ajout_name, "R2 Ajusté"] = 1 - (1-r2_score(ytest_log, ypred))*(len(ytest_log)-1)/(len(ytest_log)-X_test.shape[1]-1)
        else:
            r2_rmse_time.loc[ajout_name, "R2"] = r2_score(ytest_log, ypred)
        
        r2_rmse_time.loc[ajout_name, "Time"] = elapsed_time

    return r2_rmse_time, model_df

#------------------------------------------

def getRhoPValueHeatmaps(data, interest_cols, FEATURE, THRESHOLD):
    '''
        Calculates the rho of Spearman for a given dataset
        and a given set of features compared to FEATURE.
        It returns the dataframe corresponding to a heatmap of the rho
        containing only the features whose rho is superior to THRESHOLD.
        
        Parameters
        ----------------
        data            : pandas dataframe
                          Data containing interest_cols and FEATURE as features.
                          All features must be numeric.
                                 
        interest_cols   : list
                          The names of the features to calculate the rho
                          of Spearman compared to FEATURE
        
        FEATURE         : string
                          The name of the feature to calculate the rho of Spearman
                          against.
        
        THRESHOLD       : float
                          The function will return only those features whose rho is
                          superior to THRESHOLD
        
        Returns
        ---------------
        sorted_corrs_df : pandas dataframe
                          The rhos of Spearnma, sorted by feature of highest rho
        sorted_ps_df    : pandas dataframe
                          The p-values associated to the calculated rhos
    '''
    
    # Calcul du rho de Spearman avec p-value
    corrs, ps = stats.spearmanr(data[FEATURE],
                                data[interest_cols])

    # Transformation des arrays en DataFrame
    corrs_df = pd.DataFrame(corrs)
    ps_df = pd.DataFrame(ps)

    # Renommage des colonnes
    interest_cols.insert(0, FEATURE)
    corrs_df.columns=interest_cols
    ps_df.columns=interest_cols

    # Suppression des colonnes dont le rho est < THRESHOLD
    x = corrs_df[corrs_df[FEATURE]>THRESHOLD].index

    corrs_df = corrs_df.iloc[x,x].reset_index(drop=True).sort_values([FEATURE], ascending=False)
    ps_df = ps_df.iloc[x,x].reset_index(drop=True)

    # Classement des colonnes par ordre de rho décroissant
    sort_rows_ps_df = pd.DataFrame()

    for x in corrs_df.index.tolist():
        sort_rows_ps_df = pd.concat([sort_rows_ps_df, pd.DataFrame(ps_df.iloc[x, :]).T])

    # Tri par ordre de plus grand rho
    sorted_corrs_df = pd.DataFrame()
    sorted_ps_df = pd.DataFrame()

    for x in corrs_df.sort_values([FEATURE], ascending=False).index.tolist():
        sorted_corrs_df = pd.concat([sorted_corrs_df, corrs_df.iloc[:,x]], axis=1)
        sorted_ps_df = pd.concat([sorted_ps_df, sort_rows_ps_df.iloc[:,x]], axis=1)

    sorted_corrs_df["Index"] = sorted_corrs_df.columns
    sorted_corrs_df.set_index("Index", inplace=True)
    sorted_ps_df["Index"] = sorted_ps_df.columns
    sorted_ps_df.set_index("Index", inplace=True)

    sorted_corrs_df.index.name = None
    sorted_ps_df.index.name = None

    return (sorted_corrs_df, sorted_ps_df)

#------------------------------------------

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

#------------------------------------------

def plotRegplot(data_x, data_y, title, long, larg):
    '''
        Plots a regression plot of the columns col_y and
        col_x in data
        
        Parameters
        ----------------
        - data  : dataframe
                  Working data containing the col_x and col_y columns
        - col_x : string
                  the name of a column present in data
        - col_y : string
                  the name of a column present in data
        - title : string
                  the title of the figure
        - long  : int
                  the length of the figure
        - larg  : int
                  the widht of the figure
        
        Returns
        ---------------
        _
    '''

    TITLE_SIZE = 30
    TITLE_PAD = 1
    TICK_SIZE = 20
    TICK_PAD = 20
    LABEL_SIZE = 20
    LABEL_PAD = 30
    LINE_WIDTH = 3.5
    LEGEND_SIZE = 30

    sns.set(style="whitegrid")

    sns.set_palette(sns.dark_palette("purple", reverse=True))

    plt.rcParams["font.weight"] = "bold"

    plt.rc('font', size=LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=LABEL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)

    f, ax = plt.subplots(figsize=(long, larg))

    f.suptitle(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, y=TITLE_PAD)

    b = sns.regplot(x=data_x, y=data_y)

    lx = ax.get_xlabel()
    ax.set_xlabel(lx.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")
        
    ly = ax.get_ylabel()
    ax.set_ylabel(ly.upper(), fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    plt.show()
    
#------------------------------------------

def model_scores(pip,step,X_train):
    '''
        Retour des meilleurs scores NMAE et R2
        Stockage du dataframe de resultats du modèle

    '''
    df_results = pd.DataFrame.from_dict(pip.named_steps[step].cv_results_) \
                    .sort_values('rank_test_neg_mean_absolute_error')
    
    best_nmae = pip.named_steps[step].best_score_
    best_r2 = np.mean(df_results[df_results.rank_test_r2 == 1]['mean_test_r2'])
    #best_nmse = np.mean(df_results[df_results.rank_test_neg_mean_squared_error == 1] ['mean_test_neg_mean_squared_error'])
    best_params = pip.named_steps[step].best_params_
    training_time = round((np.mean(df_results.mean_fit_time)*X_train.shape[0]),2)
    print("Meilleur score MAE : {}\nMeilleur Score R2 : {}\nMeilleurs paramètres : {}\nTemps moyen d'entrainement : {}s"\
         .format(round(best_nmae,3), round(best_r2,3), best_params, training_time))
    return df_results

#------------------------------------------

def plotComparaisonResults(metrics_compare, cible):
    x = np.arange(len(metrics_compare.index))
    width = 0.35

    fig, ax = plt.subplots(1,2,figsize=(20,8), sharey=False, sharex=False)

    scores1 = ax[0].bar(x - width/2, -1*metrics_compare['mean_test_neg_mean_absolute_error'], width, label='Test')
    scores2 = ax[0].bar(x + width/2, -1*metrics_compare['mean_train_neg_mean_absolute_error'], width, label='Train')
    ax[0].set_ylabel('Mean Absolute Error', fontsize=14)
    ax[0].set_title('Comparaison des scores par modèle', fontsize=18)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(metrics_compare.index, fontsize=14)
    ax[0].legend()
    ax[0].bar_label(scores1, padding=3)
    ax[0].bar_label(scores2, padding=3)

    times1 = ax[1].bar(x - width/2, metrics_compare['mean_score_time'], width, label='Predict')
    times2 = ax[1].bar(x + width/2, metrics_compare['mean_fit_time'], width, label='Fit')
    ax[1].set_ylabel('Temps(s)', fontsize=14)
    ax[1].set_title("Comparaison des temps d'entrainement et prédiction", fontsize=18)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(metrics_compare.index, fontsize=14)
    ax[1].legend()
    ax[1].bar_label(times1, padding=3, fmt='%.3f')
    ax[1].bar_label(times2, padding=3, fmt='%.3f')

    plt.suptitle("Modélisations sur la variable "+ cible, fontsize=40, fontweight='bold')
    fig.tight_layout()

    plt.style.use('default')
    plt.show()

#------------------------------------------
    
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
        
    ------
    Code from :
        https://johaupt.github.io/
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [f for f in column]

        return [f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        feature_names.extend(get_names(trans))
    
    return feature_names    

#------------------------------------------

def metrics_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = y_true - y_pred
    mae = np.mean(abs(diff))
    r2 = 1-(sum(diff**2)/sum((y_true-np.mean(y_true))**2))
    dict_metrics = {"Métrique":["MAE", "R²"], "Résultats":[mae, r2]}
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics

#------------------------------------------

def plot_pred_true(y_true, y_pred, color=None, title=None):
    X_plot = [y_true.min(), y_true.max()]
    fig = plt.figure(figsize=(12,8))
    plt.scatter(y_true, y_pred, color=color, alpha=.6)
    plt.plot(X_plot, X_plot, color='r')
    plt.xlabel("Valeurs réélles")
    plt.ylabel("Valeurs prédites")
    plt.title("Valeurs prédites VS valeurs réélles | Variable {}".format(title), fontsize=18)
    plt.show()