import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
from math import ceil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patheffects as pe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from category_encoders import JamesSteinEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
from joblib import dump, load
import pytz
import os


def get_unique(df, n=20, sort='none', list=True, strip=False, count=False, percent=False, plot=False, cont=False):
    """
    Version 0.2
    Obtains unique values of all variables below a threshold number "n", and can display counts or percents
    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - n: int (default is 20). Maximum number of unique values to consider (avoid iterating continuous data)
    - sort: str, optional (default='none'). Determines the sorting of unique values:
        'none' will keep original order,
        'name' will sort alphabetically/numerically,
        'count' will sort by count of unique values (descending)
    - list: boolean, optional (default=True). Shows the list of unique values
    - strip: boolean, optional (default=False). True will remove single quotes in the variable names
    - count: boolean, optional (default=False). True will show counts of each unique value
    - percent: boolean, optional (default=False). True will show percentage of each unique value
    - plot: boolean, optional (default=False). True will show a basic chart for each variable
    - cont: boolean, optional (default=False). True will analyze variables over n as continuous

    Returns: None
    """
    # Calculate # of unique values for each variable in the dataframe
    var_list = df.nunique(axis=0)

    # Iterate through each categorical variable in the list below n
    print(f"\nCATEGORICAL: Variables with unique values equal to or below: {n}")
    for i in range(len(var_list)):
        var_name = var_list.index[i]
        unique_count = var_list[i]

        # If unique value count is less than n, get the list of values, counts, percentages
        if unique_count <= n:
            number = df[var_name].value_counts(dropna=False)
            perc = round(number / df.shape[0] * 100, 2)
            # Copy the index to a column
            orig = number.index
            # Strip out the single quotes
            name = [str(n) for n in number.index]
            name = [n.strip('\'') for n in name]
            # Store everything in dataframe uv for consistent access and sorting
            uv = pd.DataFrame({'orig': orig, 'name': name, 'number': number, 'perc': perc})

            # Sort the unique values by name or count, if specified
            if sort == 'name':
                uv = uv.sort_values(by='name', ascending=True)
            elif sort == 'count':
                uv = uv.sort_values(by='number', ascending=False)
            elif sort == 'percent':
                uv = uv.sort_values(by='perc', ascending=False)

            # Print out the list of unique values for each variable
            if list:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                for w, x, y, z in uv.itertuples(index=False):
                    # Decide on to use stripped name or not
                    if strip:
                        w = x
                    # Put some spacing after the value names for readability
                    w_str = str(w)
                    w_pad_size = uv.name.str.len().max() + 7
                    w_pad = w_str + " " * (w_pad_size - len(w_str))
                    y_str = str(y)
                    y_pad_max = uv.number.max()
                    y_pad_max_str = str(y_pad_max)
                    y_pad_size = len(y_pad_max_str) + 3
                    y_pad = y_str + " " * (y_pad_size - len(y_str))
                    if count and percent:
                        print("\t" + str(w_pad) + str(y_pad) + str(z) + "%")
                    elif count:
                        print("\t" + str(w_pad) + str(y))
                    elif percent:
                        print("\t" + str(w_pad) + str(z) + "%")
                    else:
                        print("\t" + str(w))

            # Plot countplot if plot=True
            if plot:
                print("\n")
                if strip:
                    if sort == 'count':
                        sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name)
                    else:
                        sns.barplot(data=uv, x=uv.loc[0], y='number', order=uv.sort_values('name', ascending=True).name)
                else:
                    if sort == 'count':
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('number', ascending=False).orig)
                    else:
                        sns.barplot(data=uv, x='orig', y='number', order=uv.sort_values('orig', ascending=True).orig)
                plt.title(var_name)
                plt.xlabel('')
                plt.ylabel('')
                plt.xticks(rotation=45)
                plt.show()

    if cont:
        # Iterate through each categorical variable in the list below n
        print(f"\nCONTINUOUS: Variables with unique values greater than: {n}")
        for i in range(len(var_list)):
            var_name = var_list.index[i]
            unique_count = var_list[i]

            if unique_count > n:
                print(f"\n{var_name} has {unique_count} unique values:\n")
                print(var_name)
                print(df[var_name].describe())

                # Plot countplot if plot=True
                if plot:
                    print("\n")
                    sns.histplot(data=df, x=var_name)
                    # plt.title(var_name)
                    # plt.xlabel('')
                    # plt.ylabel('')
                    # plt.xticks(rotation=45)
                    plt.show()


def plot_charts(df, plot_type='both', n=10, ncols=3, fig_width=20, subplot_height=4, rotation=45, strip=False,
                cat_cols=None, cont_cols=None, dtype_check=True, sample_size=None):
    """
    Version 0.2
    Plot barplots for categorical columns, or histograms for continuous columns, in a grid of subplots.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - plot_type: string, optional (default='both'). Type of charts to plot: 'cat' for categorical, 'cont' for
        continuous, 'both' for both
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - fig_width: int, optional (default=20). The width of the entire plot figure (not the subplot width)
    - subplot_height: int, optional (default=4). The height of each subplot.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.
    - strip: boolean, optional (default=False). Will strip single quotes from ends of column names
    - cat_cols: list, optional (default=None). A list of column names to treat as categorical variables. If not
        provided, inferred based on the unique count.
    - cont_cols: list, optional (default=None). A list of column names to treat as continuous variables. If not
        provided, inferred based on the unique count.
    - dtype_check: boolean, optional (default=True). If True, consider only numeric types (int64, float64) for
        continuous variables.
    - sample_size: float or int, optional (default=None). If provided and less than 1, the fraction of the data to
        sample. If greater than or equal to 1, the number of samples to draw.

    Returns: None
    """

    # Helper function to plot continuous variables
    def plot_continuous(df, cols, ncols, fig_width, subplot_height, strip, sample_size):
        nrows = ceil(len(cols) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = np.array(axs).ravel()  # Ensure axs is always a 1D numpy array

        # Loop through all continuous columns
        for i, col in enumerate(cols):
            if sample_size:
                sample_count = int(len(df[col].dropna()) * sample_size)  # Calculate number of samples
                data = df[col].dropna().sample(sample_count)
            else:
                data = df[col].dropna()

            if strip:
                sns.stripplot(x=data, ax=axs[i])
            else:
                sns.histplot(data, ax=axs[i], kde=False)

            axs[i].set_title(f'{col}', fontsize=20)
            axs[i].tick_params(axis='x', rotation=rotation)
            axs[i].set_xlabel('')

        # Remove empty subplots
        for empty_subplot in axs[len(cols):]:
            empty_subplot.remove()

    # Helper function to plot categorical variables
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size):
        nrows = ceil(len(cols) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = np.array(axs).ravel()  # Ensure axs is always a 1D numpy array

        # Loop through all categorical columns
        for i, col in enumerate(cols):
            uv = df[col].value_counts().reset_index().rename(columns={col: 'name', 'count': 'number'})
            uv['perc'] = uv['number'] / uv['number'].sum()

            if sample_size:
                uv = uv.sample(sample_size)

            sns.barplot(data=uv, x='name', y='number', order=uv.sort_values('number', ascending=False).name, ax=axs[i])

            axs[i].set_title(f'{col}', fontsize=20)
            axs[i].tick_params(axis='x', rotation=rotation)
            axs[i].set_ylabel('Count')
            axs[i].set_xlabel('')

        # Remove empty subplots
        for empty_subplot in axs[len(cols):]:
            empty_subplot.remove()

    # Compute unique counts and identify categorical and continuous variables
    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, strip, sample_size)


def plot_charts_with_hue(df, plot_type='both', n=10, ncols=3, fig_width=20, subplot_height=4, rotation=0,
                         cat_cols=None, cont_cols=None, dtype_check=True, sample_size=None, hue=None, color_discrete_map=None, normalize=False, kde=False, multiple='layer'):
    """
    Version 0.1
    Plot barplots for categorical columns, or histograms for continuous columns, in a grid of subplots.
    Option to pass a 'hue' parameter to dimenions the plots by a variable/column of the dataframe.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - plot_type: string, optional (default='both'). Type of charts to plot: 'cat' for categorical, 'cont' for continuous, 'both' for both
    - n: int (default=20). Threshold of unique values for categorical (equal or below) vs. continuous (above)
    - ncols: int, optional (default=3). The number of columns in the subplot grid.
    - fig_width: int, optional (default=20). The width of the entire plot figure (not the subplot width)
    - subplot_height: int, optional (default=4). The height of each subplot.
    - rotation: int, optional (default=45). The rotation of the x-axis labels.
    - cat_cols: list, optional (default=None). A list of column names to treat as categorical variables. If not provided, inferred based on the unique count.
    - cont_cols: list, optional (default=None). A list of column names to treat as continuous variables. If not provided, inferred based on the unique count.
    - dtype_check: boolean, optional (default=True). If True, consider only numeric types (int64, float64) for continuous variables.
    - sample_size: float or int, optional (default=None). If provided and less than 1, the fraction of the data to sample. If greater than or equal to 1, the number of samples to draw.
    - hue: string, optional (default=None). Name of the column to dimension by passing as 'hue' to the Seaborn charts.
    - color_discrete_map: name of array or array, optional (default=None). Pass a color mapping for the values in the 'hue' variable.
    - normalize: boolean, optional (default=False). Set to True to normalize categorical plots and see proportions instead of counts
    - kde: boolean, optional (default=False). Set to show KDE line on continuous countplots
    - multiple: 'layer', 'dodge', 'stack', 'fill', optional (default='layer'). Choose how to handle hue variable when plotted on countplots
    Returns: None
    """
    def plot_categorical(df, cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize):
        if sample_size:
            df = df.sample(sample_size)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows*subplot_height), constrained_layout=True)
        axs = axs.ravel()

        for i, col in enumerate(cols):
            if normalize:
                # Normalize the counts
                df_copy = df.copy()
                data = df_copy.groupby(col)[hue].value_counts(normalize=True).rename('proportion').reset_index()
                sns.barplot(data=data, x=col, y='proportion', hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Proportion', fontsize=12)
            else:
                sns.countplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i])
                axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].tick_params(axis='x', rotation=rotation)

        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)

    def plot_continuous(df, cols, ncols=3, fig_width=15, subplot_height=5, sample_size=None, hue=None, color_discrete_map=None, kde=False, multiple=multiple):
        if sample_size:
            df = df.sample(sample_size)
        nplots = len(cols)
        nrows = nplots//ncols
        if nplots % ncols:
            nrows += 1
            
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, nrows * subplot_height), constrained_layout=True)
        axs = axs.ravel()
        
        for i, col in enumerate(cols):
            if hue is not None:
                sns.histplot(data=df, x=col, hue=hue, palette=color_discrete_map, ax=axs[i], kde=kde, multiple=multiple)
            else:
                sns.histplot(data=df, x=col, ax=axs[i])
            axs[i].set_title(col, fontsize=16, pad=10)
            axs[i].set_ylabel('Count', fontsize=12)
            axs[i].set_xlabel(' ', fontsize=12)
            
        # Remove empty subplots
        for empty_subplot in axs[nplots:]:
            fig.delaxes(empty_subplot)
            
    unique_count = df.nunique()
    if cat_cols is None:
        cat_cols = unique_count[unique_count <= n].index.tolist()
        if hue in cat_cols:
            cat_cols.remove(hue)
    if cont_cols is None:
        cont_cols = unique_count[unique_count > n].index.tolist()

    if dtype_check:
        cont_cols = [col for col in cont_cols if df[col].dtype in ['int64', 'float64']]

    if plot_type == 'cat' or plot_type == 'both':
        plot_categorical(df, cat_cols, ncols, fig_width, subplot_height, rotation, sample_size, hue, color_discrete_map, normalize)
    if plot_type == 'cont' or plot_type == 'both':
        plot_continuous(df, cont_cols, ncols, fig_width, subplot_height, sample_size, hue, color_discrete_map, kde, multiple)


def plot_corr(df, column, n, meth='pearson', size=(15, 8), rot=45, pal='RdYlGn', rnd=2):
    """
    Version 0.2
    Create a barplot that shows correlation values for one variable against others.
    Essentially one slice of a heatmap, but the bars show the height of the correlation
    in addition to the color. It will only look at numeric variables.

    Parameters:
    - df: dataframe that contains the variables you want to analyze
    - column: string. Column name that you want to evaluate the correlations against
    - n: int. The number of correlations to show (split evenly between positive and negative correlations)
    - meth: optional (default='pearson'). See df.corr() method options
    - size: tuple of ints, optional (default=(15, 8)). The size of the plot
    - rot: int, optional (default=45). The rotation of the x-axis labels
    - pal: string, optional (default='RdYlGn'). The color map to use
    - rnd: int, optional (default=2). Number of decimal places to round to

    Returns: None
    """
    # Calculate correlations
    corr = round(df.corr(method=meth, numeric_only=True)[column].sort_values(), rnd)

    # Drop column from correlations (correlating with itself)
    corr = corr.drop(column)

    # Get the most negative and most positive correlations, sorted by absolute value
    most_negative = corr.sort_values().head(n // 2)
    most_positive = corr.sort_values().tail(n // 2)

    # Concatenate these two series and sort the final series by correlation value
    corr = pd.concat([most_negative, most_positive]).sort_values()

    # Generate colors based on correlation values using a colormap
    cmap = plt.get_cmap(pal)
    colors = cmap((corr.values + 1) / 2)

    # Plot the chart
    plt.figure(figsize=size)
    plt.axhline(y=0, color='lightgrey', alpha=0.8, linestyle='-')
    bars = plt.bar(corr.index, corr.values, color=colors)

    # Add value labels to the end of each bar
    for bar in bars:
        yval = bar.get_height()
        if yval < 0:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval - 0.05, yval, va='top')
        else:
            plt.text(bar.get_x() + bar.get_width() / 3.0, yval + 0.05, yval, va='bottom')

    plt.title('Correlation with ' + column, fontsize=20)
    plt.ylabel('Correlation', fontsize=14)
    plt.xlabel('Other Variables', fontsize=14)
    plt.xticks(rotation=rot)
    plt.ylim(-1, 1)
    plt.show()


def split_dataframe(df, n):
    """
    Split a DataFrame into two based on the number of unique values in each column.

    Parameters:
    - df: DataFrame. The DataFrame to split.
    - n: int. The maximum number of unique values for a column to be considered categorical.

    Returns:
    - df_cat: DataFrame. Contains the columns of df with n or fewer unique values.
    - df_num: DataFrame. Contains the columns of df with more than n unique values.
    """
    df_cat = pd.DataFrame()
    df_num = pd.DataFrame()

    for col in df.columns:
        if df[col].nunique() <= n:
            df_cat[col] = df[col]
        else:
            df_num[col] = df[col]

    return df_cat, df_num


def thousands(x, pos):
    """
    Format a number with thousands separators.

    Parameters:
    - x: float. The number to format.
    - pos: int. The position of the number.

    Returns:
    - s: string. The formatted number.
    """
    s = '{:0,d}'.format(int(x))
    return s

def thousand_dollars(x, pos):
    """
    Format a number with thousands separators.

    Parameters:
    - x: float. The number to format.
    - pos: int. The position of the number.

    Returns:
    - s: string. The formatted number.
    """
    s = '${:0,d}'.format(int(x))
    return s

def visualize_kmeans(df, x_var, y_var, centers=3, iterations=100):
    # Select centers at random
    starting_centers = df.sample(centers).reset_index(drop=True)

    # Make a list to hold the center values
    center_values = [starting_centers[[x_var, y_var]].iloc[i].values for i in range(centers)]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_var, y=y_var, palette='tab10')
    plt.scatter(starting_centers[x_var], starting_centers[y_var], marker='*', s=400, c='red', edgecolor='black')
    plt.title('Starting Centers')
    plt.show()

    # For each iteration
    for i in range(iterations):
        # Determine intercluster variance
        dists = [np.linalg.norm(df[[x_var, y_var]] - center_values[j], axis=1)**2 for j in range(centers)]
        dist_X = pd.DataFrame(np.array(dists).T, columns=['d' + str(j+1) for j in range(centers)])

        # Make cluster assignments
        df['Cluster Label'] = np.argmin(dist_X.values, axis=1)

        # Update centroids
        new_centers = df.groupby('Cluster Label').mean()

        # Update the center values
        center_values = [new_centers[[x_var, y_var]].iloc[j].values for j in range(centers)]

        plt_title = 'Iteration ' + str(i+1)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_var, y=y_var, hue='Cluster Label', palette='tab10')
        plt.scatter(new_centers[x_var], new_centers[y_var], marker='*', s=400, c='red', edgecolor='black')
        plt.title(plt_title)
        plt.show()

    return df


import seaborn as sns
import plotly.express as px

def plot_3d(df, x, y, z, color=None, color_map=None, scale='linear'):
    """
    Create a 3D scatter plot using Plotly Express.

    Parameters:
    - df: DataFrame. The input dataframe.
    - x: str. The column name to be used for the x-axis.
    - y: str. The column name to be used for the y-axis.
    - z: str. The column name to be used for the z-axis.
    - color: str, optional (default=None). The column name to be used for color coding the points.
    - color_map: list of str, optional (default=None). The color map to be used. If None, the seaborn default color palette will be used.
    - scale: str, optional (default='linear'). The scale type for the axis. Use 'log' for logarithmic scale.

    Returns: None
    """
    if color_map is None:
        color_map = sns.color_palette().as_hex()

    fig = px.scatter_3d(df, 
                    x=x, 
                    y=y, 
                    z=z, 
                    color=color,
                    color_discrete_sequence=color_map,
                    height=600, 
                    width=1000)
    title_text = "{}, {}, {} by {}".format(x, y, z, color)
    fig.update_layout(title={'text': title_text, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      showlegend=True,
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=x,
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            type=scale), 
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title=y,
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            type=scale), 
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black', 
                                            gridcolor='#f0f0f0',
                                            title=z,
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            type=scale)))
    fig.update_traces(marker=dict(size=3, opacity=1, line=dict(color='black', width=0.1)))
    fig.show()


def plot_map_ca(df, lon='Longitude', lat='Latitude', hue=None, size=None, size_range=(50, 200), title='Geographic Chart', dot_size=None, alpha=0.8, color_map=None, fig_size=(12, 12)):
    """
    Version 0.1
    Plots a geographic map of California with data points overlaid.

    Parameters:
    - df: DataFrame containing the data to be plotted
    - lon: str, optional (default='Longitude'). Column name in `df` representing the longitude coordinates
    - lat: str, optional (default='Latitude'). Column name in `df` representing the latitude coordinates
    - hue: str, optional (default=None). Column name in `df` for color-coding the points
    - size: str, optional (default=None). Column name in `df` to scale the size of points
    - size_range: tuple, optional (default=(50, 200)). Range of sizes if the `size` parameter is used
    - title: str, optional (default='Geographic Chart'). Title of the plot
    - dot_size: int, optional (default=None). Size of all dots if you want them to be uniform
    - alpha: float, optional (default=0.8). Transparency of the points
    - color_map: colormap, optional (default=None). Colormap to be used if `hue` is specified
    - fig_size: tuple, optional (default=(12, 12)). Size of the figure

    Returns: None
    """
    # Define the locations of major cities
    large_ca_cities = {'Name': ['Fresno', 'Los Angeles', 'Sacramento', 'San Diego', 'San Francisco', 'San Jose'],
                       'Latitude': [36.746842, 34.052233, 38.581572, 32.715328, 37.774931, 37.339386],
                       'Longitude': [-119.772586, -118.243686, -121.494400, -117.157256, -122.419417, -121.894956],
                       'County': ['Fresno', 'Los Angeles', 'Sacramento', 'San Diego', 'San Francisco', 'Santa Clara']}
    df_large_cities = pd.DataFrame(large_ca_cities)

    # Create a figure that utilizes Cartopy
    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -114, 32, 42])

    # Add geographic details
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

    # Add county boundaries
    counties = gpd.read_file('data/cb_2018_us_county_5m.shp')
    counties_ca = counties[counties['STATEFP'] == '06']
    counties_ca = counties_ca.to_crs("EPSG:4326")
    for geometry in counties_ca['geometry']:
        ax.add_geometries([geometry], crs=ccrs.PlateCarree(), edgecolor='grey', alpha=0.3, facecolor='none')

    # Draw the scatterplot of data
    if dot_size:
        ax.scatter(df[lon], df[lat], s=dot_size, cmap=color_map, alpha=alpha, transform=ccrs.PlateCarree())
    else:
        sns.scatterplot(data=df, x=lon, y=lat, hue=hue, size=size, alpha=alpha, ax=ax, palette=color_map, sizes=size_range)

    # Add cities
    ax.scatter(df_large_cities['Longitude'], df_large_cities['Latitude'], transform=ccrs.PlateCarree(), edgecolor='black')
    for x, y, label in zip(df_large_cities['Longitude'], df_large_cities['Latitude'], df_large_cities['Name']):
        text = ax.text(x + 0.05, y + 0.05, label, transform=ccrs.PlateCarree(), fontsize=12, ha='left', fontname='Arial')
        text.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

    # Finish up the chart
    ax.set_title(title, fontsize=18, pad=15)
    ax.set_xlabel('Longitude', fontsize=14, labelpad=15)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.gridlines(draw_labels=True, color='lightgrey', alpha=0.5)
    plt.show()
    
    
import pandas as pd
import numpy as np

def get_corr(df, n=5, var=None, show_results=True, return_arrays=False):
    """
    Gets the top n positive and negative correlations in a dataframe. Returns them in two
    arrays. By default, prints a summary of the top positive and negative correlations.

    Parameters
    ----------
    - df : pandas.DataFrame. The dataframe you wish to analyze for correlations
    - n : int, default 5. The number of top positive and negative correlations to list.
    - var : str, (optional) default None. The variable of interest. If provided, the function
        will only show the top n positive and negative correlations for this variable.
    - show_results : boolean, default True. Print the results.
    - return_arrays : boolean, default False. If true, return arrays with column names

    Returns: Tuple (if return_arrays == True)
        - positive_variables: array of variable names involved in top n positive correlations
        - negative_variables: array of variable names involved in top n negative correlations
    """
    pd.set_option('display.expand_frame_repr', False)

    corr = round(df.corr(numeric_only=True), 2)
    
    # Unstack correlation matrix into a DataFrame
    corr_df = corr.unstack().reset_index()
    corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # If a variable is specified, filter to correlations involving that variable
    if var is not None:
        corr_df = corr_df[(corr_df['Variable 1'] == var) | (corr_df['Variable 2'] == var)]

    # Remove self-correlations and duplicates
    corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]
    corr_df[['Variable 1', 'Variable 2']] = np.sort(corr_df[['Variable 1', 'Variable 2']], axis=1)
    corr_df = corr_df.drop_duplicates(subset=['Variable 1', 'Variable 2'])

    # Sort by absolute correlation value from highest to lowest
    corr_df['AbsCorrelation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values(by='AbsCorrelation', ascending=False)

    # Drop the absolute value column
    corr_df = corr_df.drop(columns='AbsCorrelation').reset_index(drop=True)

    # Get the first n positive and negative correlations
    positive_corr = corr_df[corr_df['Correlation'] > 0].head(n).reset_index(drop=True)
    negative_corr = corr_df[corr_df['Correlation'] < 0].head(n).reset_index(drop=True)

    # Print the results
    if show_results:
        print("Top", n, "positive correlations:")
        print(positive_corr)
        print("\nTop", n, "negative correlations:")
        print(negative_corr)

    # Return the arrays
    if return_arrays:
        # Remove target variable from the arrays
        positive_variables = positive_corr[['Variable 1', 'Variable 2']].values.flatten()
        positive_variables = positive_variables[positive_variables != var]

        negative_variables = negative_corr[['Variable 1', 'Variable 2']].values.flatten()
        negative_variables = negative_variables[negative_variables != var]

        return positive_variables, negative_variables

    
def sk_vif(exogs, data):
    # Set a high threshold, e.g., 1e10, for very large VIFs
    MAX_VIF = 1e10

    vif_dict = {}

    for exog in exogs:
        not_exog = [i for i in exogs if i !=exog]
        # split the dataset, one independent variable against all others
        X, y = data[not_exog], data[exog]

        # fit the model and obtain R^2
        r_squared = LinearRegression().fit(X,y).score(X,y)

        # compute the VIF, with a check for r_squared close to 1
        if 1 - r_squared < 1e-5:  # or some other small threshold that makes sense for your application
            vif = MAX_VIF
        else:
            vif = 1/(1-r_squared)

        vif_dict[exog] = vif

    return pd.DataFrame({"VIF": vif_dict})


def calc_vif(X):

    # Calculate Variance Inflation Factor (VIF) to find which features have mutlticollinearity
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif.sort_values(by='VIF', ascending=False))


def calc_fpi(model, X, y, n_repeats=10, random_state=42):
    
    # Calculate Feature Permutation Importance to find out which features have the most effect
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

    return pd.DataFrame({"Variables": X.columns,
                         "Score Mean": r.importances_mean,
                         "Score Std": r.importances_std}).sort_values(by="Score Mean", ascending=False)


from sklearn import linear_model

# List of classes that support the .coef_ attribute
SUPPORTED_COEF_CLASSES = (
    linear_model.LogisticRegression,
    linear_model.LogisticRegressionCV,
    linear_model.PassiveAggressiveClassifier,
    linear_model.Perceptron,
    linear_model.RidgeClassifier,
    linear_model.RidgeClassifierCV,
    linear_model.SGDClassifier,
    linear_model.SGDOneClassSVM,
    linear_model.LinearRegression,
    linear_model.Ridge,
    linear_model.RidgeCV,
    linear_model.SGDRegressor,
    linear_model.ElasticNet,
    linear_model.ElasticNetCV,
    linear_model.Lars,
    linear_model.LarsCV,
    linear_model.Lasso,
    linear_model.LassoCV,
    linear_model.LassoLars,
    linear_model.LassoLarsCV,
    linear_model.LassoLarsIC,
    linear_model.OrthogonalMatchingPursuit,
    linear_model.OrthogonalMatchingPursuitCV,
    linear_model.ARDRegression,
    linear_model.BayesianRidge,
    linear_model.HuberRegressor,
    linear_model.QuantileRegressor,
    linear_model.RANSACRegressor,
    linear_model.TheilSenRegressor
)

def supports_coef(estimator):
    """Check if estimator supports .coef_"""
    return isinstance(estimator, SUPPORTED_COEF_CLASSES)

def extract_features_and_coefficients(grid_or_pipe, X, debug=False):
    # Determine the type of the passed object and set flags
    if hasattr(grid_or_pipe, 'best_estimator_'):
        estimator = grid_or_pipe.best_estimator_
        is_grid = True
        is_pipe = False
        if debug:
            print('Grid: ', is_grid)
    else:
        estimator = grid_or_pipe
        is_pipe = True
        is_grid = False
        if debug:
            print('Pipe: ', is_pipe)

    # Initial setup
    current_features = list(X.columns)
    if debug:
        print('current_features: ', current_features)
    mapping = pd.DataFrame({
        'feature_name': current_features,
        'intermediate_name1': current_features,
        'selected': [True] * len(current_features),
        'coefficients': [None] * len(current_features)
    })

    for step_name, step_transformer in estimator.named_steps.items():
        if debug:
            print(f"Processing step: {step_name} in {step_transformer}")  # Debugging

        n_features_in = len(current_features)  # Number of features at the start of this step

        # If transformer is a ColumnTransformer
        if isinstance(step_transformer, ColumnTransformer):
            new_features = []  # Collect new features from this step
            step_transformer_list = step_transformer.transformers_
            for name, trans, columns in step_transformer_list:
                # OneHotEncoder or similar expanding transformers
                if hasattr(trans, 'get_feature_names_out'):
                    out_features = list(trans.get_feature_names_out(columns))
                    new_features.extend(out_features)
                else:
                    new_features.extend(columns)

            current_features = new_features

            # Update mapping based on current_features
            mapping = pd.DataFrame({
                'feature_name': current_features,
                'intermediate_name1': current_features,
                'selected': [True] * len(current_features),
                'coefficients': [None] * len(current_features)
            })
            if debug:
                print("Mapping: ", mapping)

        # Reduction
        elif hasattr(step_transformer, 'get_support'):
            mask = step_transformer.get_support()
            # Update selected column in mapping
            mapping.loc[mapping['feature_name'].isin(current_features), 'selected'] = mask
            current_features = mapping[mapping['selected']]['feature_name'].tolist()

    # Inside your extract_features_and_coefficients function:

    # If there's a model with coefficients in this step, update coefficients
    if supports_coef(step_transformer):
        coefficients = step_transformer.coef_.ravel()
        selected_rows = mapping[mapping['selected']].index
        if debug:
            print("Coefficients: ", coefficients)
            print(f"Number of coefficients: {len(coefficients)}")  # Debugging
            print(f"Number of selected rows: {len(selected_rows)}")  # Debugging

        if len(coefficients) == len(selected_rows):
            mapping.loc[selected_rows, 'coefficients'] = coefficients.tolist()
        else:
            print(f"Mismatch in coefficients and selected rows for step: {step_name}")

    # For transformers inside ColumnTransformer
    if isinstance(step_transformer, ColumnTransformer):
        if debug:
            print("ColumnTransformer:", step_transformer)
        transformers = step_transformer.transformers_
        if debug:
            print("Transformers: ", transformers)
        new_features = []  # Collect new features from this step
        for name, trans, columns in transformers:
            # OneHotEncoder or similar expanding transformers
            if hasattr(trans, 'get_feature_names_out'):
                out_features = list(trans.get_feature_names_out(columns))
                new_features.extend(out_features)
                if debug:
                    print("Out features: ", out_features)
                    print("New features: ", new_features)
            else:
                new_features.extend(columns)
        
        current_features = new_features

        # Update mapping based on current_features
        mapping = pd.DataFrame({
            'feature_name': current_features,
            'intermediate_name1': current_features,
            'selected': [True] * len(current_features),
            'coefficients': [None] * len(current_features)
        })
        if debug:
            print("Mapping: ", mapping)
    # Filtering the final selected features and their coefficients
    final_data = mapping[mapping['selected']]
    
    return final_data[['feature_name', 'coefficients']]

    
    


# MODEL ITERATION: default_config, create_pipeline, iterate_model

# default_config: Version 0.1
# Default configuration of parameters used by iterate_model and create_pipeline
# New configurations can be passed in by the user when function is called
# 

# create_pipeline: Version 0.1
#
def create_pipeline(transformer_keys=None, scaler_key=None, selector_key=None, model_key=None, config=None, X_cat_columns=None, X_num_columns=None):
    """
    Creates a pipeline for data preprocessing and modeling.

    This function allows for flexibility in defining the preprocessing and 
    modeling steps of the pipeline. You can specify which transformers to apply 
    to the data, whether to scale the data, and which model to use for predictions. 
    If a step is not specified, it will be skipped.

    Parameters:
    - model_key (str): The key corresponding to the model in the config['models'] dictionary.
    - transformer_keys (list of str, str, or None): The keys corresponding to the transformers 
        to apply to the data. This can be a list of string keys or a single string key corresponding 
        to transformers in the config['transformers'] dictionary. If not provided, no transformers will be applied.
    - scaler_key (str or None): The key corresponding to the scaler to use to scale the data. 
        This can be a string key corresponding to a scaler in the config['scalers'] dictionary. 
        If not provided, the data will not be scaled.
    - selector_key (str or None): The key corresponding to the feature selector. 
        This can be a string key corresponding to a scaler in the config['selectors'] dictionary. 
        If not provided, no feature selection will be performed.
    - X_num_columns (list-like, optional): List of numeric columns from the input dataframe. This is used
        in the default_config for the relevant transformers.
    - X_cat_columns (list-like, optional): List of categorical columns from the input dataframe. This is used
        in the default_config for the elevant encoders.

    Returns:
    pipeline (sklearn.pipeline.Pipeline): A scikit-learn pipeline consisting of the specified steps.

    Example:
    >>> pipeline = create_pipeline('linreg', transformer_keys=['ohe', 'poly2'], scaler_key='stand', config=my_config)
    """

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        # If no column lists are provided, raise an error
        if not X_cat_columns and not X_num_columns:
            raise ValueError("If no config is provided, X_cat_columns and X_num_columns must be passed.")
        config = {
            'transformers': {
                'ohe': (OneHotEncoder(drop='if_binary', handle_unknown='ignore'), X_cat_columns),
                'ord': (OrdinalEncoder(), X_cat_columns),
                'js': (JamesSteinEncoder(), X_cat_columns),
                'poly2': (PolynomialFeatures(degree=2, include_bias=False), X_num_columns),
                'poly2_bias': (PolynomialFeatures(degree=2, include_bias=True), X_num_columns),
                'poly3': (PolynomialFeatures(degree=3, include_bias=False), X_num_columns),
                'poly3_bias': (PolynomialFeatures(degree=3, include_bias=True), X_num_columns),
                'log': (FunctionTransformer(np.log1p, validate=True), X_num_columns)
            },
            'scalers': {
                'stand': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler()
            },
            'selectors': {
                'sfs': SequentialFeatureSelector(LinearRegression()),
                'sfs_7': SequentialFeatureSelector(LinearRegression(), n_features_to_select=7),
                'sfs_6': SequentialFeatureSelector(LinearRegression(), n_features_to_select=6),
                'sfs_5': SequentialFeatureSelector(LinearRegression(), n_features_to_select=5),
                'sfs_4': SequentialFeatureSelector(LinearRegression(), n_features_to_select=4),
                'sfs_3': SequentialFeatureSelector(LinearRegression(), n_features_to_select=3),
                'sfs_bw': SequentialFeatureSelector(LinearRegression(), direction='backward')
            },
            'models': {
                'linreg': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(random_state=42),
                'random_forest': RandomForestRegressor(),
                'gradient_boost': GradientBoostingRegressor(),
            }
        }

    # Initialize an empty list for the transformation steps
    steps = []

    # If transformers are provided, add them to the steps
    if transformer_keys is not None:
        transformer_steps = []

        for key in (transformer_keys if isinstance(transformer_keys, list) else [transformer_keys]):
            transformer, cols = config['transformers'][key]

            transformer_steps.append((key, transformer, cols))

        # Create column transformer
        col_trans = ColumnTransformer(transformer_steps, remainder='passthrough')
        transformer_name = 'Transformers: ' + '_'.join(transformer_keys) if isinstance(transformer_keys, list) else 'Transformers: ' + transformer_keys
        steps.append((transformer_name, col_trans))


    # If a scaler is provided, add it to the steps
    if scaler_key is not None:
        scaler_obj = config['scalers'][scaler_key]
        steps.append(('Scaler: ' + scaler_key, scaler_obj))

    # If a selector is provided, add it to the steps
    if selector_key is not None:
        selector_obj = config['selectors'][selector_key]
        steps.append(('Selector: ' + selector_key, selector_obj))

    # If a model is provided, add it to the steps
    if model_key is not None:
        model_obj = config['models'][model_key]
        steps.append(('Model: ' + model_key, model_obj))

    # Create and return pipeline
    return Pipeline(steps)

 
# Define the results_df global variable that will store the results of iterate_model if Save=True:
results_df = None
# This needs to be initialized in notebook with the following lines of code:
# import mytools as my
# my.results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE', 'Train MAE',
#                'Test MAE', 'Train R^2 Score', 'Test R^2 Score', 'Pipeline', 'Best Grid Params', 'Note', 'Date'])

def iterate_model(Xn_train, Xn_test, yn_train, yn_test, model=None, transformers=None, scaler=None, selector=None, drop=None, iteration='1', note='', save=False, export=False, plot=False, coef=False, perm=False, vif=False, cross=False, cv_folds=5, config=None, debug=False, grid=False, grid_params=None, grid_cv=None, grid_score='r2', grid_verbose=1, decimal=2, lowess=False):
    """
    Creates a pipeline from specified parameters for transformers, scalers, and models. Parameters must be
    defined in configuration dictionary containing 3 dictionaries: transformer_dict, scaler_dict, model_dict.
    See 'default_config' in this library file for reference, customize at will. Then fits the pipeline to the passed
    training data, and evaluates its performance with both test and training data. Options to see plots of residuals
    and actuals vs. predicted, save results to results_df with user-defined note, display coefficients, calculate
    permutation feature importance, variance inflation factor (VIF), and cross-validation.
    
    Parameters:
    - Xn_train, Xn_test: Training and test feature sets.
    - yn_train, yn_test: Training and test target sets.
    - config: Configuration dictionary of parameters for pipeline construction (see default_config)
    - model: Key for the model to be used (ex: 'linreg', 'lasso', 'ridge').
    - transformers: List of transformation keys to apply (ex: ['ohe', 'poly2']).
    - scaler: Key for the scaler to be applied (ex: 'stand')
    - selector: Key for the selector to be applied (ex: 'sfs')
    - drop: List of columns to be dropped from the training and test sets.
    - iteration: A string identifier for the iteration.
    - note: Any note or comment to be added for the iteration.
    - save: Boolean flag to decide if the results should be saved to the global results dataframe (results_df).
    - plot: Flag to plot residual plots and actuals vs. predicted for training and test data.
    - coef: Flag to print and plot model coefficients.
    - perm: Flag to compute and display permutation feature importance.
    - vif: Flag to calculate and display Variance Inflation Factor for features.
    - cross: Flag to perform cross-validation and print results.
    - cv_folds: Number of folds to be used for cross-validation if cross=True.
    - debug: Flag to show debugging information like the details of the pipeline.

    Prerequisites:
    - Dictionaries of parameters for transformers, scalers, and models: transformer_dict, scaler_dict, model_dict.
    - Lists identifying columns for various transformations and encodings, e.g., ohe_columns, ord_columns, etc.

    Outputs:
    - Prints results, performance metrics, and other specified outputs.
    - Updates the global results dataframe if save=True.
    - Displays plots based on flags like plot, coef.
    
    Usage:
    >>> iterate_model(X_train, X_test, y_train, y_test, transformers=['ohe','poly2'], scaler='stand', model='linreg', drop=['col1'], iteration="1", save=True, plot=True)
    """
    # Drop specified columns from Xn_train and Xn_test
    if drop is not None:
        Xn_train = Xn_train.drop(columns=drop)
        Xn_test = Xn_test.drop(columns=drop)
        if debug:
            print('Drop:', drop)
            print('Xn_train.columns', Xn_train.columns)
            print('Xn_test.columns', Xn_test.columns)

    # Check for configuration file parameter, if none, use default in library
    if config is None:
        X_num_columns = Xn_train.select_dtypes(include=[np.number]).columns.tolist()
        X_cat_columns = Xn_train.select_dtypes(exclude=[np.number]).columns.tolist()
        if debug:
            print('Config:', config)
            print('X_num_columns:', X_num_columns)
            print('X_cat_columns:', X_cat_columns)
    else:
        X_num_columns = None
        X_cat_columns = None

    # Create a pipeline from transformer and model parameters
    pipe = create_pipeline(transformer_keys=transformers, scaler_key=scaler, selector_key=selector, model_key=model, config=config, X_cat_columns=X_cat_columns, X_num_columns=X_num_columns)
    if debug:
        print('Pipeline:', pipe)
        print('Pipeline Parameters:', pipe.get_params())

    # Construct format string
    format_str = f',.{decimal}f'
        
    # Print some metadata
    print(f'\nITERATION {iteration} RESULTS\n')
    pipe_steps = " -> ".join(pipe.named_steps.keys())
    print(f'Pipeline: {pipe_steps}')
    if note: print(f'Note: {note}')
    # Get the current date and time
    current_time = datetime.now(pytz.timezone('US/Pacific'))
    timestamp = current_time.strftime('%b %d, %Y %I:%M %p PST')
    print(f'{timestamp}\n')

    if cross or grid:
        print('Cross Validation:\n')
    # Before fitting the pipeline, check if cross-validation is desired:
    if cross:
        # Flatten yn_train for compatibility
        yn_train_flat = yn_train.values.flatten() if isinstance(yn_train, pd.Series) else np.array(yn_train).flatten()
        cv_scores = cross_val_score(pipe, Xn_train, yn_train_flat, cv=cv_folds, scoring='r2')

        print(f'Cross-Validation (R^2) Scores for {cv_folds} Folds:')
        for i, score in enumerate(cv_scores, 1):
            print(f'Fold {i}: {score:{format_str}}')
        print(f'Average: {np.mean(cv_scores):{format_str}}')
        print(f'Standard Deviation: {np.std(cv_scores):{format_str}}\n')

    if grid:
        
        grid = GridSearchCV(pipe, param_grid=config['params'][grid_params], cv=config['cv'][grid_cv], scoring=grid_score, verbose=grid_verbose)
        if debug:
            print('Grid: ', grid)
            print('Grid Parameters: ', grid.get_params())
        # Fit the grid and predict
        grid.fit(Xn_train, yn_train)
        #best_model = grid.best_estimator_
        best_model = grid
        yn_train_pred = grid.predict(Xn_train)
        yn_test_pred = grid.predict(Xn_test)
        if debug:
            print("First 10 actual train values:", yn_train[:10])
            print("First 10 predicted train values:", yn_train_pred[:10])
            print("First 10 actual test values:", yn_test[:10])
            print("First 10 predicted test values:", yn_test_pred[:10])
        best_grid_params = grid.best_params_
        best_grid_score = grid.best_score_
        best_grid_estimator = grid.best_estimator_
        best_grid_index = grid.best_index_
        grid_results = grid.cv_results_
    else:
        best_grid_params = None
        best_grid_score = None
        # Fit the pipeline and predict
        pipe.fit(Xn_train, yn_train)
        best_model = pipe
        yn_train_pred = pipe.predict(Xn_train)
        yn_test_pred = pipe.predict(Xn_test)

    # MSE
    yn_train_mse = mean_squared_error(yn_train, yn_train_pred)
    yn_test_mse = mean_squared_error(yn_test, yn_test_pred)

    # RMSE
    yn_train_rmse = np.sqrt(yn_train_mse)
    yn_test_rmse = np.sqrt(yn_test_mse)

    # MAE
    yn_train_mae = mean_absolute_error(yn_train, yn_train_pred)
    yn_test_mae = mean_absolute_error(yn_test, yn_test_pred)

    # R^2 Score
    if grid:
        if grid_score == 'r2':
            train_score = grid.score(Xn_train, yn_train)
            test_score = grid.score(Xn_test, yn_test)
        else:
            train_score = 0
            test_score = 0
    else:
        train_score = pipe.score(Xn_train, yn_train)
        test_score = pipe.score(Xn_test, yn_test)

    # Print Grid best parameters
    if grid:
        print(f'\nBest Grid mean score ({grid_score}): {best_grid_score:{format_str}}')
        #print(f'Best Grid parameters: {best_grid_params}\n')        
        param_str = ', '.join(f"{key}: {value}" for key, value in best_grid_params.items())
        print(f"Best Grid parameters: {param_str}\n")
        #print(f'Best Grid estimator: {best_grid_estimator}')
        #print(f'Best Grid index: {best_grid_index}')
        #print(f'Grid results: {grid_results}')

    # Print the results
    print('Predictions:')
    print(f'{"":<15} {"Train":>15} {"Test":>15}')
    #print('-'*55)
    print(f'{"MSE:":<15} {yn_train_mse:>15{format_str}} {yn_test_mse:>15{format_str}}')
    print(f'{"RMSE:":<15} {yn_train_rmse:>15{format_str}} {yn_test_rmse:>15{format_str}}')
    print(f'{"MAE:":<15} {yn_train_mae:>15{format_str}} {yn_test_mae:>15{format_str}}')
    print(f'{"R^2 Score:":<15} {train_score:>15{format_str}} {test_score:>15{format_str}}')

    if save:
        # Access to the dataframe for storing results
        global results_df
        # Check if results_df exists in the global scope
        if 'results_df' not in globals():
            # Create results_df if it doesn't exist   
            results_df = pd.DataFrame(columns=['Iteration', 'Train MSE', 'Test MSE', 'Train RMSE', 'Test RMSE',
                'Train MAE', 'Test MAE', 'Train R^2 Score', 'Test R^2 Score', 'Best Grid Mean Score',
                'Best Grid Params', 'Pipeline', 'Note', 'Date'])
            print("\n'results_df' not found in global scope. A new one has been created.")

        # Store results in a dictionary   
        results = {
            'Iteration': iteration,
            'Train MSE': yn_train_mse,
            'Test MSE': yn_test_mse,
            'Train RMSE': yn_train_rmse,
            'Test RMSE': yn_test_rmse,
            'Train MAE': yn_train_mae,
            'Test MAE': yn_test_mae,
            'Train R^2 Score': train_score,
            'Test R^2 Score': test_score,
            'Best Grid Mean Score': best_grid_score,
            'Best Grid Params': best_grid_params,
            'Pipeline': pipe_steps,
            'Note': note,
            'Date': timestamp
        }

        # Convert the dictionary to a dataframe
        df_iteration = pd.DataFrame([results])

        # Append the results dataframe to the existing results dataframe
        results_df = pd.concat([results_df, df_iteration], ignore_index=True)        
        
    # Permutation Feature Importance
    if perm:
        print("\nPermutation Feature Importance:")
        if grid:
            perm_imp_res = calc_fpi(grid, Xn_train, yn_train)
        else:
            perm_imp_res = calc_fpi(pipe, Xn_train, yn_train)

        # Create a Score column
        perm_imp_res['Score'] = perm_imp_res['Score Mean'].apply(lambda x: f"{x:{format_str}}") + " ± " + perm_imp_res['Score Std'].apply(lambda x: f"{x:{format_str}}")

        # Adjust the variable names for better alignment in printout
        perm_imp_res['Variables'] = perm_imp_res['Variables'].str.ljust(25)

        # Create a copy for printing and rename the 'Variables' column header to empty
        print_df = perm_imp_res.copy()
        print_df = print_df.rename(columns={"Variables": ""})

        # Print the DataFrame with only the Variables and the Score column
        print(print_df[['', 'Score']].to_string(index=False))

    if vif:
        all_numeric = not bool(Xn_train.select_dtypes(exclude=[np.number]).shape[1])

        if all_numeric:
            suitable = True
        else:
            # Check if transformers is not empty
            if transformers:
                transformer_list = [transformers] if isinstance(transformers, str) else transformers
                suitable_for_vif = {'ohe', 'ord', 'ohe_drop'}
                if any(t in suitable_for_vif for t in transformer_list):
                    suitable = True
                else:
                    suitable = False
            elif drop:
                suitable = True
            else:
                suitable = False

        if suitable:
            print("\nVariance Inflation Factor:")
            if all_numeric:
                vif_df = Xn_train
            else:
                if transformers is not None:
                    # Create a pipeline with the transformers only
                    #vif_pipe = create_pipeline(transformer_keys=transformers, config=config, X_cat_columns=X_cat_columns, X_num_columns=X_num_columns)
                    if grid:
                        vif_pipe = grid
                        feature_names = grid.best
                    elif pipe:
                        vif_pipe = pipe
                    if debug:
                        print('VIF Pipeline:', vif_pipe)
                        print('VIF Pipeline Parameters:', vif_pipe.get_params())
                    #vif_pipe.fit(Xn_train, yn_train)
                    #feature_names = vif_pipe.get_feature_names_out()
                    #
                    transformed_data = vif_pipe.transform(Xn_train)
                    vif_df = pd.DataFrame(transformed_data, columns=feature_names)
            vif_results = sk_vif(vif_df.columns, vif_df).sort_values(by='VIF', ascending=False)
            vif_results['VIF'] = vif_results['VIF'].apply(lambda x: f'{{:,.{decimal}f}}'.format(x))
            print(vif_results)
        else:
            print("\nVIF calculation skipped. The transformations applied are not suitable for VIF calculation.")

    if plot:
        print('')
        yn_train = yn_train.values.flatten() if isinstance(yn_train, pd.Series) else np.array(yn_train).flatten()
        yn_test = yn_test.values.flatten() if isinstance(yn_test, pd.Series) else np.array(yn_test).flatten()

        yn_train_pred = yn_train_pred.flatten()
        yn_test_pred = yn_test_pred.flatten()

        # Generate residual plots
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        sns.residplot(x=yn_train, y=yn_train_pred, lowess=lowess, scatter_kws={'s': 30, 'edgecolor': 'white'}, line_kws={'color': 'red', 'lw': '1'})
        plt.title(f'Training Residuals - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.residplot(x=yn_test, y=yn_test_pred, lowess=lowess, scatter_kws={'s': 30, 'edgecolor': 'white'}, line_kws={'color': 'red', 'lw': '1'})
        plt.title(f'Test Residuals - Iteration {iteration}')

        plt.show()

        # Generate predicted vs actual plots
        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=yn_train, y=yn_train_pred, s=30, edgecolor='white')
        plt.plot([yn_train.min(), yn_train.max()], [yn_train.min(), yn_train.max()], color='red', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Training Predicted vs. Actual - Iteration {iteration}')

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=yn_test, y=yn_test_pred, s=30, edgecolor='white')
        plt.plot([yn_test.min(), yn_test.max()], [yn_test.min(), yn_test.max()], color='red', linewidth=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Test Predicted vs. Actual - Iteration {iteration}')

        plt.show()

            
    #Calculate coefficients if model supports
    if coef:
        # Extract features and coefficients using the function
        coefficients_df = extract_features_and_coefficients(
            grid.best_estimator_ if grid else pipe, Xn_train, debug=debug
        )

        # Check if there are any non-NaN coefficients
        if coefficients_df['coefficients'].notna().any():
            # Ensure the coefficients are shaped as a 2D numpy array
            coefficients = coefficients_df[['coefficients']].values
        else:
            coefficients = None

        # Debugging information
        if debug:
            print("Coefficients: ", coefficients)
            # Print the number of coefficients and selected rows
            print(f"Number of coefficients: {len(coefficients)}")

        if coefficients is not None:
            print("\nCoefficients:")
            with pd.option_context('display.float_format', lambda x: f'{x:,.{decimal}f}'.replace('-0.00', '0.00')):
                coefficients_df.index = coefficients_df.index + 1
                coefficients_df = coefficients_df.rename(columns={'feature_name': 'Feature', 'coefficients': 'Value'})
                print(coefficients_df)

            if plot:
                coefficients = coefficients.ravel()
                plt.figure(figsize=(12, 3))
                x_values = range(1, len(coefficients) + 1)
                plt.bar(x_values, coefficients)
                plt.xticks(x_values)
                plt.xlabel('Feature')
                plt.ylabel('Value')
                plt.title('Coefficients')
                plt.axhline(y = 0, color = 'black', linestyle='dotted', lw=1)
                plt.show()

    if export:
        filestamp = current_time.strftime('%Y%m%d_%H%M%S')
        filename = f'iteration_{iteration}_model_{filestamp}.joblib'
        dump(best_model, filename)

        # Check if file exists and display a message
        if os.path.exists(filename):
            print(f"\nModel saved successfully as {filename}")
        else:
            print(f"\FAILED to save the model as {filename}")
        
    if grid:
        return best_model, grid_results
    else:
        return best_model
       
def split_outliers(df, columns=None, iqr_multiplier=1.5):
    """
    Splits a DataFrame into two: one with outliers and one without.
    
    Uses the IQR method to determine outliers, based on the provided multiplier.
    
    Params:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): List of columns to consider for outlier detection. If None, all columns are considered.
    - iqr_multiplier (float): The multiplier for the IQR range to determine outliers. Default is 1.5.
    
    Returns:
    - df_no_outliers (pd.DataFrame): DataFrame without outliers.
    - df_outliers (pd.DataFrame): DataFrame with only the outliers.
    """
    
    # If columns parameter is not provided, use all columns in the dataframe
    if columns is None:
        columns = df.columns
    
    # Create an initial mask with all False values (meaning no outliers)
    outlier_mask = pd.Series(False, index=df.index)
    
    # For each specified column, update the outlier mask to mark outliers
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Update mask for outliers in current column
        outlier_mask |= (df[col] < (Q1 - iqr_multiplier * IQR)) | (df[col] > (Q3 + iqr_multiplier * IQR))
    
    # Use the mask to split the data
    df_no_outliers = df[~outlier_mask]
    df_outliers = df[outlier_mask]
    
    return df_no_outliers, df_outliers

def log_transform(df, columns=None):
    """
    Apply a log transformation to specified columns in a DataFrame.
    
    Params:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): List of columns to transform. If None, all columns are considered.
    
    Returns:
    - df_log (pd.DataFrame): DataFrame with the log-transformed columns appended.
    """
    
    df_log = df.copy(deep=True)

    if columns is None:
        columns = df.columns

    log_columns=[]
    
    for col in columns:
        if df[col].min() < 0:
            raise ValueError(f"Column '{col}' has negative values and cannot be log-transformed.")
        
        df_log[col + '_log'] = np.log1p(df[col])
        log_columns.append(col + '_log')
    
    return df_log

def format_columns(val, col_type):
    if col_type == "large":
        return '{:,.0f}'.format(val)
    elif col_type == "small":
        return '{:.2f}'.format(val)
    else:
        return val

def format_df(df, large_num_cols=None, small_num_cols=None):
    """
    Returns a formatted DataFrame.
    
    Parameters:
    - df: the DataFrame to format.
    - large_num_cols: list of columns with large numbers to be formatted.
    - small_num_cols: list of columns with small numbers to be formatted.
    
    Returns:
    - formatted_df: DataFrame with specified columns formatted.
    """
    
    formatted_df = df.copy()

    if large_num_cols:
        for col in large_num_cols:
            formatted_df[col] = formatted_df[col].apply(lambda x: format_columns(x, "large"))

    if small_num_cols:
        for col in small_num_cols:
            formatted_df[col] = formatted_df[col].apply(lambda x: format_columns(x, "small"))
        
    return formatted_df

def plot_residuals(results, rotation=45):
    """
    Plot the residuals of an ARIMA model along with their histogram, autocorrelation function (ACF), 
    and partial autocorrelation function (PACF). The residuals are plotted with lines indicating 
    standard deviations from the mean.

    Parameters:
    - results (object): The result object typically obtained after fitting an ARIMA model.
      This object should have a `resid` attribute containing the residuals.

    Outputs:
    A 2x2 grid of plots displayed using matplotlib:
    - Top-left: Residuals with standard deviation lines.
    - Top-right: Histogram of residuals with vertical standard deviation lines.
    - Bottom-left: Autocorrelation function of residuals.
    - Bottom-right: Partial autocorrelation function of residuals.

    Note:
    Ensure that the necessary libraries (like matplotlib, statsmodels) are imported before using this function.
    """
    residuals = results.resid
    std_dev = residuals.std()
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    # Plot residuals
    ax[0, 0].axhline(y=0, color='lightgrey', linestyle='-', lw=1)
    ax[0, 0].axhline(y=std_dev, color='red', linestyle='--', lw=1, label=f'1 STD (±{std_dev:.2f})')
    ax[0, 0].axhline(y=2*std_dev, color='red', linestyle=':', lw=1, label=f'2 STD (±{2*std_dev:.2f})')
    ax[0, 0].axhline(y=-std_dev, color='red', linestyle='--', lw=1)
    ax[0, 0].axhline(y=2*-std_dev, color='red', linestyle=':', lw=1)
    ax[0, 0].plot(residuals, label='Residuals')
    ax[0, 0].tick_params(axis='x', rotation=rotation)
    ax[0, 0].set_title('Residuals from ARIMA Model')
    ax[0, 0].legend()

    # Plot histogram of residuals
    ax[0, 1].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    ax[0, 1].axvline(x=std_dev, color='red', linestyle='--', lw=1, label=f'1 STD (±{std_dev:.2f})')
    ax[0, 1].axvline(x=2*std_dev, color='red', linestyle=':', lw=1, label=f'2 STD (±{2*std_dev:.2f})')
    ax[0, 1].axvline(x=-std_dev, color='red', linestyle='--', lw=1)
    ax[0, 1].axvline(x=2*-std_dev, color='red', linestyle=':', lw=1)
    ax[0, 1].set_title("Histogram of Residuals")
    ax[0, 1].set_xlabel("Residual Value")
    ax[0, 1].set_ylabel("Frequency")
    ax[0, 1].legend()

    # Plot ACF of residuals
    plot_acf(residuals, lags=40, ax=ax[1, 0])
    ax[1, 0].set_title("ACF of Residuals")

    # Plot PACF of residuals
    plot_pacf(residuals, lags=40, ax=ax[1, 1])
    ax[1, 1].set_title("PACF of Residuals")

    plt.tight_layout()
    plt.show()




