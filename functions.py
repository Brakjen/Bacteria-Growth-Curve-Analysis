import pandas as pd
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Helper dictionary for switching between a character and a number
coordMap = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}


def getPlateGroup(well, template):
    """Based on the well ID, return the generic group
    from the template"""
    row = int(coordMap[well[0].upper()]) - 1
    col = int(well[1:]) - 1
    return str(template[row][col])


def convertTemplateToNumeric(template):
    """Convert the template to a a form with generic group names."""
    df = pd.DataFrame(template)
    groups = set([col for row in template for col in row])
    for i, g in enumerate(groups):
        if g is None:
            continue
        df.replace(g, str(i), inplace=True)
    return df.values.tolist()

def toLongForm(df, template):
    """Convert the passed wide-form dataframe to long-form. Uses the template
    to retrieve the group ID, which can be used for coloring coding in plots,
    and for knowing how to compute averages and STDs correctly.

    Assumptions:
        1. id_var is called 'Time_hours'
        2. value_vars start at the fourth column
    """
    wells = df.columns[3:]
    melted = pd.melt(df, id_vars='Time_hours', value_vars=wells, value_name='Absorption', var_name='Well')
    melted['Row'] = melted['Well'].str[0]
    melted['Column'] = melted['Well'].str[1:]
    melted['Group'] = melted['Well'].map(lambda well: getPlateGroup(well, template))
    return melted


def graphTemplate(template, savefig=False, show=True, annot_size=10, filename='Fig_Template.png'):
    """Read the manually generated template, and make a heatmap
    visualization of the plate layout.
    
    <savefig> controls whether to save figure to png file or not.
    
    <show> controls whether to display the figure."""
    xlabels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ylabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    df = pd.DataFrame(convertTemplateToNumeric(template)).astype(float)
    cmap = sns.color_palette("cubehelix", as_cmap=True)

    fig, ax = plt.subplots(figsize=(15, 5))
    FS = 16
    ax = sns.heatmap(df,
                ax=ax,
                annot=template,
                fmt='',
                annot_kws={'fontsize': annot_size},
                linewidths=2,
                linecolor='black',
                cbar=False,
                square=False,
                cmap=cmap,
                xticklabels=xlabels,
                yticklabels=ylabels)

    ax.tick_params(labelsize=FS)
    ax.tick_params(axis='y', rotation=0)
    ax.set_title('96-Well Layout', fontsize=FS)
    fig.tight_layout()
    if savefig:
        plt.savefig(filename)
    if show:
        plt.show()
    return ax


def graphAllGrowthCurves(df, col_wrap=10, savefig=False, show=True, filename='Fig_AllCurves.png'):
    """Plot all growth curves in a grid, color coded by group ID.

    <col_wrap> controls the number of columns in each row. Adjust to each
    individual plate design.
    
    <savefig> controls whether to save figure to png file or not.
    
    <show> controls whether to display the figure."""
    grid = sns.relplot(data=df,
                       x='Time_hours',
                       y='Absorption',
                       hue='Group',
                       kind='scatter',
                       col='Well', col_wrap=col_wrap,
                       legend=False,
                       s=10,
                       height=2,
                       palette='bright')
    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
    if show:
        plt.show()
    return grid


def graphCombinedGrowthCurves(df, col_wrap=5, augment=True, log=True, savefig=False, show=True, filename='Fig_CombinedCurves.png'):
    """Plot combined growth curves in a grid.
    Note that blank sample wells must be called 'Blank' in order for
    this function to be able to identify them.

    <col_wrap> controls the number of columns in each row. Adjust to each
    individual plate design.

    <augment> controls whether to add error bands and blank comparison

    <log> controls whether to use log transformed data or not.
    
    <savefig> controls whether to save figure to png file or not.
    
    <show> controls whether to display the figure."""
    sns.set_theme(style='ticks')
    grid = sns.relplot(data=df,
                       x='Time_hours',
                       y='Mean' if not log else 'logMean',
                       kind='scatter',
                       col='Group', col_wrap=col_wrap,
                       legend=False,
                       s=40,
                       marker='.',
                       height=3,
                       color='black',
                       zorder=99)

    # Iterate over the individual axes in the grid to add individual modifications
    if augment:
        for group, ax in grid.axes_dict.items():
            # Add error bands equal to +- 1*StDev
            sub = df.loc[df.Group == group]
            x = sub.Time_hours
            if log:
                plus = sub.logStDevUpper
                minus = sub.logStDevLower
            else:
                plus = sub.Mean + sub.StDev
                minus = sub.Mean - sub.StDev

            ax.fill_between(x, plus, minus, alpha=0.2, color='skyblue')

            # Add blank growth curves to all groups for a comparison
            blanks = df.loc[df.Group == 'Blank']
            sns.scatterplot(data=blanks, x='Time_hours', y='Mean' if not log else 'logMean', ax=ax, s=15, color='crimson', marker='.')

    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
    if show:
        plt.show()
    return grid


def combineReplicates(df):
    """Takes the long-form df as input, which contains
    information about the group IDs."""
    final = pd.DataFrame()
    for group in df.Group.unique().tolist():
        sub = df.loc[df.Group == group].reset_index()
        g = sub.groupby(by=['Time_hours']).describe()
        n = g['Absorption', 'count']
        mean = g['Absorption', 'mean']
        std = g['Absorption', 'std']

        tmp = pd.DataFrame([n, mean, std]).transpose()
        tmp.columns = ['n', 'Mean', 'StDev']
        tmp['Group'] = group
        final = pd.concat([final, tmp], axis=0)

    # Add log transformed columns
    final['logMean'] = np.log2(final.Mean)
    final['logStDevUpper'] = np.log2(final.Mean + final.StDev)
    final['logStDevLower'] = np.log2(final.Mean - final.StDev)
    return final.reset_index()


def richardsCurveFitting(x, y, initial, bounds, maxfev=1000):
    """Perform a generalized, logistic curve fit (Also known as
    Richard's Curve). 
    
    Returns
    -------
    func: function for generating the y values
    popt: optimized curve parameters
    
    Parameters
    ----------
    x: x axis data
    y: y axis data
    initial <list>: List of initial values for the curve parameters (below)
    bounds <>: For constrained optimization of curve parameters. The parameters
               will be within the passed lower and upper bounds
               
    Richard's Curve Parameters
    --------------------------
    U: Upper asymptote
    L: Lower asymptote
    k: Maximum growth rate (rate at inflextion point)
    v: Controls the assymetry of the curve
       v > 1       => inflection point closer to upper asymptote
       0 < v < 1   => inclection point closer to lower asymptote
       
    Reference
    ---------
    https://en.wikipedia.org/wiki/Generalised_logistic_function
    """
    def func(x, U, L, k, v):
        return L + ( (U - L) / (1 + 0.5 * np.exp(-k*x))**(1/v) )
    
    popt, pcov = curve_fit(func, x, y, p0=initial, bounds=bounds)
    return func, popt


def graphCurveFitting(df, col_wrap=5, savefig=False, show=True, filename='Fig_CurveFitting.png'):
    """Plot all the combined growth curves, overlayed with the
    fitted Richard's Curve."""
    sns.set_theme(style='ticks')
    grid = sns.relplot(data=df,
                       x='Time_hours',
                       y='logMean',
                       kind='scatter',
                       col='Group', col_wrap=col_wrap,
                       legend=False,
                       s=40,
                       marker='.',
                       height=3,
                       color='black',
                       zorder=0)

    # Iterate over the individual axes in the grid to add individual modifications
    for group, ax in grid.axes_dict.items():
        sub = df.loc[df.Group == group]
        x = sub['Time_hours']
        y = sub['logMean']
        init = [max(y), min(y), 0.5, 0.5]
        lower = (max(y) - 0.5, min(y) - 0.05, 0.01, 0.01)
        upper = (max(y) + 0.5, min(y) + 0.05, 5.0, 10.0)
        
        func, popt = richardsCurveFitting(x, y, initial=init, bounds=(lower, upper))
        sns.lineplot(ax=ax, x=x, y=func(x, *popt), color='red')
        print('Done', end='\n')

    plt.tight_layout()
    if savefig:
        plt.savefig(filename)
    if show:
        plt.show()
    return grid