import pandas as pd
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Helper dictionary for switching between a character and a number
coordMap = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}


def getPlateGroup(well, template):
    """Based on the well ID, return the generic group
    from the template"""
    row = int(coordMap[well[0].upper()]) - 1
    col = int(well[1:]) - 1
    return str(template[row][col])


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


def graphTemplate(template, savefig=False):
    """Read the manually generated template, and make a heatmap
    visualization of the plate layout."""
    df = pd.DataFrame(template).astype(float)
    xlabels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ylabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cmap = sns.color_palette("cubehelix", as_cmap=True)

    fig, ax = plt.subplots()
    FS = 16
    sns.heatmap(df,
                ax=ax,
                annot=True,
                linewidths=2,
                linecolor='black',
                cbar=False,
                square=True,
                cmap=cmap,
                xticklabels=xlabels,
                yticklabels=ylabels)

    ax.tick_params(labelsize=FS)
    ax.tick_params(axis='y', rotation=0)
    ax.set_title('96-Well Layout', fontsize=FS)
    fig.tight_layout()
    if savefig:
        plt.savefig('FigTemplate.png')
    else:
        plt.show()


def graphAllGrowthCurves(df, col_wrap=10, savefig=False):
    """Plot all growth curves in a grid, color coded by group ID.

    col_wrap controls the number of columns in each row. Adjust to each
    individual plate design."""
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
        plt.savefig('FigAllGrowthCurves.png')
    else:
        plt.show()


def graphCombinedGrowthCurves(df, col_wrap=5, augment=True, log=True, savefig=False):
    """Plot combined growth curves in a grid.

    <col_wrap> controls the number of columns in each row. Adjust to each
    individual plate design.

    <augment> controls whether to add error bands and blank comparison

    <log> controls whether to use log transformed data or not."""
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
            # Group '0' means blank
            blanks = df.loc[df.Group == '0']
            sns.scatterplot(data=blanks, x='Time_hours', y='Mean' if not log else 'logMean', ax=ax, s=15, color='crimson', marker='.')

    plt.tight_layout()
    if savefig:
        plt.savefig(f'FigCombinedGrowthCurves_log={log}_augment={augment}.png')
    else:
        plt.show()


def combineReplicates(df):
    """Takes the long-form df as input, which contains
    information about the group IDs."""
    final = pd.DataFrame()
    for group in sorted(df.Group.unique().tolist(), key=lambda g: int(g)):
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
