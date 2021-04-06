import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from models import Richards, RichardsModified, LogisticSymmetric


class Plate:
    """Class representing a plate experiment."""
    def __init__(self, f, widefile, blank_name='Blank', ID=None):
        """
        Parameters
        -----------
        f <str>: path to template CSV file
        widefile <str>: path to wideform CSV file
        blank_name <str>: Name of blank wells
        """
        print('Initializing ... ', end='')
        self.platefile = f
        self.widefile = widefile
        
        if ID is None:
            self.ID = os.path.basename(self.widefile).split('.')[0]
        else:
            self.ID = ID
        
        self.blank_name = blank_name
        self.plate = self._read_file()
        self.wide = self._read_wide()
        self.long = self._to_longform()
        self.combined = self._combine_replicates()
        
        self.long['ID'] = self.ID
        self.combined['ID'] = self.ID
        print('Done', end='\n')
        
    def __add__(self, other):
        """Overload the + operator so that two Plates can easily be concatenated."""
        added_long = pd.concat([self.long, other.long], axis=0)
        added_comb = pd.concat([self.combined, other.combined], axis=0)
        return added_long, added_comb
        
    def _read_file(self):
        """Load plate template to DataFrame."""
        return pd.read_csv(self.platefile, index_col=0)
    
    def _read_wide(self):
        """Load wideform CSV file to DataFrame"""
        return pd.read_csv(self.widefile, delimiter=';')
    
    def _to_numeric(self):
        """Convert string group names to numbers for visualization."""
        groups = self.groups()
        numeric = self.plate.copy(deep=True)
        for i, g in enumerate(groups):
            if g is None:
                continue
            numeric.replace(g, i, inplace=True)
        return numeric
    
    def groups(self):
        """Get unique groups in plate."""
        groups = []
        for row in self.plate.values.tolist():
            for g in set(row):
                if not isinstance(g, float):
                    groups.append(g)
        return set(groups)
    
    def show(self, savefig=False, filename=None):
        """Visualize plate layout as a heatmap."""
        xlabels = list(range(1, 13))
        ylabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        cmap = sns.color_palette('cubehelix', as_cmap=True)
        
        fig, ax = plt.subplots(figsize=(15, 5))
        FS = 16
        ax = sns.heatmap(
                self._to_numeric(),
                ax=ax,
                annot=self.plate,
                fmt='',
                annot_kws={'fontsize': 10},
                linewidths=2,
                linecolor='black',
                cbar=False,
                square=False,
                cmap=cmap,
                xticklabels=xlabels,
                yticklabels=ylabels)

        ax.tick_params(labelsize=FS)
        ax.tick_params(axis='y', rotation=0)
        ax.set_title('Plate Layout', fontsize=FS)
        fig.tight_layout()
        
        if savefig:
            if filename is None:
                filename = f'{self.ID}_PlateDesign.png'
            plt.savefig(filename)
        
    def tolist(self):
        """Convert plate DataFrame to list of lists."""
        return self.plate.values.tolist()
    
    def get_xy(self, group=None, log=True):
        """Return the x and y values for the passed group."""
        sub = self.combined.loc[self.combined.Group == group]
        x = sub.Time_hours
        y = sub.Mean if not log else sub.logMean
        return x, y
    
    def _to_longform(self):
        """Melt wideform to longform."""
        wells = self.wide.columns[3:]
        melted = pd.melt(self.wide, 
                         id_vars=['Time_hours', 'Temp'],
                         var_name='Well',
                         value_vars=wells,
                         value_name='Absorption')
        melted['Group'] = melted.Well.map(lambda well: self._get_plate_group(well))
        return melted
    
    def _get_plate_group(self, well):
        """Get well group name from well ID.
        
        Parameters
        -----------
        well <str>: Well ID (e.g. D9)"""
        coord_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        row = int(coord_map[well[0].upper()])
        col = int(well[1:]) - 1
        return str(self.plate.iloc[row, col])
    
    def add_metadata(self, key, val):
        """Add column to long and combined DataFrames.
        
        Paramters
        ----------
        key <str>: Name of column to add
        val <str>: Column value to add"""
        self.long[key] = val
        self.combined[key] = val
        
    def remove_metadata(self, key):
        """Remove column in-place.
        
        Parameters
        -----------
        key <str>: Name of column to remove."""
        self.long.drop(key, axis=1, inplace=True)
        self.combined.drop(key, axis=1, inplace=True)
    
    def show_all_curves(self, col_wrap=10, savefig=False, filename=None):
        """Plot all growth curves in a Seaborn relplot.
        
        Parameters
        -----------
        col_wrap <int>: Number of columns in each row
        savefig <bool>: Write figure to a PNG file
        filename <str>: Filename of figure PNG file
        """
        grid = sns.relplot(data=self.long,
                           x='Time_hours',
                           y='Absorption',
                           hue='Group',
                           kind='scatter',
                           col='Well',
                           col_wrap=col_wrap,
                           legend=False,
                           s=10,
                           height=2,
                           palette='bright')
        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f'{self.ID}_AllCurves.png'
            plt.savefig(filename)
        return grid
    
    def _combine_replicates(self):
        """Combine same group names and calculate Mean, StDev
        and log transforms."""
        final = pd.DataFrame()
        for group in self.long.Group.unique():
            sub = self.long.loc[self.long.Group == group]
            g = sub.groupby(by=['Time_hours']).describe()
            n = g['Absorption', 'count']
            mean = g['Absorption', 'mean']
            std = g['Absorption', 'std']
            temp = g['Temp', 'mean']
            temp_std = g['Temp', 'std']
            
            tmp = pd.DataFrame([temp, temp_std, n, mean, std]).transpose()
            tmp.columns = ['Temp_Mean','Temp_StDev', 'n', 'Mean', 'StDev']
            tmp['Group'] = group
            final = pd.concat([final, tmp], axis=0)
            
        final['logMean'] = np.log2(final.Mean)
        final['logStDevUpper'] = np.log2(final.Mean + final.StDev)
        final['logStDevLower'] = np.log2(final.Mean - final.StDev)
        return final.reset_index()
    
    def show_combined_curves(self, col_wrap=5, log=True, savefig=False, filename=None):
        """Plot combined growth curves, overlayed with stdev ranges and blanks.
        
        Parameters
        -----------
        col_wrap <int>: 
        log <bool>: 
        savefig <bool>: 
        filename <str>: 
        """
        grid = sns.relplot(data=self.combined,
                           x='Time_hours',
                           y='Mean' if not log else 'logMean',
                           kind='scatter',
                           col='Group',
                           col_wrap=col_wrap,
                           legend=False,
                           s=40,
                           marker='.',
                           height=3,
                           color='black')
        
        for group, ax in grid.axes_dict.items():
            sub = self.combined.loc[self.combined.Group == group]
            x = sub.Time_hours
            if log:
                plus = sub.logStDevUpper
                minus = sub.logStDevLower
            else:
                plus = sub.Mean + sub.StDev
                minus = sub.Mean - sub.StDev
            
            ax.fill_between(x, plus, minus, alpha=0.25, color='gray')
            blanks = self.combined.loc[self.combined.Group == self.blank_name]
            sns.scatterplot(data=blanks,
                            x='Time_hours',
                            y='Mean' if not log else 'logMean',
                            ax=ax,
                            s=15,
                            color='crimson',
                            marker='.')
        if savefig:
            if filename is None:
                filename = f'{self.ID}_CombinedCurves.png'
            plt.savefig(filename)
        return grid
        
    def fit_model(self, modelname='richardsmodified', group=None):
        """Fit model to growth curves on log transformed data.
        
        Parameters
        -----------
        modelname <str>: name of model to use
        group <str>: fit for individual group
        """
        # Set the model to use
        assert modelname.lower() in ['richards', 'richardsmodified', 'logisticsymmetric'], 'Model not available!'
        if modelname.lower() == 'richards':
            model = Richards
        elif modelname.lower() == 'richardsmodified':
            model = RichardsModified
        elif modelname.lower() == 'logisticsymmetric':
            model = LogisticSymmetric
        
        if group is not None:
            # Single group mode
            x, y = self.get_xy(group=group, log=True)
            m = model(x, y)
            m.fit()
            return m
        else:
            models = {g: None for g in self.groups()}
            # Batch mode
            for group in self.groups():
                x, y = self.get_xy(group=group, log=True)
                m = model(x, y)
                try:
                    m.fit()
                    models[group] = m
                except:
                    print(f'Could not find optimized parameters for: ', group)
            return models
        
    def show_fitted_curves(self, col_wrap=5, modelname='richardsmodified', savefig=False, filename=None):
        """Perform curve fitting on all groups.
        
        Parameters
        -----------
        col_wrap <int>: Number of columns in each row
        modelname <str>: Which function to use for fitting
        """
        grid = sns.relplot(data=self.combined,
                           x='Time_hours',
                           y='logMean',
                           kind='scatter',
                           col='Group',
                           col_wrap=col_wrap,
                           legend=False,
                           s=40,
                           marker='.',
                           height=3,
                           color='black')
        
        models = self.fit_model(modelname=modelname)
        for group, ax in grid.axes_dict.items():
            m = models[group]
            if m is None:
                continue
            x, _ = self.get_xy(group=group, log=True)
            y = m.func(x, *m.popt)
            sns.lineplot(ax=ax, x=x, y=y, color='red')
            
            a = min(x)
            b = max(x)
            
            verts = [(a, min(y)), *zip(x, y), (b, min(y))]
            poly = Polygon(verts, facecolor='crimson', alpha=0.1)
            ax.add_patch(poly)
        
        if savefig:
            if filename is None:
                filename = f'{self.ID}_FittedCurves.png'
            plt.savefig(filename)
            
        return models, grid
            
    def show_fitted_curve(self, modelname='richardsmodified', group=None, savefig=False, filename=None):
        """Perform curve fitting on a single group.
        
        Parameters
        -----------
        modelname <str>: Which function to use for fitting
        group <str>: Which group on which to perform fitting
        """
        x, y = self.get_xy(group=group, log=True)
        model = self.fit_model(modelname=modelname, group=group)
        yfit = model.func(x, *model.popt)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(ax=ax, x=x, y=y, s=40, color='black')
        sns.lineplot(ax=ax, x=x, y=yfit, color='red')
        
        if savefig:
            if filename is None:
                filename = f'{self.ID}_FittedCurve_{group}.png'
            plt.savefig(filename)
        
        return model, ax
    
    def _collect_model_params(self, models):
        """Given the models, get optimized parameters and return as a DataFrame."""
        params = []
        for g, m in models.items():
            if m is None:
                continue
            params.append((g, *m.popt))
        return params
    
    def analyze(self, modelname='richardsmodified', savefiles=False):
        """Perform curve fitting on all groups, and display all plots."""
        print('Fitting models')
        models, grid = self.show_fitted_curves(modelname=modelname, savefig=savefiles)
        if modelname.lower() == 'richards':
            columns = ['Group', 'U', 'L', 'k', 'v', 'x0']
        elif modelname.lower() == 'richardsmodified':
            columns = ['Group', 'U', 'L', 'k1', 'k2', 'v', 'x0']
        elif modelname.lower() == 'logisticsymmetric':
            columns = ['Group', 'U', 'L', 'k', 'x0', 'offset']
            
        params = pd.DataFrame(self._collect_model_params(models), columns=columns)
        
        print('Computing areas and rates')
        areas = []
        for g, m in models.items():
            if m is None:
                continue
            A = m.area_under_curve()
            r = m.max_growth_rate()
            areas.append((g, A, r))
        df = pd.DataFrame(areas, columns=['Group', 'Area', 'Rate'])
        
        self.show(savefig=savefiles)
        self.show_temperatures(savefig=savefiles)
        self.show_all_curves(savefig=savefiles)
        self.show_combined_curves(savefig=savefiles)
        
        print(params)
        print(df)
        
        if savefiles:
            params.to_csv(f'{self.ID}_OptimizedModelParameters.csv')
            df.to_csv(f'{self.ID}_AreasAndRates.csv')
            
    def show_temperatures(self, savefig=False, filename=None):
        """Visualize the temperature distribution throughout the experiment."""
        x = self.combined.Time_hours
        y = self.combined.Temp_Mean
        ax = sns.scatterplot(x=x, y=y, color='black', marker='.', s=40)
        ax.set_title('Temperatures')
        
        if savefig:
            if filename is None:
                filename = f'{self.ID}_Temperatures.png'
            plt.savefig(filename)
        return ax
