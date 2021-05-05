import sys
import os
from datetime import date
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from copy import deepcopy
import traceback

from models import BaseModel, LogisticZero, Logistic2k, Logistic2kZero, Richards, RichardsZero, Gompertz, GompertzZero


class Plate:
    """Class representing a plate experiment."""
    def __init__(self, template=None, data=None, blank_name='Blank', ID=None, delimiter=','):
        """
        Parameters
        -----------
        template <str>: path to template CSV file
        data <str>: path to wideform CSV file
        blank_name <str>: Name of blank wells
        ID <str>: Unique identifier for this plate object
        delimiter <str>: which delimiter used in CSV files (Default: ,)
        """
        self.platefile = template
        self.datafile = data
        self.ID = ID
        self.defaultmodel = 'Logistic2kZero'
        
        self.blank_name = blank_name
        self.delimiter = delimiter
        self.current_models = None
        
        if all([self.platefile is not None, self.datafile is not None]):
            self.plate = self._read_template()
            self.raw, self.long, self.combined = self._read_data()
        else:
            self.plate = pd.DataFrame()
            self.raw = pd.DataFrame()
            self.long = pd.DataFrame()
            self.combined = pd.DataFrame()
            
        self.long['ID'] = self.ID
        self.combined['ID'] = self.ID
        
    def __add__(self, other):
        """Overload the + operator so that two Plates can easily be concatenated."""
        long = pd.concat([self.long, other.long], axis=0)
        comb = pd.concat([self.combined, other.combined], axis=0)
        
        p = Plate(ID=f'{self.ID}+{other.ID}')
        p.long = long
        p.combined = comb
        return p
        
    def _read_template(self):
        """Load plate template to DataFrame."""
        return pd.read_csv(self.platefile, delimiter=self.delimiter)
    
    def _read_data(self):
        """..."""
        offset = 3
        absorptions = pd.read_csv(self.datafile, delimiter=self.delimiter).sort_values(by='Time_hours', ascending=True)
        #ratios = absorptions.apply(lambda col: col/col.iloc[0] if absorptions.columns.get_loc(col.name) > 2 else col)
        ratios = absorptions.apply(lambda col: col/col.min() if absorptions.columns.get_loc(col.name) > 2 else col)

        absmelt = pd.melt(absorptions, id_vars=['Time_hours', 'Temp'], value_vars=absorptions.columns[offset:], var_name='Well', value_name='Absorption')
        ratmelt = pd.melt(ratios, id_vars=['Time_hours', 'Temp'], value_vars=ratios.columns[offset:], var_name='Well', value_name='Ratio')
        
        long = absmelt.merge(ratmelt, on=['Time_hours', 'Temp', 'Well'])
        long['Group'] = long.Well.map(lambda w: self._get_plate_group(w))
        
        mean = long.groupby(['Group', 'Time_hours']).mean().reset_index().rename(columns={'Temp': 'MeanTemp', 'Absorption': 'MeanAbs', 'Ratio': 'MeanRatio'})
        std = long.groupby(['Group', 'Time_hours']).std().reset_index().rename(columns={'Temp': 'StDevTemp', 'Absorption': 'StDevAbs', 'Ratio': 'StDevRatio'})
        final = mean.merge(std, on=['Group', 'Time_hours'])
        
        final['logAbs'] = np.log2(final.MeanAbs)
        final['logRatio'] = np.log2(final.MeanRatio)
        
        final['logStDevAbsUpper'] = np.log2(final.MeanAbs + final.StDevAbs)
        final['logStDevAbsLower'] = np.log2(final.MeanAbs - final.StDevAbs)
        
        final['logStDevRatioUpper'] = np.log2(final.MeanRatio + final.StDevRatio)
        final['logStDevRatioLower'] = np.log2(final.MeanRatio - final.StDevRatio)
        
        return absorptions, long, final
    
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
    
    def show(self, savefig=False, filename=None, outputdir='.'):
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

        if savefig:
            if filename is None:
                filename = f'{self.ID}_PlateDesign.png'
            fig.tight_layout()
            fig.savefig(os.path.join(outputdir, filename))
            plt.close(fig)

        return fig, ax
        
    def tolist(self):
        """Convert plate DataFrame to list of lists."""
        return self.plate.values.tolist()
    
    def get_xy(self, group=None):
        """Return the x and y values for the passed group."""
        sub = self.combined.loc[self.combined.Group == group]
        x = sub.Time_hours
        y = sub.logRatio
        return x, y
    
    def _get_plate_group(self, well):
        """Get well group name from well ID.
        
        Parameters
        -----------
        well <str>: Well ID (e.g. D9)
        """
        coord_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        row = int(coord_map[well[0].upper()])
        col = int(well[1:]) - 1
        return str(self.plate.iloc[row, col])
    
    def add_metadata(self, key, val):
        """Add column to long and combined DataFrames.
        
        Paramters
        ----------
        key <str>: Name of column to add
        val <str>: Column value to add
        """
        self.long[key] = val
        self.combined[key] = val
        
    def remove_metadata(self, key):
        """Remove column in-place.
        
        Parameters
        -----------
        key <str>: Name of column to remove."""
        self.long.drop(key, axis=1, inplace=True)
        self.combined.drop(key, axis=1, inplace=True)
    
    def show_all_curves(self, col_wrap=10, savefig=False, filename=None, sharey=True, temp=True, outputdir='.'):
        """Plot all growth curves in a Seaborn relplot.
        
        Parameters
        -----------
        col_wrap <int>: Number of columns in each row
        savefig <bool>: Write figure to a PNG file
        filename <str>: Filename of figure PNG file
        """
        grid = sns.relplot(data=self.long,
                           x='Time_hours',
                           y='Ratio',
                           hue='Group',
                           kind='scatter',
                           col='Well',
                           col_wrap=col_wrap,
                           legend=False,
                           s=10,
                           height=2,
                           palette='bright',
                           facet_kws={'sharey': sharey})

        for well, ax in grid.axes_dict.items():
            ax.set_title(well)

        if temp:
            m, s = self.mean_temperature()
            plt.suptitle(f'Temperature: {m:.2f} $\pm$ {s:.1f}', fontsize=20)
        plt.tight_layout()

        if savefig:
            if filename is None:
                filename = f'{self.ID}_AllCurves.png'
            plt.savefig(os.path.join(outputdir, filename))
        return grid

    def mean_temperature(self):
        mean = self.combined.MeanTemp.mean()
        std = self.combined.MeanTemp.std()
        return mean, std
    
    def show_combined_curves(self, col_wrap=5, savefig=False, filename=None, sharey=True, temp=True, outputdir='.'):
        """Plot combined growth curves, overlayed with stdev ranges and blanks.
        
        Parameters
        -----------
        col_wrap <int>: 
        savefig <bool>: 
        filename <str>: 
        """
        grid = sns.relplot(data=self.combined,
                           x='Time_hours',
                           y='logRatio',
                           kind='scatter',
                           col='Group',
                           col_wrap=col_wrap,
                           legend=False,
                           s=40,
                           marker='.',
                           height=3,
                           color='black',
                           facet_kws={'sharey': sharey})
        
        for group, ax in grid.axes_dict.items():
            sub = self.combined.loc[self.combined.Group == group]
            x = sub.Time_hours
            plus = sub.logStDevRatioUpper
            minus = sub.logStDevRatioLower
            
            ax.fill_between(x, plus, minus, alpha=0.25, color='gray')
            blanks = self.combined.loc[self.combined.Group == self.blank_name]
            sns.scatterplot(data=blanks,
                            x='Time_hours',
                            y='logRatio',
                            ax=ax,
                            s=15,
                            color='crimson',
                            marker='.')

            ax.set_title(group)

        if temp:
            m, s = self.mean_temperature()
            plt.suptitle(f'Temperature: {m:.2f} $\pm$ {s:.1f}', fontsize=20)

        plt.tight_layout()
        if savefig:
            if filename is None:
                filename = f'{self.ID}_CombinedCurves.png'
            plt.savefig(os.path.join(outputdir, filename))
        return grid
        
    def fit_model(self, modelname=None, group=None, maxfev=None):
        """Fit model to growth curves on log transformed data.
        
        Parameters
        -----------
        modelname <str>: name of model to use
        group <str>: fit for individual group
        """
        # Set the model to use
        if modelname.lower() == 'richards':
            model = Richards
        elif modelname.lower() == 'richardszero':
            model = RichardsZero
        elif modelname.lower() == 'logisticzero':
            model = LogisticZero
        elif modelname.lower() == 'gompertz':
            model = Gompertz
        elif modelname.lower() == 'gompertzzero':
            model = GompertzZero
        elif modelname.lower() == 'logistic2k':
            model = Logistic2k
        elif modelname.lower() == 'logistic2kzero':
            model = Logistic2kZero
        else:
            model = Logistic2kZero
        
        if group is not None:
            # Single group mode
            x, y = self.get_xy(group=group)
            m = model(x, y, maxfev=maxfev)
            m.fit()
            return m
        else:
            # Batch mode
            models = {g: None for g in self.groups()}
            for group in self.groups():
                x, y = self.get_xy(group=group)
                m = model(x=x, y=y, maxfev=maxfev)
                try:
                    m.fit()
                    models[group] = m
                except Exception:
                    print(f'Could not find optimized parameters for: ', group)
                    print(traceback.format_exc())
            self.current_models = models
            return models
        
    def show_fitted_curves(self, col_wrap=5, modelname='richards2kzero', savefig=False, filename=None, sharey=True, temp=True, outputdir='.', a=None, b=None):
        """Perform curve fitting on all groups.
        
        Parameters
        -----------
        col_wrap <int>: Number of columns in each row
        modelname <str>: Which function to use for fitting
        """
        grid = sns.relplot(data=self.combined,
                           x='Time_hours',
                           y='logRatio',
                           kind='scatter',
                           col='Group',
                           col_wrap=col_wrap,
                           legend=False,
                           s=40,
                           marker='.',
                           height=3,
                           color='black',
                           facet_kws={'sharey': sharey})

        if self.current_models is None:
            models = self.fit_model(modelname=modelname)
        else:
            models = self.current_models

        for group, ax in grid.axes_dict.items():
            m = models[group]
            if m is None:
                continue
            x, _ = self.get_xy(group=group)
            y = m.func(x, *m.popt)
            sns.lineplot(ax=ax, x=x, y=y, color='red')
            
            a = min(x) if a is None else a
            b = max(x) if b is None else b
            
            #verts = [(a, m.func(a, *m.popt)), *zip(x, y), (b, m.func(b, *m.popt))]
            #poly = Polygon(verts, facecolor='crimson', alpha=0.1)
            #ax.add_patch(poly)

            xs = np.linspace(a, b, 200)
            ax.fill_between(xs, m.func(xs, *m.popt), color='crimson', alpha=0.1)

            ax.set_title(group)

        if temp:
            m, s = self.mean_temperature()
            plt.suptitle(f'Temperature: {m:.2f} $\pm$ {s:.1f}', fontsize=20)
        plt.tight_layout()
        
        if savefig:
            if filename is None:
                filename = f'{self.ID}_FittedCurves.png'
            plt.savefig(os.path.join(outputdir, filename))
            
        return grid, models
            
    def show_fitted_curve(self, modelname='richards2k', group=None, savefig=False, filename=None, outputdir='.'):
        """Perform curve fitting on a single group.
        
        Parameters
        -----------
        modelname <str>: Which function to use for fitting
        group <str>: Which group on which to perform fitting
        """
        x, y = self.get_xy(group=group)
        model = self.fit_model(modelname=modelname, group=group)
        yfit = model.func(x, *model.popt)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(ax=ax, x=x, y=y, s=40, color='black')
        sns.lineplot(ax=ax, x=x, y=yfit, color='red')

        if savefig:
            if filename is None:
                filename = f'{self.ID}_FittedCurve_{group}.png'
            fig.tight_layout()
            fig.savefig(os.path.join(outputdir, filename))
            plt.close(fig)
        
        return fig, ax, model

    def clear_current_model(self):
        self.current_models = None
    
    def _collect_model_params(self, models):
        """Given the models, get optimized parameters and return as a DataFrame."""
        params = []
        for g, m in models.items():
            if m is None:
                continue
            params.append((g, *m.popt))
        return params
    
    def analyze(self, modelname='logistic2kzero', savefiles=False, outputdir=None):
        """Perform curve fitting on all groups, and display all plots."""
        today = date.today()
        if outputdir is None:
            outputdir = os.path.join(os.getcwd(), f'{today.strftime("%Y_%m_%d")}-{self.ID}')

        print(f'Files will be stored in {outputdir}')
        if os.path.exists(outputdir):
            sys.exit('Error: Output directory already exists!')
        os.mkdir(outputdir)

        print('Fitting models')
        models = self.fit_model(modelname=modelname, maxfev=2000)
        columns = {
            'richards': ['Group', 'U', 'L', 'k', 'v', 'x0'],
            'richardszero': ['Group', 'U', 'k', 'v', 'x0'],
            'logisticzero': ['Group', 'U', 'k', 'x0'],
            'logistic2k': ['Group', 'U', 'L', 'k1', 'k2', 'x0'],
            'logistic2kzero': ['Group', 'U', 'k1', 'k2', 'x0'],
            'gompertz': ['Group', 'U', 'L', 'k', 'x0'],
            'gompertzzero': ['Group', 'U', 'k', 'x0']
        }
            
        params = pd.DataFrame(self._collect_model_params(models), columns=columns[modelname])
        
        print('Computing areas and rates')
        areas = []
        for g, m in models.items():
            if m is None:
                continue
            A = m.area_under_curve()
            r = m.max_growth_rate()
            areas.append((g, A, r))
        df = pd.DataFrame(areas, columns=['Group', 'Area', 'Rate'])

        print('Generating figures')
        self.show(savefig=savefiles, outputdir=outputdir)
        self.show_temperatures(savefig=savefiles, outputdir=outputdir)
        self.show_all_curves(savefig=savefiles, outputdir=outputdir)
        self.show_combined_curves(savefig=savefiles, outputdir=outputdir)
        self.show_fitted_curves(savefig=savefiles, outputdir=outputdir, modelname=modelname)
        self.show_areas(savefig=True, outputdir=outputdir, modelname=modelname)
        self.show_rates(savefig=True, outputdir=outputdir, modelname=modelname)
        self.show_model_derivatives(modelname=modelname, savefig=savefiles, outputdir=outputdir)

        if savefiles:
            params.to_csv(os.path.join(outputdir, f'{self.ID}_OptimizedModelParameters.csv'), index=False)
            df.to_csv(os.path.join(outputdir, f'{self.ID}_AreasAndRates.csv'), index=False)
        else:
            print(params)
            print(df)
            
    def show_temperatures(self, savefig=False, filename=None, outputdir='.'):
        """Visualize the temperature distribution throughout the experiment."""
        x = self.combined.Time_hours
        y = self.combined.MeanTemp
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(ax=ax, x=x, y=y, color='black', marker='.', s=40)
        ax.set_title('Temperatures')
        
        if savefig:
            if filename is None:
                filename = f'{self.ID}_Temperatures.png'
            fig.tight_layout()
            fig.savefig(os.path.join(outputdir, filename))
            plt.close(fig)

        return fig, ax, (x, y)

    def show_comparison_to(self, others, savefig=False, filename=None, outputdir='.'):
        """Visually compare to another Plate object using Seaborn.
        
        Parameters
        -----------
        others <Plate/list>: if Plate, compare to the passed Plate. If list, compare to all Plates in list
        savefig <bool>: 
        filename <str>: 
        """
        if isinstance(others, Plate):
            final = self + others
        elif isinstance(others, list):
            final = deepcopy(self)
            for other in others:
                final += other
        else:
            sys.exit('Invalid input data')
        
        grid = sns.relplot(data=final.combined,
                           x='Time_hours',
                           y = 'logRatio',
                           kind='scatter',
                           col='Group',
                           hue='ID',
                           marker='.',
                           s=40,
                           col_wrap=5,
                           height=3)
        
        if savefig:
            if filename is None:
                filename = final.ID+'.png'
            plt.savefig(os.path.join(outputdir, filename))
        return grid, final
    
    def show_areas(self, modelname='logistic2kzero', savefig=False, filename=None, outputdir='.'):
        """..."""
        FS = 20
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.tick_params(axis='x', labelrotation=90, labelsize=FS)
        ax.tick_params(axis='y', labelsize=FS)
        ax.set_ylabel('Estimated biomass', fontsize=FS)

        areas = []
        if self.current_models is None:
            models = self.fit_model(modelname=modelname)
        else:
            models = self.current_models

        for group, model in models.items():
            if model is None:
                areas.append((group, 0))
            else:
                areas.append((group, model.area_under_curve()))

        for g, a in sorted(areas, key=lambda x: x[-1]):
            color = 'skyblue' if g != 'Blank' else 'salmon'
            ax.bar(g, a, color=color, edgecolor='black', lw=3)

        if savefig:
            if filename is None:
                filename = f'{self.ID}_Areas.png'
            fig.tight_layout()
            fig.savefig(os.path.join(outputdir, filename))
            plt.close(fig)

        return fig, ax, areas
    
    def show_rates(self, modelname='logistic2kzero', savefig=False, filename=None, outputdir='.'):
        """..."""
        FS = 20
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.tick_params(axis='x', labelrotation=90, labelsize=FS)
        ax.tick_params(axis='y', labelsize=FS)
        ax.set_ylabel('Max Growth Rate', fontsize=FS)

        rates = []
        if self.current_models is None:
            models = self.fit_model(modelname=modelname)
        else:
            models = self.current_models

        for group, model in models.items():
            if model is None:
                rates.append((group, 0))
            else:
                rates.append((group, model.max_growth_rate()))

        for g, a in sorted(rates, key=lambda x: x[-1]):
            color = 'skyblue' if g != 'Blank' else 'salmon'
            ax.bar(g, a, color=color, edgecolor='black', lw=3)

        if savefig:
            if filename is None:
                filename = f'{self.ID}_Rates.png'
            fig.tight_layout()
            fig.savefig(os.path.join(outputdir, filename))
            plt.close(fig)

        return fig, ax, rates

    def show_model_derivatives(self, job=None, modelname='logistic2kzero', savefig=False, filename=None, outputdir='.'):
        """
        Plot the fitted functions and their first and second derivatives. This serves as a visual guide that
        the maximum growth rates make sense. The vertical dashed lines correspond to the x coordinate of the
        inflection point in the fitted curve (as this is where the max rate is computed). The horizontal dashed
        line in the 2nd derivative plots are at y=0. Look for the following things:

        - The dashed line go through the maximum of the first derivative
        - The 2nd derivative goes through the intersection of the two dashed lines
        - The vertical dashed line visually appears to be located at the point of steepest growth for the fitted
          curves.
        - A good fit is usually achieved if the 1st and 2nd derivatives are able to reach zero close to x=0
        - If the 1st derivative maximum is located at x=0, then this is a sign that the fitting was not successful.
          The lowest allowed value for the parameter `x0` is the minimum value of x (`min(x)`), which usually is zero.
        Parameters
        ----------
        group : None/str. If str, make plot for a single group. If None, make plot for all groups
        modelname : which model to use for the fitting
        savefig : bool, save figure to png
        filename : filename of saved png
        outputdir : destination folder for png

        Returns
        -------

        """
        if job is not None:
            if isinstance(job, BaseModel):
                m = job
            elif isinstance(job, str):
                m = self.fit_model(group=job, modelname=modelname)

            xs = np.linspace(min(m.x.values)-5, 24, 300)
            xdata = m.x.loc[m.x < 24]
            ydata = m.y[:len(xdata)]
            infl = m.inflection_point()

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 3), dpi=150)

            ax1.plot(xs, m.func(xs, *m.popt))
            ax1.scatter(xdata, ydata, color='black', s=0.5)
            ax1.axvline(infl, ls=':', color='black', lw=0.8, zorder=99)

            ax2.plot(xs, m.deriv1(xs, *m.popt))
            ax2.axvline(infl, ls=':', color='black', lw=0.8, zorder=99)

            ax3.plot(xs, m.deriv2(xs, *m.popt))
            ax3.axvline(infl, ls=':', color='black', lw=0.8, zorder=99)
            ax3.axhline(0.0, ls=':', color='black', lw=0.8, zorder=99)

            ax1.set_title('Function')
            ax2.set_title('First Derivative')
            ax3.set_title('Second Derivative')
            plt.tight_layout(h_pad=1)
            if savefig:
                if filename is None:
                    filename = f'{self.ID}_group={job}_Derivatives.png'
                plt.savefig(os.path.join(outputdir, filename))
            return fig, (ax1, ax2, ax3)
        else:
            fig, axes = plt.subplots(ncols=3, nrows=len(self.groups()), figsize=(4, 15), dpi=200)
            counter = 0
            for (col1, col2, col3), group in zip(axes, self.groups()):
                m = self.fit_model(group=group, modelname=modelname)
                xs = np.linspace(min(m.x.values)-5, 24, 300)
                xdata = m.x.loc[m.x < 24]
                ydata = m.y[:len(xdata)]
                infl = m.inflection_point()

                col1.set_ylabel(group, fontsize=5)
                if counter == 0:
                    col1.set_title('Function', fontsize=5)
                    col2.set_title('1st Derivative', fontsize=5)
                    col3.set_title('2nd Derivative', fontsize=5)
                counter += 1

                col1.set_yticklabels([])
                col2.set_yticklabels([])
                col3.set_yticklabels([])
                col1.set_yticks([])
                col2.set_yticks([])
                col3.set_yticks([])
                col1.set_xticks([])
                col2.set_xticks([])
                col3.set_xticks([])

                col1.plot(xs, m.func(xs, *m.popt))
                col1.scatter(xdata, ydata, color='black', s=0.5)
                col1.axvline(infl, ls=':', color='black', lw=0.8, zorder=99)

                col2.plot(xs, m.deriv1(xs, *m.popt))
                col2.axvline(infl, ls=':', color='black', lw=0.8, zorder=99)

                col3.plot(xs, m.deriv2(xs, *m.popt))
                col3.axvline(infl, ls=':', color='black', lw=0.8, zorder=99)
                col3.axhline(0.0, ls=':', color='black', lw=0.8, zorder=99)

            fig.tight_layout(h_pad=0)
            if savefig:
                if filename is None:
                    filename = f'{self.ID}_Derivatives.png'
                plt.savefig(os.path.join(outputdir, filename))
            return fig, axes

    def show_rates_and_areas_comparison(self, models=None, savefig=False, outputdir='.', a=None, b=None):
        if models is None:
            models = ['Logistic2k', 'Logistic2kZero', 'LogisticZero', 'Richards', 'RichardsZero', 'Gompertz', 'GompertzZero']

        data = []
        for model in models:
            print(f'Fitting models for {model}')

            fits = self.fit_model(modelname=model, maxfev=2000)
            for group, fit in fits.items():
                rate = fit.max_growth_rate()
                area = fit.area_under_curve(a=a, b=b)
                row = (model, group, rate, area)
                data.append(row)

        df = pd.DataFrame(data, columns=['Model', 'Group', 'Rate', 'Area'])

        # Rates
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.lineplot(ax=ax, data=df, x='Group', y='Rate', hue='Model', marker='.', mec='black')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(ls=':', lw=0.5)
        ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=4)
        fig.tight_layout()
        if savefig:
            filename = f'{self.ID}_ModelComparison_Rates.png'
            fig.savefig(os.path.join(outputdir, filename))

        # Areas
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.lineplot(ax=ax, data=df, x='Group', y='Area', hue='Model', marker='.', mec='black')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(ls=':', lw=0.5)
        ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=4, title=f'Integration limits: ({a}, {b})')
        fig.tight_layout()
        if savefig:
            filename = f'{self.ID}_ModelComparison_Areas.png'
            fig.savefig(os.path.join(outputdir, filename))
