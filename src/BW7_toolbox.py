import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.signal as sgn
import scipy.stats as stat
import matplotlib.gridspec as grds
import matplotlib.mlab as mlab

from colorama import Fore,Style

import statsmodels.tsa.stattools as sta


def correlation_study(file='challenge-data-v2.csv'):
    sns.set_style("whitegrid")

    # *****************************************************
    # Loading the data
    data = pd.read_csv('challenge-data-v2.csv', index_col='event_date', parse_dates=True)

    # *****************************************************
    # Changing the name of the colums to a more suitable (shorter) form
    data.columns = ['sups', 'moff', 'mon', 'hd']

    # *****************************************************
    # Scaling the values to work in a more suitable way (avoiding super-high values)
    for feature in data.columns:
        if data[feature].values.dtype != 'object':
            scale_factor = np.round(np.log10(np.mean(data[feature])))
            data[feature] = data[feature] / 10 ** scale_factor
            print Fore.BLUE+'The feature: ' + feature + ' was rescaled by a factor of ' + str(10 ** scale_factor)

    print 'IMPORTANT: Take these rescaling in account when reading the plots'
    print(Style.RESET_ALL)

    # *****************************************************
    # Computation of the distributions per year for the different features
    years = np.unique(data.index.year)

    # Definition of the figure.
    figid = plt.figure('Temporal distributions by year', figsize=(20, 10))

    # Definition of the matrix of plots.
    col = len(data.columns[data.columns != 'hd'])
    gs = grds.GridSpec(3, col)

    for i in range(col):

        ax1 = plt.subplot(gs[0, i])
        ax2 = plt.subplot(gs[1, i])

        legend = []
        for y in years:
            dat = data[data.columns[i]][str(y)].values
            ax1.plot(np.arange(len(dat)), dat, '-')
            legend.append(str(y))
        ax1.legend(legend)
        ax1.set_title('Daily ' + data.columns[i])

        legend = []
        for y in years:
            dat = data[data.columns[i]][str(y)].resample('M').values
            ax2.plot(np.arange(len(dat)) + 1, dat, '-o')
            legend.append(str(y))
        ax2.legend(legend)
        ax2.set_title('Monthly ' + data.columns[i])
        plt.xlim([1, 12])

        ax3 = plt.subplot(gs[2, i])
        legend = []
        for y in years:
            dat = data[data.columns[i]][str(y)]
            #         dat = dat.groupby(data.index.dayofweek).mean()
            dat = dat.groupby(dat.index.dayofweek).mean()
            dat.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
            dat.plot(style='-o')
            legend.append(str(y))
        ax3.legend(legend)
        ax3.set_title('Day week ' + data.columns[i])

    # *****************************************************
    # Computation of the distribution of activity per month for the different time series that are present in the data
    years = np.unique(data.index.year)
    # Definition of the figure.
    figid = plt.figure('Monthly distribution of features',figsize=(20, 10))

    # Definition of the matrix of plots. In this case the situation is more complex that is why I need to define a
    # matrix. It will be a dim[2x3] matrix.
    col = len(data.columns[data.columns!='hd'])
    rows = len(years)
    gs = grds.GridSpec(rows,col)
    months=['Jan','Feb','Mar','Aprl','May','Jun','Jul','Agst','Sep','Oct','Nov','Dec']
    colors = sns.hls_palette(12, l=.5, s=.6)


    for c in range(col):
        for r in range(rows):
            ax1 = plt.subplot(gs[r, c])

            dat_year = data[data.columns[c]][str(years[r])]

            for m in range(1,13):
                dat = dat_year[dat_year.index.month==m].values
                ax1.plot(np.arange(len(dat)),dat,'-', color=colors[m-1])
            if r==0 and c==col-1:
                ax1.legend(months, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.,ncol=2, fancybox=True,frameon=True)

            if c==0:
                ax1.set_ylabel('Year: '+str(years[r]))

            if r==0:
                ax1.set_title('Feature: '+str(data.columns[c]))

    # *****************************************************
    # Computation of the distribution of cross correlations per month for the different time series that are present in the data
    years = np.unique(data.index.year)

    # Definition of the figure.
    figid = plt.figure('Monthly features cross-correlations', figsize=(20, 10))

    # Definition of the matrix of plots.
    feature1 = [0, 0, 2]
    feature2 = [2, 1, 1]

    col = len(data.columns[data.columns != 'hd'])
    rows = len(years)

    gs = grds.GridSpec(rows, col)
    months = ['Jan', 'Feb', 'Mar', 'Aprl', 'May', 'Jun', 'Jul', 'Agst', 'Sep', 'Oct', 'Nov', 'Dec']
    # colors = sns.color_palette("Set2", 12)
    colors = sns.hls_palette(12, l=.5, s=.6)

    for c in range(col):
        for r in range(rows):
            ax1 = plt.subplot(gs[r, c])

            dat_year_feat1 = data[data.columns[feature1[c]]][str(years[r])]
            dat_year_feat2 = data[data.columns[feature2[c]]][str(years[r])]

            for m in range(1, 13):
                dat_feat1 = dat_year_feat1[dat_year_feat1.index.month == m].values
                dat_feat1 = np.subtract(dat_feat1, np.mean(dat_feat1))
                dat_feat2 = dat_year_feat2[dat_year_feat2.index.month == m].values
                dat_feat2 = np.subtract(dat_feat2, np.mean(dat_feat2))
                dat = sgn.correlate(dat_feat1, dat_feat2, mode='same')
                ax1.plot(np.linspace(-15, 15, len(dat)), dat, '-', color=colors[m - 1])
            if c == 0:
                ax1.set_ylabel('Year: ' + str(years[r]))
            if r == 0 and c == col - 1:
                ax1.legend(months, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., ncol=2, fancybox=True,
                           frameon=True)
            if r == 0:
                ax1.set_title('Xcorr: ' + str(data.columns[feature1[c]]) + ' and ' + str(data.columns[feature2[c]]))



    # *****************************************************
    # Computation of the distribution of activity per month for the different time series that are present in the data
    years = np.unique(data.index.year)

    # Definition of the matrix of plots.
    feature1 = [0,0,2]
    feature2 = [2,1,1]

    for f in range(len(feature1)):

        figid = plt.figure(
            'Rolling correlation coefficient and weekly activity of ' + data.columns[feature1[f]] + ' and ' + data.columns[
                feature2[f]], figsize=(20, 10))
        rows = len(years)

        gs = grds.GridSpec(4, 4)

        for r in range(rows):
            ax1 = plt.subplot(gs[0, :])
            ax2 = plt.subplot(gs[1, :])
            ax3 = plt.subplot(gs[2, :])

            dat_year_feat1 = data[data.columns[feature1[f]]][str(years[r])]
            dat_year_feat2 = data[data.columns[feature2[f]]][str(years[r])]

            ref = ax1.plot(dat_year_feat1.resample('W'))
            ax2.plot(dat_year_feat2.resample('W'))

            xcorr = pd.rolling_corr(dat_year_feat1, dat_year_feat2, 14)
            ax3.plot(xcorr)

            ax4 = plt.subplot(gs[3, r])
            n, bins, patches = ax4.hist(xcorr.values[np.logical_not(np.isnan(xcorr.values))], bins=np.round(len(xcorr) / 6),
                                        facecolor=ref[0].get_color(), edgecolor=ref[0].get_color())
            mediana = ax4.axvline(np.median(xcorr.values[np.logical_not(np.isnan(xcorr.values))]), color='r', linestyle='--')
            ax4.set_xlim([-1, 1])
            ax4.set_xlabel('CorrCoef year '+str(years[r]))
            ax4.set_label(mediana)

            print '-------------------------------------------------'
            print 'Correlation distribution for year ' + str(years[r])
            print 'Mean:', xcorr.mean()
            print 'Median:', xcorr.median()
            print 'Standard deviation:', xcorr.std()
            print 'Kurtosis:', xcorr.kurtosis()  # Kurtosis is mainly related with outliers not with the central peak
            print 'Skewness:', xcorr.skew() #Take in account that this value has not the substraction of the skew of a normal distribution (3)

            if (np.abs(xcorr.skew())) < 0.65:
                mu, sigma = stat.norm.fit(xcorr.values[np.logical_not(np.isnan(xcorr.values))])
                print 'Normal distribution fitted!'
                print 'mu=' + str(mu)
                print 'sigma=' + str(sigma)

                fitted_normal = mlab.normpdf(bins, mu, sigma) * np.max(
                    xcorr.values[np.logical_not(np.isnan(xcorr.values))])
                # print fitted_normal
                normfit = ax4.plot(bins, fitted_normal * np.max(n), 'r--', linewidth=2, color="#3498db")
                ax4.legend(['Median '+str(round(xcorr.median(),2)), 'N(' + str(round(mu, 1)) + ',' + str(round(sigma, 2)) + ')'], loc='best')
            else:
                ax4.legend(['Median '+str(round(xcorr.median(),2))], loc='best')


        ax1.set_ylabel(data.columns[feature1[f]])
        ax2.set_ylabel(data.columns[feature2[f]])
        ax3.set_ylabel('Rolling correlation, 14 days period')


        ax1.legend(years, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., ncol=1, fancybox=True, frameon=True)

    if file=='challenge-data-v2.csv':
        conclusions =""
    return True


def seasons_study(file='challenge-data-v2.csv'):
    sns.set_style("whitegrid")

    # *****************************************************
    # Loading the data
    data = pd.read_csv('challenge-data-v2.csv', index_col='event_date', parse_dates=True)

    # *****************************************************
    # Changing the name of the colums to a more suitable (shorter) form
    data.columns = ['sups', 'moff', 'mon', 'hd']

    # *****************************************************
    # Scaling the values to work in a more suitable way (avoiding super-high values)
    for feature in data.columns:
        if data[feature].values.dtype != 'object':
            scale_factor = np.round(np.log10(np.mean(data[feature])))
            data[feature] = data[feature] / 10 ** scale_factor
            print Fore.BLUE + 'The feature: ' + feature + ' was rescaled by a factor of ' + str(10 ** scale_factor)

    print 'IMPORTANT: Take these rescaling in account when reading the plots'
    print(Style.RESET_ALL)


    # *****************************************************
    # Computation of the distributions per year for sign ups
    years = np.unique(data.index.year)

    # Definition of the figure.
    figid = plt.figure('Temporal Sign-ups distributions by year', figsize=(20, 10))

    # Definition of the matrix of plots. In this case the situation is more complex that is why I need to define a
    # matrix. It will be a dim[2x3] matrix.
    col = len(data.columns[data.columns == 'sups'])
    gs = grds.GridSpec(3, col)

    for i in range(col):

        ax1 = plt.subplot(gs[0, i])
        ax2 = plt.subplot(gs[1, i])

        legend = []
        for y in years:
            dat = data[data.columns[i]][str(y)].values
            ax1.plot(np.arange(len(dat)), dat, '-')
            legend.append(str(y))
        ax1.set_title('Daily ' + data.columns[i])

        legend = []
        for y in years:
            dat = data[data.columns[i]][str(y)].resample('M').values
            ax2.plot(np.arange(len(dat)) + 1, dat, '-o')
            legend.append(str(y))
        ax2.set_title('Monthly ' + data.columns[i])
        plt.xlim([1, 12])

        ax3 = plt.subplot(gs[2, i])
        legend = []
        for y in years:
            dat = data[data.columns[i]][str(y)]
            dat1 = dat.groupby(dat.index.dayofweek).mean()
            dat1.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
            dat1.plot(style='-o')

            legend.append(str(y))
        ax3.set_title('Day week ' + data.columns[i])

        if i == 0:
            ax1.legend(years, bbox_to_anchor=(1, 1), loc='best', borderaxespad=0., ncol=1, fancybox=True, frameon=True)

    # *****************************************************
    # Computation of different spectral estimators per year for sign ups
    years = np.unique(data.index.year)

    # Definition of the figure.
    figid = plt.figure('Spectral estimators per year', figsize=(20, 10))

    # Definition of the matrix of plots.
    col = 3
    rows = len(years)
    gs = grds.GridSpec(rows, col)

    # Spectral parameters
    Window_length = 32  # in days
    # This is equivalent to multitaper estimation but just one taper.
    window = sgn.slepian(Window_length, width=0.3)
    overlap = 0.85

    for r in range(rows):

        ax1 = plt.subplot(gs[r, 0])
        ax2 = plt.subplot(gs[r, 1])
        ax3 = plt.subplot(gs[r, 2])

        dat = data['sups'][str(years[r])].values

        spec = np.fft.fft(dat)
        n = dat.size
        f = np.fft.fftfreq(n)

        ax1.plot(f[1:round(len(spec) / 2)], np.abs(spec[1:round(len(spec) / 2)]))

        f, t, Sxx = sgn.spectrogram(dat, nperseg=Window_length, window=window,
                                    noverlap=np.round(Window_length * overlap))

        #     print 'Spectrogram size',Sxx.shape
        #     print 't',t.shape
        #     print 'f',f.shape

        #     print 'f',f

        spec = np.abs(Sxx)

        ax2.imshow(spec, vmax=np.percentile(spec, 98), aspect='auto', cmap='viridis', interpolation='none',
                   origin='lower', extent=[np.min(t), np.max(t), np.min(f), np.max(f)])
        ax3.imshow(np.log(spec), vmax=np.percentile(np.log(spec), 98), aspect='auto', cmap='viridis',
                   interpolation='none', origin='lower', extent=[np.min(t), np.max(t), np.min(f), np.max(f)])

        ax2.set_ylabel('Year: ' + str(years[r]) + '\nf[Hz]', multialignment='center')

        if r == 0:
            ax1.set_title('PSD[n.u.]')
            ax2.set_title('Spectrogram[n.u.]')
            ax3.set_title('Spectrogram[log10]')
        if r == rows - 1:
            ax1.set_xlabel('f[Hz]')
            ax2.set_xlabel('Time[days of the year]')
            ax3.set_xlabel('Time[days of the year]')


    return True