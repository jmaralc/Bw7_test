# Bw7_test

This test was originally done on March 2017. The company liked it and I joined BW7 as Data Scientist. It is a bit embarasing for me to review the code some years later but it may be helpful to share it.

For executing it just run:
```
make run
```
from the root folder of this repository. This will generate a docker image and it will run it. From the shell one can copy the token for the Jupyter notebook session. Once within the session, just enter the notebooks folder and run `Test challenge _seasonality`.

The text below is the original answer that can be also found in the notebook. 

# Test challenge_seasonality by Javier Martinez

For answering the requested case of study I have programmed a python file called "BW7_toolbox.py" that is attached in my submission. This file contains two functions: correlation_study and seasons_study. Each one is the answer for each of the question suggested in the case of study. In the same project there is also a __init__ file just for the sake of generating a python package.

To execute the functions just unzip the file in the folder where regularly is used as a python path and use "import BW7_toolbox as BW7" to add it to your project. Once in your project you could call the evaluation of the solutions as BW7.correlation_study(filename) or BW7.seasons_study(filename). By default the data for this case of study will be loaded but any csv file that is in the same folder that the python files coudl be also used just by using the proper filename (with the csv extension as part of the name). 

All the plots during the execution will be executed in different windows to allow better insight/control of the plots. 

## Q1.1) Using Python and the libraries and packages of your own choice, write a well-structured and readable program that can be used to analyse the data set for correlations between the time series. The program should work with data files that are similar, but not necessarily identical to the sample provided. Your response should include the program itself and brief instructions about any dependencies that should be installed before running it.

### Proposal


The idea behind the solution of this first question is to provide the user the most usefol plots and information to observe any correlation. With that in mind after the execution of "correlation_study" the next plots will be presented on the screen (different windows):
* Temporal distribution by year. The idea of this plot is to provide the user a broad overview of the activity of the different features. For that reason each feature has assigned one colum while the rows represent different distributions:
    * First row, distribution of features accross the year for different years.
    * Second row, distribution of the features per month for different years. 
    * Third row, distribution of the featyres during the weekdays for different years. 

* Monthly distribution of features. In this plot I present the monthly distribution of the different features. The idea here is to provide some insight into the data before the next plot. 
* Monthly features crosscorrelations. This plot address explicitly the question of the correlation between the different time series of the data set in the temporal scale of a month. I did it in this way to address that the data is non-stationary and hence this relationship depends strongly of the period of the year (month). Take in account that the first zero lag is from the first time series mentioned in the plot to the second. Then if you observed a peak in the positive side of the crosscorrelograms should be interpreted as a larger probability of having the event two in that lag of time respect the first time series mentioned in the plot. The cross correlations are computed to have the same number of samples than the original time series (I find a 'full' version non informative in this time scale). Whenever you find an empty plot is because there was not enough collected data to calculate any statistics. 

* Rolling corelation coefficient and weekly activity of the different features. In my opinion this is the more informative than the previous one but address the problem a bit different. This figure shows the moving correlation coefficient (with a temporal window of 14 days, 2 weeks) together with the dynamics of the features (resampled to weekly activity for the sake of smooth it). Interestingly in this way it is possible to extract some statistics on what it is going with this coefficient divided by years. This is shown in the last row of the plot. This row presents the distributions of the correlations during the year and provide a nice picture of the business choices taken during that year. It also measure the effectiveness of whatever event because this will have an impact in the shape and other moments of those distributions. The median (assuming that in general the situation wont be gaussian) is shown as a parameter for decision making (or index of correlation for a year). In the cases the skweness of the distribution is close to a gaussian, a normal distribution will be fitted to the data (and the mean and variance of the distribution will be shown). In addition to this in the console there will be reflected also the activity of the different moments of the distributions per year. 

These plots are just tools, a window to the observer to have an insight on what it is happening with the correlation of the features. 

Non-numeric features will not be considered for this study. That is the reason for discarding (not using) the holydays.

In the next cells I present the same code that is used in the function that will help in the reporting of the specific case of study.

There is some redundancy in the code but this was done with the idea of reusability of blocks of code, not only for me but for other developers too.


#    Q1.2) Run the program on the provided data sample from PetFood and comment on the output.

The analysis of this data is quite interesting.

The signups distribution along the year suffer from some seasonality in the years with only online adds. When the new offline adds started the distribution is completely different and the total amount of sign ups is increased what looks like 20%. The same happend with the distribution during the days of the week. But with less impact and having always the friday as the larger sign up day. 

Comparing the two methods of advertisements the online seems a stable investment of money with a better prediction on the inpact of the sign ups than the offline methods. These ones consume a big budget of investment and do not provide a correlation with the activity of the sign ups. 

The monthly distribution of all the features shows this situation too. The peaks if sign ups were well stablished before the offline add, but seem to vanish and shows more noisy picture in the last two years.
The investment on online add is really seasonal having usualy minimums during the first month of the year. 


The monthly crosscorrelations show clear that the online services are moderate correlated with the sign ups (response of the customer) during the whole year, following a period of 5 days. The situation is a bit different, though, for november which present high(moderate high for 2014) degree of correlation. In addition the activity of the sign ups is after the zero which indicates a possibility of causation (warning!) of the online add on the behaviour of the consumers. 

The offline shows much noisy uncorrelated picture. There is no structure (periodic) on the crosscorrelogram and the activity of the customers seems to happend BEFORE the major investments on these adds. This is interesting. It could be a proof that the campaign on those media is failing, going fater the noisy behavior of the market or the will of a non expert. It seems like after checking slightly/substantial increasings in signups with moderate spends on the offline, the company decided to invest more or invest in a way less atractive to the customer. It could be also a saturation effect on the decision of the customer that suffer from unnecesary adds on TV. This is specially clear in November but it is a general trend. 

The relation between the two ways of adds seems uncorrelated except for november 2016 where probably due to the christmas campaing the company invest more money in both...but having only major effect on the customer at the level of the online media. 

The plots of the rolling cross correlation coefficient shows a tendency of better performance and predictability of customer choices with the online adds. The median of the correlation was higher for 2014 and 2015 while there is a decrease of this when the offline add started (2016,2017). Then it is not only that the offline is not really predicting the behaviour of the customer but it seems (well, this is a causal declaration but the data point towards that) that is decreasing the correlation of the customer behavior based on the online media. 

Summarizing, despite the total increasing of sign ups specially after the big first investment. This ofline campaign seems not useful for predicting the behavior of the customers. The first two years the nice are nice and according to the distributions of signups the situation seems to be fitting a Poisson process which is really convenient when one has to scale system (serers, services, etc) with the general queue theory. The new offline adds disrupted this situation and made it more difficult to predict putting almost at random level the investment with the effect on the customer. 


#     Q1.3) Comment on additional approaches that could be used to search for various types of correlations in the data set.

Further ideas that one can use here for correlations:

* Study of the cross-spectral density and coherence
* Study of granger causality
* Study of PCA/ICA-ish analysis on the time series to see any pattern that coud be capturing the variance in a meaningful way
* Use of the statsmodels package to apply some statistical test on the correlation http://statsmodels.sourceforge.net/ I am currently exploring this toolbox but I could not extract the whole potential
* Split the offline spends based on the type of media (TV,newspaper, etc). Probably these media has really different approach from the customer behavior side


# Q2) SEASONS AND CYCLES 	
## Q2.1) Using Python and the libraries and packages of your own choice, write	a well-structured and readable program that can be used to identify periodic behaviour in the “signups” time series. The program should work with data files that are similar, but not necessarily identical to the sample provided. Your response should include the program itself and brief instructions about any dependencies that should be installed to run it.

## Proposal

The idea behind the solution the second question is to provide the user the most useful plots and information to observe any seasonality or cycles in the data. With that in mind after the execution of "seasons_study" the next plots will be presented on the screen (different windows):

* Temporal Sign-ups distributions by year. The purpose of this plot is to give certain broad view on the data. It is also used in the previous function because helps to visualize the activity of the different years and why is convenient to split the analysis per year (due to the non-stationarity of the processes). Each row represents:
    * First row, the daily activity of the sign ups during the period of one year (color coded per year).
    * Second row, the monthly activity of the sign ups. This provides a broader view that happend in a wider time scale; ignoring the noisy daily signal some insights could be observed.
    * Third row, the activity of sign ups during the week days. This is a small time scale but helps to see the relevance of certain days of the week. 
    
* Spectral estimators per year. This plots is the core of the idea to capture the seasons and cycles. It is based on the well-known Fourier analysis. Eacho row represent the espectral estimators for each year. The colums represent:
    * The power spectral density of the signal (PSD). This concept capture the general distribution of power of the signal during the observation of time (one year in this case). It provides a first estimator of which spectral components are relevant for the time series. The peaks in frequency represent periods (seasons) of the signal.
    * Spectrogram. This spectral estimator provides slightly more insight in the temporal components of the time series in the frequency domain. Using a sliding window one can compute the Fourier transform on those windows and chunking them together while the window is moving, presenting a 2-D map of time x frequency (x axis for time, y axis for frequency). Using a window of 32 days (samples, just for convenience of being a power of 2) and overlapping each window 80% of the neighbour windows I provide a powerful tool for evaluating the periodicity. 
    * Log10 scale spectrogram. This colum is similar to the previous but in log10 scale to provide no so crispy visual input and observe other components. 

The DC components (f=0) is not shown on the PSD.

These plots are just tools, a window to the observer to have an insight on what it is happening with the sign up periodicity. 

In the next cells I present the same code that is used in the function that will help in the reporting of the specific case of study.

There is some redundancy in the code but this was done with the idea of reusability of blocks of code, not only for me but for other developers too.

# Q2.2) Run the program on the data sample from PetFood, and comment on	the output

Once the function is called the results point clear to certain level of seasonality/ periodicity similar for the two first years and different for the last two years.

The temporal distributions per year shows that the 2014 and 2015 have a quite similar behaviour in the monthly scale and also in the day of the week. Two troughs can be observed in april/ may and october, while two peaks are visible at february and june. The first year with the new offline adds shows also two troughs and two peaks, the earliers as before april/may and october, but the later, the peaks, with a slightly shift now at february and july. 

Observing the PSD per year it is possible to observe peaks of activity at 0.15 $\mu Hz$ and 0.28 $\mu Hz$ for 2014. With a $f_s = \frac{1}{1day} Hz$ that is $f_s=11.574 \mu Hz$  this represent for the first case 1 cycle every 2.7 months, while the other peak means 1 cycle every 1.3 months. Despite of the peaks the PSD seems to have the main power concentrated around 0.1-0.09 which will mean 1 cycle every 3.8-4.2 months. This last observation is compatible with the observation in the previous plot. 

The PSD for 2015 shows less spiky activity but rather more power at 0.1 with a small peak at 0.28. It is important to note also that a lot of power, as also happened before with 2014, it is located around 0.05 which it translate in a cycle every, aprox, 7 months. This could suggest a general trend in that range.  

The PSD of 2016 shows also this weekly periodicity at around 7 days. The peak at 0.15 have changed here to a wider distribution, more peaky at 0.128 which means 1 cycle every 3 months (slower). The peak at 0.28 seems still present though. 

The PSD of 2017 has so few samples that is difficult to consider the estimator as good. Nonetheless one can take a llok and check the trends. The trend is that there is a distribution around 0.1 which means 3.85 months.

Observing the spectrograms the temporal seasons of the spectral componenets are much more clear. 2014 and 2015 share a common tendency of having a season of 0.1 at may/june and november. 2016 despite of having that compoonent too (a bit slower) it show it in april and november. Despite of that the three years have a clear component in the season befor christmas, having almost the same (a bit shifted) components for the range of november/december. 


This results shows that the method is valid to capture the seasons of the sign ups time series. 

# Q2.3) Discuss any additional methods and data sources that would be useful to improve the detection of cycles in the number of signups.

Fruther methods that could be explored:


* Multitaper estimator, this provides a better estimator.
* Wavelet estimator of the spectrum
* Phase relationships with Hilbert transform as support to extract the phase 
* The ideas of this paper, "Brain Oscillations and the Importance of Waveform Shape" http://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(16)30218-2?_returnURL=http%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1364661316302182%3Fshowall%3Dtrue


#  Q2.4) Discuss to what degree this same code solution can be expected to	work for a completely different customer, selling a	completely different product, in a different market. Would the approach	need to be adjusted to accommodate such a general setting?

The use of Fourier transform and the spectogram is a general approach to any time series. In that sense the method is agnostic to the customer, product or market; hence it is validity is appropriate for the task of getting insights into the cyclic dynamics of any system.

Despite of that the main concern that one should take when dealing with this spectral estimators is the method that one used because the parameters could have a big impact in the representation of the estimator. This happens also with the wavelet transform and is the fact of having trade off when using the technique. One critical point is the window for the spectrogram because there should be big enough to capture enough degrees of freedom but narrow to fit the characteristic of the oscillations that one wants to observe. The bias and variance of the estimator is also a matter of debate. 

None the less these are technical details that do not affect the customer. Only the data scientist must be aware of this situation and have wide open eyes to do not let these effects uncontrolled. 

 

