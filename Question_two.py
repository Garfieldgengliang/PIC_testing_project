'''This piece of script is for data analysis of file Pork Price.xlsx'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind

'''Part 1: Data inspection'''
'''Load the raw data into python'''
# to run following line of code, please set your IDE working directory to the folder containing the excel file
xlsx = pd.ExcelFile('./Slaughter Data.xlsx')
slaughter_data = pd.read_excel(xlsx, sheets = 'sheet1')
slaughter_data.columns = ['Sire Line', 'NO.', 'HCW', 'Grade', 'BF']
slaughter_data = slaughter_data[['Sire Line', 'NO.', 'HCW', 'BF','Grade']]

'''Check null values and typos'''
slaughter_data.isnull().sum()  # no blank data
slaughter_data['Sire Line'].unique()  # two lines
slaughter_data.groupby(['Sire Line'])['NO.'].count()  # 11419 versus 11195, basically the same
basic_statistics = slaughter_data[['HCW', 'BF','Grade']].describe()
# all the values seems good, for example, the min and max are both in reasonable range from mean w.r.t sd

'''Part 2: Data Visualization'''
'''The distribution of numeric variables HCW, BF given each of two sire liness'''
Sire_N = slaughter_data.loc[slaughter_data['Sire Line'] == 'N']
Sire_P = slaughter_data.loc[slaughter_data['Sire Line'] == 'P']

figure_four, axes_four = plt.subplots(nrows=2, ncols=2, sharey=True, figsize = (14,12))
figure_four.suptitle('Distribution of HCW & BF given each of two sire lines', fontsize = 20)

sbn.distplot(Sire_N['HCW'], kde = False, bins = 50, color = 'coral', ax = axes_four[0,0])
axes_four[0,0].set(title = 'Sire N, HCW', xlabel = 'HCW(kg)', ylabel = 'Frequency')

sbn.distplot(Sire_P['HCW'], kde = False, bins = 50, color = 'deepskyblue', ax = axes_four[1,0])
axes_four[1,0].set(title = 'Sire P, HCW', xlabel = 'HCW(kg)', ylabel = 'Frequency')

sbn.distplot(Sire_N['BF'], kde = False, bins = 20, color = 'coral', ax = axes_four[0,1])
axes_four[0,1].set(title = 'Sire N, BF', xlabel = 'BF at 6-7 ribs(mm)')

sbn.distplot(Sire_P['BF'], kde = False, bins = 20, color = 'deepskyblue', ax = axes_four[1,1])
axes_four[1,1].set(title = 'Sire P, BF', xlabel = 'BF at 6-7 ribs(mm)')

plt.subplots_adjust(wspace=0.05, hspace=0.3)
plt.subplots_adjust(left = 0.05, bottom = 0.1,
                    right = 0.95, top = 0.9)
plt.show()

'''Part 3: Compare details performances and grades of each sire line'''
# First we get some basic statistcs for each group so we can have an overall idea of their distributions
groupby_sireline_statistics = slaughter_data[['Sire Line', 'HCW', 'BF','Grade']].groupby(['Sire Line']).describe().transpose()
# Then we can use Kolmogorov¨CSmirnov test to see
# if all these three variables for Sire N and Sire P are from same distribute
_, ks_p_HCW = ks_2samp(Sire_N['HCW'], Sire_P['HCW'])  # p = 1.90 * 10^-7
_, ks_p_BF = ks_2samp(Sire_N['BF'], Sire_P['BF'])  # p = 4.45 * 10^-18
_, ks_p_Grade = ks_2samp(Sire_N['Grade'], Sire_P['Grade']) # p = 0.0087

# Now as Sire N and P has basically same amount of samples and same std for HCW, BF and Grade,
# we can use student t-test to compare if these group of samples have same mean for HCW, BF and Grade
_, ttest_p_HCW = ttest_ind(Sire_N['HCW'], Sire_P['HCW'])  # p = 1.55 * 10^-8
_, ttest_p_BF = ttest_ind(Sire_N['BF'], Sire_P['BF'])  # p = 1.16 * 10^-16
_, ttest_p_Grade = ttest_ind(Sire_P['Grade'], Sire_N['Grade'])  # p = 0.206

groupby_sireline_statistics.to_excel('./Sire_Line_statistics.xlsx')

'''Part 4: Model construction'''
# As a bonus part of this task, I want to build a classification model for grade evaluation given HCW and BF
# based on the assumption that grade is only determined by these two factors
# In reality, such criterion could already exists and could be determined by more factors such as ADG

working_data = slaughter_data[['HCW', 'BF', 'Grade']]
working_data['Grade'] = pd.Categorical(working_data.Grade) # setting grade column as categorical

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

kf = KFold(n_splits=5, random_state=1, shuffle=True)
accuracy_list = []
for train_index, test_index in kf.split(working_data):
    X_train, X_test = working_data.loc[train_index, ['HCW', 'BF']].to_numpy(), working_data.loc[test_index, ['HCW', 'BF']].to_numpy()
    y_train, y_test = working_data.loc[train_index, 'Grade'].to_numpy(), working_data.loc[test_index, 'Grade'].to_numpy()
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    cnb = GaussianNB()  # a Gaussian Naive Bayes model
    # Train the model using the training sets
    cnb.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = cnb.predict(X_test)
    accuracy_list.append(metrics.accuracy_score(y_test, y_pred))

print('Five-folder cross validation accuracy is: ', "{:.2f}".format(100*np.mean(accuracy_list))+ '%')
# about 57% accuracy for 5-class classification problem, acceptable but could be improved


