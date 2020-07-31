'''This piece of script is for data analysis of file Pork Price.xlsx'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

'''Part 1: Data wrangling and data cleaning'''

'''Load the raw data into python'''
# to run following line of code, please set your IDE working directory to the folder containing the excel file
xlsx = pd.ExcelFile('./Pork Price.xlsx')

'''Wrangle the data into the form where each column represent a dimension'''
Dimension_names = ["Date", "City", "MeatPart", "Market", "Price"]  # there are five dimensions in total
all_data_wrangled = pd.DataFrame(columns=Dimension_names)  # data frame to store all data

for sheets in xlsx.sheet_names:  # iterate through all sheets

    current_sheet = pd.read_excel(xlsx, sheets, header=None)
    nan_sheet = pd.isnull(current_sheet)  # nan boolean table of current sheet
    sheet_wrangled = pd.DataFrame(columns=Dimension_names)  # create new data frame for wrangled sheet

    for rowindex in range(2, current_sheet.shape[0]):  # iterate through rows
        city_index = 0  # for each row, start from first city
        for colindex in range(1, current_sheet.shape[1]):  # iterate through cols
            Date = sheets  # Date
            if nan_sheet.iloc[0, colindex] == False:
                city_index += 1
            City = "No." + str(city_index)  # City
            MeatPart = current_sheet.iloc[rowindex, 0]
            if 'Wet' in current_sheet.iloc[1, colindex]:
                Market = 'Wet Market'
            else:
                Market = 'Supermarket'
            Price = float(current_sheet.iloc[rowindex, colindex])
            currentRow = pd.DataFrame([[Date, City, MeatPart, Market, Price]],
                         columns=Dimension_names)  # Current row, consist of one single data from original sheet
            sheet_wrangled = sheet_wrangled.append(currentRow, ignore_index=True)

    all_data_wrangled = all_data_wrangled.append(sheet_wrangled, ignore_index=True)

'''Check outliers/typos'''
print('Dates are: ', all_data_wrangled['Date'].unique())
print('Cities are: ', all_data_wrangled['City'].unique())
print('MeatParts are: ', all_data_wrangled['MeatPart'].unique())
print('Markets types are: ', all_data_wrangled['Market'].unique())
print("lowest price in all records is: ",  all_data_wrangled.loc[:,'Price'].min())
print("highest price in all records is: ",  all_data_wrangled.loc[:, 'Price'].max())
# They are both reasonable price according to the provided information,
# thus for the following analysis, I will assume there is no typos in this dataset

'''Dealing with missing data'''
total_data_num = all_data_wrangled.shape[0]  # 984 data in total
null_columns = all_data_wrangled.columns[all_data_wrangled.isnull().any()] # only price has nan value
blank_data_num = all_data_wrangled[null_columns].isnull().sum() # there are total 48 nan value in price attribute

# so basically there are two ways of dealing with missing data, drop or fill
# for this analysis, I choose to try filling these blank cells by first building a regression model
# with the data given, test the performance of the model and if the model works well, I'll use the model
# to predict these missing data, if don't, I'll drop the data
drop_na_data = all_data_wrangled.dropna() # first drop all nan data
predictor_drop_na = drop_na_data[['Date', 'City', 'MeatPart', 'Market']]
# use one-hot encoding to deal with these categorical variables
predictor_drop_na = pd.get_dummies(data=predictor_drop_na, drop_first=True)
dependent_drop_na = drop_na_data[['Price']]

X_train, X_test, Y_train, Y_test = train_test_split(predictor_drop_na, dependent_drop_na,
                                                    test_size = .20, random_state = 5)
# In this case, I choose both a lasso regression model and a random forrest model
# other models can also be implemented such as polynomial regression...
lasso_model = linear_model.LassoCV()
lasso_model.fit(X_train, Y_train)
lasso_predicted = lasso_model.predict(X_test)
# calculate the RMSE of this model (Due to the limited time, I don't perform cross validation for this model evaluation)
lasso_rmse = ((Y_test.loc[:,'Price'] - lasso_predicted) ** 2).mean() ** .5 # RMSE is about 8

rf_model = RandomForestRegressor(n_estimators=200, random_state=0)  # random forest model
rf_model.fit(X_train, Y_train)
rf_predict = rf_model.predict(X_test)
rf_rmse = ((Y_test.loc[:,'Price'] - rf_predict.T) ** 2).mean() ** .5 # RMSE is about 6.6, which is acceptable to me

# as a result, I'll use random forest model to fill in the blank data
all_data_isnull = pd.isnull(all_data_wrangled)
predictor_all_data = all_data_wrangled[['Date', 'City', 'MeatPart', 'Market']]
predictor_all_data = pd.get_dummies(data=predictor_all_data, drop_first=True)

for rowi in range(total_data_num):
    if all_data_isnull.loc[rowi, 'Price']:  # if the current record is nan
        current_pred = predictor_all_data.iloc[rowi,:]
        current_pred = pd.DataFrame(current_pred).transpose()
        all_data_wrangled.at[rowi, 'Price'] = int(rf_model.predict(current_pred)[0]*100)/100

all_data_wrangled.to_excel('./Wrangled_data_pork_price.xlsx')

'''Part 2: Data Visualization and Inspection'''

'''First, check relationships between some important variables to build a basic concept'''
data = all_data_wrangled  # simplify the data name

time_meat_price = data[['Date','MeatPart','Price']].groupby(['Date', 'MeatPart']).mean()
time_city_price = data[['Date','City','Price']].groupby(['Date', 'City']).mean()
figure_one, axes_one = plt.subplots(1,2, sharey=True, figsize = (20,4))
figure_one.suptitle('Overall Means of each timestamp given each MeatPart/City', fontsize = 20)
sbn.heatmap(time_meat_price.unstack(), cmap = 'YlGnBu', ax = axes_one[0],
            annot=True, cbar = False, xticklabels=data['MeatPart'].unique())
sbn.heatmap(time_city_price.unstack(), cmap = 'YlGnBu', ax = axes_one[1],
            annot=True, xticklabels=data['City'].unique())
axes_one[0].set(xlabel = 'MeatPart', ylabel = 'Date')
axes_one[1].set(xlabel = 'City', ylabel = '')
plt.subplots_adjust(wspace=0.1, hspace=0)
plt.subplots_adjust(left = 0.05, bottom = 0.15,
                    right = 0.95, top = 0.9)
plt.show()

'''Now, we want to dig into variable City, MeatPart and Market given Date and Price'''
city_num = len(data['City'].unique())
part_num = len(data['MeatPart'].unique())

figure_two, axes_two = plt.subplots(nrows = 1, ncols = 9, figsize = (40,15), sharey=True)
figure_two.suptitle('Price versus Date given different market for each city ', fontsize = 20)
for axes_i in range(city_num):
    city_i = 'No.' + str(axes_i+1)
    current_city = data.loc[data['City'] == city_i]

    # boxplot, each subplot represents a city, different color represent wet market or supermarket
    sbn.boxplot(x = 'Date', y = 'Price', hue = 'Market', data = current_city,
            palette = {'Wet Market': 'orangered', 'Supermarket':'deepskyblue'},
                ax = axes_two[axes_i])
    # add a line of trending of each mean
    sbn.lineplot(x='Date', y='Price', hue='Market', sort=False, data = current_city,
                 err_style=None, ax = axes_two[axes_i], markers=True, lw = 2,
                 palette={'Wet Market': 'black', 'Supermarket': 'black'}, style='Market')

    axes_two[axes_i].set(title = 'City ' + city_i, xlabel = 'Date', ylabel = 'Price' if axes_i == 0 else '' ,
                         xticklabels = ['W0', 'W2', 'W4', 'W6'])
    axes_two[axes_i].legend(loc='best')

plt.subplots_adjust(wspace=0, hspace=0)
plt.subplots_adjust(left = 0.05, bottom = 0.1,
                    right = 0.95, top = 0.9)
plt.show()

figure_three, axes_three = plt.subplots(nrows = 2, ncols = 6, figsize = (30,20), sharex=True, sharey=True)
figure_three.suptitle('Price versus Date given different market for each meat part ', fontsize = 20)
axes_i = 0
for part_i in data['MeatPart'].unique():
    current_part = data.loc[data['MeatPart'] == part_i]
    row_i = axes_i//6
    col_i = int(axes_i%6)
    # boxplot, each subplot represents a meat part, different color represent wet market or supermarket
    sbn.boxplot(x = 'Date', y = 'Price', hue = 'Market', data = current_part,
            palette = {'Wet Market': 'orangered', 'Supermarket':'deepskyblue'},
                ax = axes_three[row_i, col_i])
    # add a line of trending of each mean
    sbn.lineplot(x='Date', y='Price', hue='Market', sort=False, data = current_part,
                 err_style=None, ax = axes_three[row_i, col_i], markers=True, lw = 2,
                 palette={'Wet Market': 'black', 'Supermarket': 'black'},
                 style='Market')

    axes_three[row_i, col_i].set(title = 'MeatPart ' + part_i, xlabel = 'Date', ylabel = 'Price' if axes_i == 0 or 6 else '',
                         xticklabels = ['W0', 'W2', 'W4', 'W6'])
    axes_three[row_i, col_i].legend(loc='best')
    axes_i += 1
plt.subplots_adjust(wspace=0, hspace=0)
plt.subplots_adjust(left = 0.05, bottom = 0.1,
                    right = 0.95, top = 0.9)
plt.show()


'''Part 3: Detailed analysis'''

'''First, some statistics about the mean/standard deviation of price given different MeatParts and Market types'''
meat_part_dimension_names = ["Part", "min_mean", "max_mean", "min_Date", "max_Date", "min_sd_Date", "max_sd_Date",
                             "overall_rise_ratio", "supermarket_rise_ratio", "wet_market_rise_ratio",
                             "supermarket_first_two_weeks_variation", "wetmarket_first_two_weeks_variation"]
# there are twelve dimensions in total
meatpart_analysis_table = pd.DataFrame(columns= meat_part_dimension_names)  # data frame to store statistics

for part_i in data['MeatPart'].unique():
    current_part = data.loc[data['MeatPart'] == part_i]
    current_mean = current_part.groupby(['Date']).mean()
    current_sd = current_part.groupby(['Date']).std()
    mean_min = float("{:.2f}".format(current_mean['Price'].min()))
    mean_max = float("{:.2f}".format(current_mean['Price'].max()))
    min_date = current_mean['Price'].idxmin()
    max_date = current_mean['Price'].idxmax()
    overall_rise_ratio = "{:.2f}".format(100*(mean_max - mean_min)/mean_min)+ '%'
    min_sd = current_sd['Price'].idxmin()
    max_sd = current_sd['Price'].idxmax()

    date_market = current_part.groupby(['Date', 'Market'])['Price'].mean().unstack()
    overall_super_rise_ratio = "{:.2f}".format(100*(date_market['Supermarket'].max() - date_market['Supermarket'].min())/date_market['Supermarket'].min())+ '%'
    overall_wet_rise_ratio = "{:.2f}".format(100*(date_market['Wet Market'].max() - date_market['Wet Market'].min())/date_market['Wet Market'].min())+ '%'
    super_rise_ratio_W2_W0 = "{:.2f}".format(100*(date_market.loc['Oct.5th', 'Supermarket'] - date_market.loc['Sep.21th', 'Supermarket'])/date_market.loc['Sep.21th', 'Supermarket'])+ '%'
    wet_rise_ratio_W2_W0 = "{:.2f}".format(100*(date_market.loc['Oct.5th', 'Wet Market'] - date_market.loc['Sep.21th', 'Wet Market'])/date_market.loc['Sep.21th', 'Wet Market'])+ '%'

    currentRow = pd.DataFrame([[part_i, mean_min, mean_max, min_date, max_date, min_sd, max_sd,
                                overall_rise_ratio, overall_super_rise_ratio, overall_wet_rise_ratio,
                                super_rise_ratio_W2_W0, wet_rise_ratio_W2_W0]],
                              columns=meat_part_dimension_names)
    meatpart_analysis_table = meatpart_analysis_table.append(currentRow, ignore_index=True)

'''Second, some statistics about the mean/standard deviation of price given different Cities and Market types'''
city_dimension_names = ["City", "min_mean", "max_mean", "min_Date", "max_Date", "min_sd_Date", "max_sd_Date",
                             "overall_rise_ratio", "supermarket_rise_ratio", "wet_market_rise_ratio",
                             "supermarket_first_two_weeks_variation", "wetmarket_first_two_weeks_variation"]
# there are twelve dimensions in total
city_analysis_table = pd.DataFrame(columns= city_dimension_names)  # data frame to store statistics

for city in range(city_num):
    city_i = 'No.' + str(city+1)
    current_city = data.loc[data['City'] == city_i]
    current_mean = current_city.groupby(['Date']).mean()
    current_sd = current_city.groupby(['Date']).std()
    mean_min = float("{:.2f}".format(current_mean['Price'].min()))
    mean_max = float("{:.2f}".format(current_mean['Price'].max()))
    min_date = current_mean['Price'].idxmin()
    max_date = current_mean['Price'].idxmax()
    overall_rise_ratio = "{:.2f}".format(100*(mean_max - mean_min)/mean_min)+ '%'
    min_sd = current_sd['Price'].idxmin()
    max_sd = current_sd['Price'].idxmax()

    date_market = current_city.groupby(['Date', 'Market'])['Price'].mean().unstack()
    overall_super_rise_ratio = "{:.2f}".format(100*(date_market['Supermarket'].max() - date_market['Supermarket'].min())/date_market['Supermarket'].min())+ '%'
    overall_wet_rise_ratio = "{:.2f}".format(100*(date_market['Wet Market'].max() - date_market['Wet Market'].min())/date_market['Wet Market'].min())+ '%'
    super_rise_ratio_W2_W0 = "{:.2f}".format(100*(date_market.loc['Oct.5th', 'Supermarket'] - date_market.loc['Sep.21th', 'Supermarket'])/date_market.loc['Sep.21th', 'Supermarket'])+ '%'
    wet_rise_ratio_W2_W0 = "{:.2f}".format(100*(date_market.loc['Oct.5th', 'Wet Market'] - date_market.loc['Sep.21th', 'Wet Market'])/date_market.loc['Sep.21th', 'Wet Market'])+ '%'

    currentRow = pd.DataFrame([[city_i, mean_min, mean_max, min_date, max_date, min_sd, max_sd,
                                overall_rise_ratio, overall_super_rise_ratio, overall_wet_rise_ratio,
                                super_rise_ratio_W2_W0, wet_rise_ratio_W2_W0]],
                              columns=city_dimension_names)
    city_analysis_table = city_analysis_table.append(currentRow, ignore_index=True)

with pd.ExcelWriter('analysis_statistcs.xlsx') as writer:
    meatpart_analysis_table.to_excel(writer, sheet_name='meatpart_analysis')
    city_analysis_table.to_excel(writer, sheet_name='city_analysis')

