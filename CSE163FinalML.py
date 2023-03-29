"""
Jarek Cruz
CSE 163 AC
Final Project

This file calculates key growth statistics, graphs them and
then applies 2 machine learning pipelines in order to investigate
correlation in generating the contract year effect"""

import statistics
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from CSE163FinalData import final
sns.set()
np.random.seed(66)


final_shift = final.shift(-1)


def calc_avg_change(stat: str) -> float:
    """
    Generates the mean difference in
    a certain statistic amongst all players in a contract year"""
    diff = final_shift[stat] - final[stat]
    diff = diff.dropna()
    diff = diff.iloc[::2]
    return statistics.mean(diff)


avg_PER_change = calc_avg_change('PER')
avg_PTS_change = calc_avg_change('PTS')
avg_AST_change = calc_avg_change('AST')
avg_TRB_change = calc_avg_change('TRB')
print(avg_PTS_change, avg_AST_change, avg_TRB_change, avg_PER_change)
PER_diff = final_shift['PER'] - final['PER']
PER_diff = PER_diff.dropna()
PER_diff = PER_diff.iloc[::2]
df = pd.merge(final, PER_diff, left_index=True, right_index=True)
df = df.rename(columns={'PER_x': 'PER'})
df = df.rename(columns={'PER_y': 'PER_diff'})
df = df.dropna()


sns.relplot(
    data=df, x="Age", y="PER_diff", height=4,
)
plt.title("Age vs PER difference in contract years")
plt.show()
plt.save('Age vs PER shift')


ml_df = df.drop(columns=['Player', 'Year', 'Contract_Year'])
features = ml_df.loc[:, ml_df.columns != 'PER']
labels = ml_df['PER']
features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
DTR = DecisionTreeRegressor()
RFR = RandomForestRegressor()


def train_model(model: any) -> None:
    """
    Trains a model according to a given algorithm
    """
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    print('Train MSE:', mean_squared_error(labels_train, train_predictions))
    test_predictions = model.predict(features_test)
    print('Test MSE:', mean_squared_error(labels_test, test_predictions))


train_model(DTR)
train_model(RFR)
