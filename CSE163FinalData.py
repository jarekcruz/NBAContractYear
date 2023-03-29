"""
Jarek Cruz
CSE 163 AC
Final Project

This file loads in the data from the zip file, and adjusts it
in order to be ready for processing.
"""


import pandas as pd


fa_list = []
for x in range(2016, 2023):
    df = pd.read_csv('/Users/jarek/NBACSE163/' + str(x) + 'FA.csv')
    df['Contract_Year'] = x
    fa_list.append(df)
fa = pd.concat(fa_list, axis=0, ignore_index=True)
fa = fa[['Player', 'Contract_Year']]


stat_list = []
for x in range(2016, 2023):
    df = pd.read_csv('/Users/jarek/NBACSE163/' + str(x) + 'Stats.csv')
    df['Year'] = x
    stat_list.append(df)
stats = pd.concat(stat_list, axis=0, ignore_index=True)


final = pd.merge(fa, stats, on=['Player'], how='left')
final['PER'] = (final['FG'] * 85.910 + final['STL'] * 53.897 +
                final['3P'] * 51.757 + final['FT'] * 46.845 +
                final['BLK'] * 39.190 + final['ORB'] * 39.190 +
                final['AST'] * 34.677 + final['DRB'] * 14.707 - 2.5 * 17.174 -
                (final['FTA'] - final['FT']) * 20.091 -
                (final['FGA'] - final['FG']) * 39.190 -
                final['TOV'] * 53.897) * (1 / final['MP'])
final = final.drop(['Rk', 'Pos', 'Tm', 'Player-additional'], axis=1)
before = (final['Contract_Year'] == final['Year'] + 1)
contract = (final['Contract_Year'] == final['Year'])
final = final[before | contract]
dups = final.duplicated(subset=['Player', 'Contract_Year'], keep=False)
final = final[dups]
final.to_csv('/Users/jarek/NBACSE163/finaldata.csv', index=False)