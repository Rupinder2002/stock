# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:45:26 2021

@author: ASHUTOSH
"""
from pandas import read_csv
import sys
import math

dataset = read_csv('18-04-2019-TO-16-04-2021RELIANCEALLN.csv', header=0)


def calculate_slope(index, slope_length):
    # print(index,"to",index+slope_length-1)
    return math.degrees(math.atan(
        (dataset.iloc[index + slope_length - 1]['Close Price'] - dataset.iloc[index]['Close Price']) / slope_length))


def calculate_profit(slope_length, input_angle, output_angle):
    profit = 0.0
    buy_indices = []
    sell_indices = []
    buy_price_indices = []
    sell_price_indices = []
    for index, row in dataset.iterrows():
        if index + slope_length <= len(dataset.index):
            if calculate_slope(index, slope_length) >= input_angle:
                # pass
                buy_price = dataset.iloc[index + slope_length - 1]['Close Price']
                buy_indices.append(index + slope_length - 1)
                buy_price_indices.append(buy_price)
                sell_price = 0.0
                for index_sell, row_sell in dataset.iterrows():
                    if index_sell > (index + slope_length - 1):
                        if index_sell + slope_length <= len(dataset.index):
                            if calculate_slope(index_sell, slope_length) <= ((-1) * output_angle):
                                sell_price = dataset.iloc[index_sell + slope_length - 1]['Close Price']
                                sell_indices.append(index_sell + slope_length - 1)
                                sell_price_indices.append(sell_price)
                                break
                profit += (sell_price - buy_price)
    return profit, buy_indices, sell_indices, buy_price_indices, sell_price_indices


slope_angles = [30]
slope_lengths = [20]
max_profit = sys.float_info.min
min_cash = sys.float_info.max
max_slope_length = max_input_angle = max_output_angle = 0
cash = 100000
optimum_buy_indices = []
optimum_sell_indices = []
optimum_buy_price_indices = []
optimum_sell_price_indices = []
for slope_length in slope_lengths:
    for input_angle in slope_angles:
        for output_angle in slope_angles:
            profit, buy_indices, sell_indices, buy_price_indices, sell_price_indices = calculate_profit(slope_length,
                                                                                                        input_angle,
                                                                                                        output_angle)
            if profit > max_profit:
                max_profit = profit
                max_slope_length = slope_length
                max_input_angle = input_angle
                max_output_angle = output_angle
                optimum_buy_indices = buy_indices
                optimum_sell_indices = sell_indices
                optimum_buy_price_indices = buy_price_indices
                optimum_sell_price_indices = sell_price_indices
print('max profit = ', max_profit)
print('optimum slope length = ', max_slope_length)
print('optimum input angle = ', max_input_angle)
print('optimum output angle = ', max_output_angle)
print('optimum buy indices = ', optimum_buy_indices)
print('optimum sell indices = ', optimum_sell_indices)


def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count


cash_list = []
for i in range(494):
    if i in optimum_buy_indices:
        # print('bought')
        cash -= optimum_buy_price_indices[optimum_buy_indices.index(i)]
        cash_list.append(cash)
    elif i in optimum_sell_indices:
        # print('sold')
        times = countX(optimum_sell_indices, i)
        # print(times)
        cash += (optimum_sell_price_indices[optimum_sell_indices.index(i)] * times)

print(min(cash_list))








