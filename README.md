# NYC_uber_pickup

## Overview
The demand for ride-sharing services in NYC can vary in different areas, time periods, or under different weather conditions. Based on the [Kaggle dataset of NYC Uber Pickups](https://www.kaggle.com/yannisp/uber-pickups-enriched) for 6 months starting from January 2015, we aimed to analyze and predict the demand (measured and represented by the number of pickups) and to analyze the relationships between the number of pickups and variables including weather, location, and several time factors. We try to produce results that could provide insights to help ride-sharing companies like Uber improve the user experience through optimizations in planning.

## Dataset
The dataset was retrieved from [Kaggle](https://www.kaggle.com/yannisp/uber-pickups-enriched). The dataset consists of the hourly data collected from 01/01/2015 to 30/06/2015. Overall, the dataset contains 29,101 instances, each of which has 13 features including the borough, the number of pickups, the temperature, and etc. More detailed feature exploration and data pre-processing can be found in our visualization notebook and report. 

## Research Objective
The goal for this research project is to analyze the pattern of the number of Uber pickups in correlation with various features to provide insights to help Uber make business decisions and resource allocations. We also aim to predict future pickups based on attainable future features. With this goal in mind, we proposed and trained the following models:
* Linear Regression
* Elastic Net
* ARIMA
* Neural Net

To improve the efficiency of our algorthm, we deployed the following performance improving techniques
* line_profiler
* itertools
* parallel programming
* python concurrency

Our final prediction of the pickup numbers achieved a RMSE of only 0.0703 against the test set. 
