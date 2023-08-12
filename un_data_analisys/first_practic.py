import pandas as pd
import numpy as np
import seaborn as sb

data = pd.read_csv('./un_data_analisys/UNdata_Export_20230807_135634988.csv', index_col='Quantity')

# print(data.head(5))
# data_new = data.drop(['Year'])
# print(data_new)
# sb.heatmap(data)

sb.barplot(x=data['Quantity Name'], y=data['Year']) # вызываем функцию barplot()