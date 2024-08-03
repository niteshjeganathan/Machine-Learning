# Importing Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from apyori import apriori

# Importing Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Data Preprocessing
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training Eclat Model
rules = apriori(transactions=transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Displaying Rules
results = list(rules)
print(results)

# Well Organised Dataframe
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsDF = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support']) 
print(resultsDF)

# Sorting Supports
resultsDF = resultsDF.nlargest(n = 10, columns='Support')
print(resultsDF)
