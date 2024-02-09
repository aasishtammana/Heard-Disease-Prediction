import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlation(dataframe):
    """
    Calculate correlation, covariance, and highly correlated features.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
    """
    # Extract column names from the dataframe
    column_names = dataframe.columns.tolist()
    
    # Get the number of features in the dataframe
    num_features = dataframe.shape[1]
    
    # Calculate the correlation matrix rounded to 2 decimal places
    correlation_matrix = dataframe.corr().round(2)
    
    # Calculate the covariance matrix rounded to 2 decimal places
    covariance_matrix = dataframe.cov().round(2)
    
    print('Correlation Matrix:')
    print(correlation_matrix)
    print('\n')
    
    print('Covariance Matrix:')
    print(covariance_matrix)
    print('\n')
    
    # Create a matrix of absolute correlations for feature selection
    abs_corr_matrix = dataframe.corr().abs()
    
    # Select only the upper triangle of the matrix
    upper_triangle = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(np.bool))
    
    # Select the top 10 highly correlated feature pairs
    top_correlations = upper_triangle.stack().nlargest(10)
    
    print('Top 10 Highly Correlated Feature Pairs:')
    print(top_correlations)
    print('\n')
    
    # Initialize arrays to store correlation information
    array_index = np.zeros(num_features)
    array_cor = np.zeros(num_features)
    array_cor_temp = np.zeros(num_features - 1)
    
    # Remove the diagonal elements (which are 1) from the correlation matrix
    array = np.identity(num_features)
    diagonal_removed_matrix = correlation_matrix - array
    
    # Find the highest correlation for each variable with every other variable and store it in arrays
    for i in range(num_features):
        temp = diagonal_removed_matrix.iloc[[i]].to_numpy()
        index_value = np.argmax(np.abs(temp))
        array_index[i] = index_value
        array_cor[i] = round(temp[0][index_value], 2)
        if i != num_features - 1:
            array_cor_temp[i] = temp[0][num_features - 1]
    
    # Create a table of correlations with the 'a1p2' variable
    correlation_table = list(zip(column_names, array_cor_temp))
    df_correlation = pd.DataFrame(correlation_table, index=list(range(num_features - 1)), columns=['Variable', 'Correlation with a1p2'])
    
    print('Correlation Matrix with a1p2:')
    print(df_correlation)
    print('\n')
    
    # Create a table of variables that are highly correlated with each other
    correlated_variable_names = [column_names[int(array_index[i])] for i in range(num_features)]
    correlation_pairs = list(zip(column_names, correlated_variable_names, array_cor))
    df_correlation_pairs = pd.DataFrame(correlation_pairs, index=list(range(num_features)), columns=['Variable 1', 'Variable 2', 'Correlation'])
    
    print('Highest Correlation of Each Variable:')
    print(df_correlation_pairs)
    print('\n')

def create_pair_plots(df):
    """
    Create pair plots of the data.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    """
    sns.pairplot(df, height=2.5, hue='a1p2', markers=['+', 'x'])
    plt.show()

# Main code
df = pd.read_csv('heart1.csv')
calculate_correlation(df)
create_pair_plots(df)
