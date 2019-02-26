import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def continuous_features(train, test = None, plot = True):

    # Concatenate train / test data
    data = pd.concat((train, test), axis = 0, sort = True)

    # Transforming heavily skewed data into a more gaussian distribution
    data['LogFare'] = np.log(data['Fare'] + 1)

    # De-concatenate
    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]

    if plot:

        plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], stacked=True, bins=30, label=['Survived','Dead'])
        plt.xlabel('Age')
        plt.ylabel('Number of passengers')
        plt.legend()
        plt.show()

        plt.hist([train[train['Survived']==1]['LogFare'], train[train['Survived']==0]['LogFare']], stacked=True, bins=30, label=['Survived','Dead'])
        plt.xlabel('Fare')
        plt.ylabel('Number of passengers')
        plt.legend()
        plt.show()

        plt.hist([train[train['Survived']==1]['Room'], train[train['Survived']==0]['Room']], stacked=True, bins=30, label=['Survived','Dead'])
        plt.xlabel('Room number')
        plt.ylabel('Number of passengers')
        plt.legend()
        plt.show()

    # Standard scaler
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

    # Scale only numeric values
    train_num = train[['Age', 'LogFare', 'Room', 'Family']].copy()
    test_num = test[['Age', 'LogFare', 'Room', 'Family']].copy()

    # Convert the scaled arrays into pandas DataFrame
    col = train_num.columns.values
    train.loc[:, train_num.columns.values] = pd.DataFrame(scaler.fit_transform(train_num), columns = col)
    test.loc[:, train_num.columns.values] = pd.DataFrame(scaler.transform(test_num), columns = col)

    return(train, test)

def basic_preprocessing(train, test):

    # Concatenate train and test set
    data = pd.concat((train, test), axis = 0)

    # Sex
    data['Sex'] = np.where(data['Sex'] == 'male', 1, 0)
    data['Sex'] = data['Sex'].astype('uint8')

    # Family
    data['Family'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = np.where(data['Family'] == 1, 1, 0)

    # Drop unnecessary variables
    data = data.drop(['SibSp', 'Parch'], axis = 1)

    # De-concatenate
    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]

    return(train, test)

def outliers_handling(train, k = 5, plot = False):
    '''
    Find the k biggest outliers (accordign to the Local Outlier Factor) and drop them.
    '''

    # Compute LOF score
    lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30)
    lof.fit(train)
    train['lof_score'] = lof.negative_outlier_factor_

    if plot:
        # Density Plot and Histogram of all arrival delays
        sns.distplot(np.abs(train['lof_score']), hist=True, kde=True,
        color = 'darkblue',
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 4}).set_title('Local Outlier Factor density and histogram before')
        plt.show()

    # Find the k_biggest training observations that have the lowest negative score
    idx_to_drop = train.sort_values(['lof_score'], ascending = True).index
    idx_to_drop = np.asarray(idx_to_drop)
    idx_to_drop = idx_to_drop[0:k]

    idx_k_first = pd.Index(idx_to_drop)

    # Drop them
    train = train.drop(idx_k_first, axis = 0)

    if plot:
        # Density Plot and Histogram of all arrival delays
        sns.distplot(np.abs(train['lof_score']), hist=True, kde=True,
        color = 'darkblue',
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 4}).set_title('Local Outlier Factor density and histogram after')
        plt.show()

    train = train.drop(['lof_score'], axis = 1)

    return(train)

def select_k_lowest(columns, clf, k = 10):
    idx = np.argpartition(lof.negative_outlier_factor_, -20)
    return(columns[idx[:k]])


def fill_age(row,grouped_median_train_data):

    condition = (
        (grouped_median_train_data['Sex'] == row['Sex']) & 
        (grouped_median_train_data['Title_Master'] == row['Title_Master']) &
        (grouped_median_train_data['Title_Miss'] == row['Title_Miss']) &
        (grouped_median_train_data['Title_Mr'] == row['Title_Mr']) &
        (grouped_median_train_data['Title_Mr'] == row['Title_Mr']) &
        (grouped_median_train_data['Title_Mrs'] == row['Title_Mrs']) &
        (grouped_median_train_data['Title_Rare'] == row['Title_Rare']) &
        (grouped_median_train_data['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train_data[condition]['Age'].values[0]

def processing_age_features(data,num_train_obs):

    grouped_train_data = data.iloc[:num_train_obs].groupby(['Sex','Pclass','Title_Master', 'Title_Miss',
       'Title_Mr', 'Title_Mrs', 'Title_Rare'])
    grouped_median_train_data = grouped_train_data.median()
    grouped_median_train_data = grouped_median_train_data.reset_index()[['Sex', 'Pclass', 'Title_Master', 'Title_Miss',
       'Title_Mr', 'Title_Mrs', 'Title_Rare', 'Age']]
    
    # a function that fills the missing values of the Age variable
    data['Age'] = data.apply(lambda row: fill_age(row,grouped_median_train_data) if np.isnan(row['Age']) else row['Age'], axis=1)
    return data
