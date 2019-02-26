import re
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style = 'white', context = 'notebook')
sns.set_palette(sns.diverging_palette(220, 20, n=7))

def handle_Pclass(train, test):
    data = concat((train, test), axis = 0)

    data["Pclass"] = data["Pclass"].astype("category")
    data = pd.get_dummies(data, columns = ["Pclass"],prefix="Pc")

    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]

    return train, test

def one_hot_embarked(train, test, plot = False):
    """
    Add dummy variables for each categorical type of Embarkment (including NAs)
    """
    data = concat((train, test), axis = 0)

    if plot:
        data_plot = data.loc[data['Survived'].isin([0, 1])]
        fig = plt.figure(figsize = (18, 8))
        ax_1 = fig.add_axes([0, 0.5, 0.5, 0.4])
        ax_1 = sns.countplot(x = "Embarked", hue = 'Survived', data = data_plot)
        ax_1.set_title('')
        ax_1.set_ylabel('Raw count')
        ax_1.set_xlabel('Port of embarkation')

    dummies = get_dummies(data['Embarked'].fillna(value = 'S'))
    dummies = dummies.rename(columns=lambda x: 'Embarked_' + str(x))
    data = concat([data, dummies], axis = 1)
    data = data.drop(['Embarked'], axis = 1)

    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]

    return train, test

def one_hot_name(train, test, plot = False):
    """
    Add number of names and do one hot encoding of the titles (with some grouping done).
    """
    data = concat((train, test), axis = 0, sort=True)

    # How many different names do they have?
    #print(data)
    #print(data['Name'].iloc[0:4] )
    #print(type(data['Name']))
    #print(data['Name'].dtype)
    #for x in data['Name']:
        #print(x)
        #data['NumberNames']=len(str(x).split(' '))
    #data['NumberNames'] = data['Name'].map(lambda x: len(re.split(' ',x)))

    #if plot:
    #    fig = plt.figure(figsize = (18, 8))
     #   ax_1 = fig.add_axes([0, 0.5, 0.5, 0.4])
      #  ax_1 = sns.countplot(x = "NumberNames", hue = 'Survived', data = data.loc[data['Survived'].isin([0, 1])])
       # ax_1.set_title('Number of names')
        #ax_1.set_ylabel('Raw count')
        #ax_1.set_xlabel(' ')

    #data.loc[data['NumberNames'] <= 5, 'NumberNames'] = 0
    #data.loc[data['NumberNames'] > 5, 'NumberNames'] = 1
    #data['NumberNames'] = np.where(data['NumberNames'] == 0, 'Low','High')

    #if plot:
    #    data_plot = data.loc[data['Survived'].isin([0, 1])]
    #    ax_2 = fig.add_axes([0, 0, 0.5, 0.4])
    #    ax_2 = sns.countplot(x = "NumberNames", hue = 'Survived', data = data_plot)
    #    ax_2.set_ylabel('Count after transformation')
    #    ax_2.set_xlabel('Number of names')

    #data['NumberNames'] = np.where(data['NumberNames'] == 'Low', 0, 1)
    #data['NumberNames'] = data['NumberNames'].astype(int)

    # Compile a regular expression pattern into a regular expression object (?)
    data['Title'] = Series([(i).split(',')[1].split('.')[0].strip() for i in data['Name']])

    if plot:
        fig = plt.figure(figsize = (18, 8))
        data_plot = data.loc[data['Survived'].isin([0, 1])]
        ax_3 = fig.add_axes([0.6, 0.5, 0.5, 0.4])
        ax_3 = sns.countplot(x = "Title", hue = 'Survived', data = data_plot)
        ax_3.set_title('Titles occurences')
        ax_3.set_ylabel(' ')
        ax_3.set_xlabel(' ')

    # Group low-occuring, related titles together
    data.loc[data['Title'].isin(['Jonkheer', 'Sir', 'Dona', 'Lady', 'the Countess']), 'Title'] = 'Nobles'
    data.loc[data['Title'].isin(['Ms','Mlle']), 'Title'] = 'Miss'
    data.loc[data['Title'] == 'Mme', 'Title'] = 'Mrs'
    data.loc[data['Title'].isin(['Capt', 'Don', 'Major', 'Col']), 'Title'] = 'Military'

    # Group doctor and
    data.loc[data['Title'].isin(['Military', 'Dr']), 'Title'] = 'Officer'

    if plot:
        data_plot = data.loc[data['Survived'].isin([0, 1])]
        ax_4 = fig.add_axes([0.6, 0, 0.5, 0.4])
        ax_4 = sns.countplot(x = "Title", hue = 'Survived', data = data_plot)
        ax_4.set_ylabel(' ')
        plt.show()

    # Group doctor and
    data.loc[data['Title'].isin(['Officer', 'Rev', 'Nobles']), 'Title'] = 'Rare'

    # Build binary features
    data = concat([data, get_dummies(data['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis = 1)

    # Remove original title
    data = data.drop(['Title'], axis = 1)
    data = data.drop(['Name'], axis = 1)

    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]

    return(train, test)

def handle_cabin(train, test, plot = False):

    data = concat((train, test), axis = 0)

    # Replace missing values with "U0"
    data.loc[data.Cabin.isnull(), 'Cabin'] = 'U0'

    # Dummy encoding of the deck
    data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data['Deck'] = data['Deck'].map(lambda x: ''.join(set(x)))
    data['Deck'] = data['Deck'].map(lambda x: x.replace(" ", ""))

    if plot:
        # Countplot of the titles
        fig = plt.figure(figsize = (18, 8))
        ax = fig.add_axes([0, 0, 0.45, 0.45])
        ax = sns.countplot(x = "Deck", hue = 'Survived', data = data.iloc[:train.shape[0],:])
        ax.set_title('Deck')
        ax.set_ylabel('Count')

    data.loc[data['Deck'].isin(['T', 'GF', 'U']), 'Deck'] = 'Low'
    data.loc[data['Deck'].isin(['A','C', 'G']), 'Deck'] = 'Medium'
    data.loc[data['Deck'].isin(['EF', 'F', 'D', 'E', 'B']), 'Deck'] = 'High'

    if plot:
        # Countplot of the titles
        data_plot = data.loc[data['Survived'].isin([0, 1])]
        ax_1 = fig.add_axes([0.5, 0, 0.45, 0.45])
        ax_1 = sns.countplot(x = "Deck", hue = 'Survived', data = data_plot)
        ax_1.set_title('Deck')
        ax_1.set_ylabel('Count after')

    decks = get_dummies(data['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
    data = concat([data, decks], axis=1)

    # Add number of cabins
    data["NumberCabins"] = data["Cabin"].map(lambda x: len(x.split(' ')))
    #data['NumberCabins'] = np.where(data['NumberCabins'] >1, 1, 0)

    if plot:
        # Countplot of the titles
        ax_1 = fig.add_axes([0.5, 0, 0.45, 0.45])
        ax_1 = sns.countplot(x = "NumberCabins", hue = 'Survived', data = data.iloc[:train.shape[0],:])
        ax_1.set_title('Number of Cabins')
        ax_1.set_ylabel('Count after')

    # Room number - lower numbers were towards the front of the boat, and higher numbers towards the rear
    data['Room'] = data['Cabin'].map(lambda x: re.findall(r'\d+', x))
    data['Room'] = data['Room'].map(lambda x: x[0] if len(x)>0 else 0)
    data['Room'] = data['Room'].map(lambda x: float(x))
    data['Room'].fillna((data['Room'].median()), inplace=True)

    # Remove original feature
    data = data.drop(['Cabin', 'Deck'], axis = 1)

    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]

    return(train, test)

# for feature in train['Deck'].unique():
#     percentage = train.loc[train['Deck'] == feature]['Survived'].sum()/float(train.loc[train['Deck'] == feature].shape[0])
#     print(feature + ': %.2f of survival'%percentage)

def process_ticket(train, test, plot = False):
    '''
    The objective of this function is extract te prefix and the number of each ticket.
    '''
    # Concatenate the data frame
    data = concat((train, test), axis = 0)

    # Extract and massage the ticket prefix
    data['TicketPrefix'] = data['Ticket'].map(lambda x : getTicketPrefix(x.upper()))
    data['TicketPrefix'] = data['TicketPrefix'].map(lambda x: re.sub('[.?/?]', '', x))
    data['TicketPrefix'] = data['TicketPrefix'].map(lambda x: re.sub('STON', 'SOTON', x))

    if plot:
        fig = plt.figure(figsize = (18, 8))
        ax_1 = fig.add_axes([0, 0.5, 1, 1])
        ax_1 = sns.countplot(x = "TicketPrefix", hue = 'Survived', data = data.iloc[0:train.shape[0]])
        ax_1.set_title('Prefix of the ticket')
        ax_1.set_ylabel('Raw count')
        ax_1.set_xlabel(' ')

    for feat in data['TicketPrefix'].unique():
        # If not enough informations, we delete.
        if data.loc[data['TicketPrefix'] == feat].shape[0]/float(data.shape[0])<0.05:
            data.loc[data['TicketPrefix'] == feat, 'TicketPrefix'] = 'A'

        data.loc[data['TicketPrefix'].isin(['CA', 'TicketPrefix']), 'TicketPrefix'] = 'A'

    if plot:
        fig = plt.figure(figsize = (18, 8))
        ax_1 = fig.add_axes([0, 0.5, 1, 1])
        ax_1 = sns.countplot(x = "TicketPrefix", hue = 'Survived', data = data.iloc[0:train.shape[0]])
        ax_1.set_title('Prefix of the ticket')
        ax_1.set_ylabel('New count')
        ax_1.set_xlabel(' ')

    # Create binary features for each prefix
    prefixes = get_dummies(data['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    data = concat((data, prefixes), axis=1)

    # Drop raw data
    data = data.drop(['TicketPrefix', 'Ticket'], axis=1)

    # De-concatenate
    train, test = data.iloc[0:train.shape[0],:], data.iloc[train.shape[0]:,:]
    return(train, test)

def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z./]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'

def getTicketNumber(ticket):
    match = re.compile("([d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

