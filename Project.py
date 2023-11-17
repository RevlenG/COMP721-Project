# outstanding players and ouliers code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocessing():
    folder_path = input("Please specify the sentiment file (and its location): ")

    #all relevant textfiles in the folder
    players = folder_path + "/players.txt"
    players_reg_season = folder_path + "/player_regular_season.txt"
    players_reg_season_career = folder_path + "/player_regular_season_career.txt"
    players_playoffs = folder_path + "/player_playoffs.txt"
    players_playoffs_career = folder_path + "/player_playoffs_career.txt"
    players_allstar = folder_path + "/player_allstar.txt"

    file = open(players_reg_season_career, "r")

    df = pd.read_csv(file)

    #remove players that have played less than 50 games
    df = df[df['gp'] >= 50]

    #remove duplicates from the dataframe
    df.drop_duplicates(keep='first', inplace=True)

    #Calculate per-game statistics and create a composite player performance metric
    df['pts/g'] = df['pts'] / df['gp']
    df['reb/g'] = df['reb'] / df['gp']
    df['asts/g'] = df['asts'] / df['gp']
    df['stl/g'] = df['stl'] / df['gp']
    df['blk/g'] = df['blk'] / df['gp']
    df['turnover/g'] = df['turnover'] / df['gp']
    df['pf/g'] = df['pf'] / df['gp']
    df['fg_ratio'] = df['fgm'] / df['fga']
    df['ft_ratio'] = df['ftm'] / df['fta']
    df['tp_ratio'] = np.where(df['tpa'] == 0, 0, df['tpm'] / df['tpa'])
    df['comp_stat'] = 0.3 * df["pts/g"] + 0.25 * df["asts/g"] + 0.15 * df["reb/g"] + 0.15 * df["stl/g"] + 0.15 * df["blk/g"]

    #Statistics of the df
    # print(df.shape)
    # print("\n", df.describe())
    # print("\n", df.info())

    #Checking for null values in the dataset
    # df.isnull().sum().sort_values(ascending=False)
    # df.head

    return df

def elbow_plot(df, sc):
    #This method is used to plot the wcss graph to determine the optimal number of clusters
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df[sc])
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method graph to find the optimal number of clusters
    print("\nView the Elbow Plot in the opened window")
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def silhouette_plot(df, sc):
    silhouette_scores = []

    for i in range (2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df[sc])
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(df[sc], labels)
        silhouette_scores.append(silhouette_avg)

    # Plot Silhouette Scores
    print("\nView the Silhouette Plot in the opened window\n")
    plt.plot(range(2, 11), silhouette_scores)
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

def kmeans_clustering(df, sc, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[sc])

    return df

def outliers(df):
    stats = df.describe()

    iqr_multiplier = 1.5

    for column in ['comp_stat', "pts/g", "reb/g", "asts/g", "stl/g", "blk/g"]:
        q1 = stats.at['25%', column]
        q3 = stats.at['75%', column]
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # Identify outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

        # Print the number of outliers and the firstname and lastname of outliers for the current column
        num_outliers = len(outliers)

        print(f'\nOutliers in {column}: {num_outliers} players')
        print(outliers[['firstname', 'lastname']].to_string(index=False))

def outstanding_players(df, n_clusters):
    best_cluster_id = None
    best_avg_comp_stat = 0

    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        avg_comp_stat = cluster_data['comp_stat'].mean()
        
        print(f'Cluster {cluster_id}: Average comp_stat = {avg_comp_stat:.2f}')
        
        if avg_comp_stat > best_avg_comp_stat:
            best_avg_comp_stat = avg_comp_stat
            best_cluster_id = cluster_id

    print(f'\nCluster {best_cluster_id} has the best players with an average of {best_avg_comp_stat:.2f} with respect to the comp_stat.')

    cluster_data = df[df['cluster'] == best_cluster_id]
    cluster_data = cluster_data.sort_values(by='comp_stat', ascending=False)
    print(f'\nCluster {best_cluster_id}: Top 20 Outstanding Players')
    top_names = cluster_data[["firstname","lastname",]].head(20)
    print(top_names.to_string(index=False))

sc = ["pts/g", "reb/g", "asts/g", "stl/g", "blk/g", "turnover/g", "pf/g", "fg_ratio", "ft_ratio", "tp_ratio"]
n_clusters = 3

df = preprocessing()

# Uncomment the next 2 lines to view the elbow and silhouette plots
# elbow_plot(df, sc)
# silhouette_plot(df, sc)

df = kmeans_clustering(df, sc, n_clusters)

outstanding_players(df, n_clusters)

outliers(df)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# game prediction code

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read in dataset
matches = pd.read_csv(f'Dataset/Game Prediction/matches.txt', header=None, names=['VISITOR', 'VISITOR_PTS', 'HOME', 'HOME_PTS'])

#Preprocess Dataset
matches.dropna()
matches.drop_duplicates(keep='first', inplace=True)
matches['VISITOR'] = matches['VISITOR'].str.replace(' ', '_')
matches['HOME'] = matches['HOME'].str.replace(' ', '_')
matches['RESULT_(HOME_W/L)'] = np.where(matches['VISITOR_PTS']>matches['HOME_PTS'], 0, 1)

# create array of teams in dataset
array_visitor = matches['VISITOR']
array_home = matches['HOME']
array = []
for x in array_visitor:
    array.append(x)
for y in array_home:
    array.append(y)

# clean array of teams in dataset and drop duplicate values
df = pd.DataFrame(array, columns=['Teams'])
df.drop_duplicates(keep='first', inplace=True)
array_new = df['Teams']
team=[]
for x in array_new:
    team.append(x)

# create 2d array of team names and its respective integer label for logistic regression
teams = []
count = 0
for z in team:
    dummy = []
    dummy.append(z)
    dummy.append(count)
    teams.append(dummy)
    count = count + 1

# print sample of teams with respective number
print("Sample of Teams with Their Respective Number Allocations: ")
print(teams[0:5])
print("")
# replace team names in dataset with its respective team value
for x in range(0, 30):
    # print(teams[x][0])
    # print(teams[x][1])
    team_replace = teams[x][0]
    id = teams[x][1]

    matches['VISITOR'].replace(team_replace, id, inplace=True)
    matches['HOME'].replace(team_replace, id, inplace=True)

# print(matches.head(6))

# split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(matches[['VISITOR', 'HOME']], matches['RESULT_(HOME_W/L)'], test_size=0.2, random_state=2)

# train and output the Logistic regression predictions
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
logistic_reg_accuracy = logistic_reg.score(X_test, y_test)
logistic_reg_prediction = logistic_reg.predict(X_test)
print("Logistic Regression Prediction Given Two Teams as as Tuple Input: ")
print("")
print(logistic_reg_prediction)
print("")
print("Logistic Regression Accuracy Score: ", logistic_reg_accuracy)
print("")

# train and output the Naive Bayes predictions and scores
classifier = GaussianNB()  # Use Gaussian Naive Bayes classifier
classifier.fit(X_train, y_train)
naive_bayes_prediction= classifier.predict(X_test)
accuracy = accuracy_score(y_test, naive_bayes_prediction)
precision = precision_score(y_test, naive_bayes_prediction)
recall = recall_score(y_test, naive_bayes_prediction)
f1 = f1_score(y_test, naive_bayes_prediction)
print("Naive Bayes Prediction Given Two Teams as as Tuple Input: ")
print("")
print(naive_bayes_prediction)
print("")
print("Naive Bayes Accuracy Score:", accuracy)
print("Naive Bayes Precision Score:", precision)
print("Naive Bayes Recall Score:", recall)
print("Naive Bayes F1-Score:", f1)
print("")

# train and output the Decision Tree predictions and scores
classifierr = DecisionTreeClassifier()  # Use Decision Tree classifier
classifierr.fit(X_train, y_train)
decision_tree_prediction = classifierr.predict(X_test)
accuracyy = accuracy_score(y_test, decision_tree_prediction)
precisionn = precision_score(y_test, decision_tree_prediction)
recalll = recall_score(y_test, decision_tree_prediction)
f11 = f1_score(y_test, decision_tree_prediction)
print("Decision Tree Prediction Given Two Teams as as Tuple Input: ")
print("")
print(decision_tree_prediction)
print("")
print("Decision Tree Accuracy Score:", accuracyy)
print("Decision Tree Precision Score:", precisionn)
print("Decision Tree Recall Score:", recalll)
print("Decision Tree F1-Score:", f11)
print("")

# Test on new data
new_data = [(5, 21), (10, 3), (6, 9), (12, 2), (16, 6)] # each number represents a team, chosen at random
new_data_test_lr = logistic_reg.predict(new_data)
new_data_test_nb = classifier.predict(new_data)
new_data_test_dt = classifierr.predict(new_data)
print("Logistic Regression Prediction on New Data: ")
print("")
print(new_data_test_lr)
print("")
print("Naive Bayes Prediction on New Data: ")
print("")
print(new_data_test_nb)
print("")
print("Decision Tree Prediction on New Data: ")
print("")
print(new_data_test_dt)
print("")
