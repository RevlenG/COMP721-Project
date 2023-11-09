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

elbow_plot(df, sc)
silhouette_plot(df, sc)

df = kmeans_clustering(df, sc, n_clusters)

outstanding_players(df, n_clusters)
