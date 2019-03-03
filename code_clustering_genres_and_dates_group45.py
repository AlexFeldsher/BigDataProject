import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pickle
from sklearn import preprocessing
import pandas as pd
import tarfile

pd.set_option('display.max_columns', None)  # print all columns
pd.set_option('display.width', 200)  # use more console width
np.set_printoptions(threshold=np.inf)  # print everything

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=20)


def print_feature_values_of_clusters(num_clusters, clusters_data):
    for i in range(num_clusters):
        print('\n')
        print("Cluster: ", i)
        print("Cluster Size: ", len(clusters_data[i]))
        print("Cluster Mean Rating: ", clusters_data[i].loc[:, 'general rating_mean'].mean())
        for feature in clusters_data[0].keys():
            print(feature, " Mean Value: ", clusters_data[i].loc[:, feature].mean())


def plot_features_cluster_center_values(num_clusters, clusters_data, cluster_centers, feature_map, colors):
    for j in range(len(feature_map)):
        for i in range(num_clusters):
            print("Cluster ", i, " color: ", colors[i],
                  " |  cluster mean_rating: ", clusters_data[i].loc[:, 'general rating_mean'].mean(),
                  " |  center value of ", feature_map[j], ": ", cluster_centers[i][j])
            # plot x-axis: cluster i_index center value at feature j_index, plot y-axis: cluster i_index mean rating
            plt.scatter(cluster_centers[i][j],
                        clusters_data[i].loc[:, 'general rating_mean'].mean(), s=250, c=colors[i])
        plt.xlim(-0.1, 1.1)  # limit x-axis plot to relevant range
        plt.ylim(6.4, 7.6)  # limit y-axis plot to relevant range
        plt.xlabel(feature_map[j], fontsize=25)  # name x-axis
        plt.ylabel('rating', fontsize=25)  # name y-axis
        plt.show()
        print("\n")


# produces a 3d plot that shows the relationship between pairs of features and the cluster rating
# the first feature is always selected, then paired up with any other feature of the selected batch
def plot_feature_relationships(num_clusters, clusters_data, cluster_centers, feature_map, colors):
    for j in range(1, len(feature_map)):
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(num_clusters):
            x_vals = cluster_centers[i][0]  # 0 is the index of Drama
            y_vals = cluster_centers[i][j]
            z_vals = clusters_data[i].loc[:, 'general rating_mean'].mean()
            ax.scatter(x_vals, y_vals, z_vals, s=250, c=colors[i])

            print("Cluster ", i, " color: ", colors[i], "  |  mean_rating: ", z_vals)
            print("X-Axis: Cluster Center ", i, " value of ", feature_map[0], ": ", cluster_centers[i][0])  # 0 is the index of Drama
            print("Y-Axis: Cluster Center ", i, " value of ", feature_map[j], ": ", cluster_centers[i][j])
            print('\n')

        ax.set_xlabel(feature_map[0], fontsize=25)
        ax.set_ylabel(feature_map[j], fontsize=25)
        ax.set_zlabel('mean rating')
        plt.show()
        print("\n")


def plot_cluster_points(num_clusters, clusters_data, colors):
    for column in clusters_data[0].keys():
        print("plotting cluster points of feature: ", column, " against their rating_mean")
        for i in range(num_clusters):
            plt.scatter(clusters_data[i][column],
                        clusters_data[i]["general rating_mean"], s=20, c=colors[i])
        plt.xlabel(column, fontsize=25)  # name x-axis
        plt.ylabel('rating', fontsize=25)  # name y-axis
        plt.show()


def create_feature_map(clean_data):
    feature_map = dict()
    for i, column in enumerate(clean_data.columns):
        print(i, column)
        feature_map[i] = column
    return feature_map


def normalize_features(clean_data):
    rating_mean = clean_data['general rating_mean']
    del clean_data['general rating_mean']

    # normalize all features (to prevent different value ranges issues when clustering)
    x = clean_data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    clean_data = pd.DataFrame(x_scaled, index=clean_data.index, columns=clean_data.columns)

    clean_data['general rating_mean'] = rating_mean

    return clean_data


def clean_and_filter_data(data):
    # filter only genres features (Drama / Comedy / Romance etc...)
    genre_features = list(filter(lambda x: 'genre' in x, data.columns))
    # filter only general features (e.g one value representing some aspect of an entire show)
    general_features = list(filter(lambda x: 'general' in x, [feature for feature in data.columns]))
    all_features = general_features + genre_features

    # filter the data to include only these features for the following examinations
    clean_data = data[all_features]

    # clean data from unreasonable values of some features
    sane_runtime = clean_data['general runtime'] < 300
    enough_votes = clean_data['general total_votes'] > 0
    reasonable_year = clean_data['general year'] < 2020
    correct_release_date = clean_data['general release_date'] > 1167609600
    clean_data = clean_data[sane_runtime]
    clean_data = clean_data[enough_votes]
    clean_data = clean_data[reasonable_year]
    clean_data = clean_data[correct_release_date]

    # filter out points with missing information about some features (a feature has the value -1 if it isn't found)
    for feature in clean_data.columns:
        clean = clean_data[feature] > -1
        clean_data = clean_data[clean]

    return clean_data, all_features, general_features, genre_features


def examine_features(clean_data, num_clusters):
    # save rating and votes features before clustering:
    rating_mean = clean_data['general rating_mean']

    # remove ratings and number of votes from data before clustering:
    del clean_data['general rating_mean']
    del clean_data['general rating_median']
    del clean_data['general total_votes']

    cluster_labels, cluster_centers, feature_map = run_kmeans(clean_data, num_clusters)

    # re-set columns after clustering:
    clean_data['general rating_mean'] = rating_mean

    return cluster_labels, cluster_centers, feature_map


def run_kmeans(clean_data, num_clusters):
    # create kmeans object:
    kmeans = KMeans(n_clusters=num_clusters)

    # fit kmeans object to data and get cluster labels
    cluster_labels = kmeans.fit(clean_data).predict(clean_data)
    cluster_centers = kmeans.cluster_centers_

    feature_map = create_feature_map(clean_data)

    return cluster_labels, cluster_centers, feature_map


def create_clusters_list(clean_data, cluster_labels):
    clean_data['cluster'] = cluster_labels
    clusters_data = []
    # split the data into the respective clusters
    for i in range(len(cluster_labels)):
        clusters_data.append(clean_data[clean_data.cluster == i])

    del clean_data['cluster']  # no need for the cluster labels column

    return clusters_data


def pick_features(clean_data, all_features, chosen_features):
    clean_data = clean_data[all_features]  # restart data with all the features

    rating_mean = clean_data['general rating_mean']
    rating_median = clean_data['general rating_median']
    total_votes = clean_data['general total_votes']

    clean_data = clean_data[chosen_features]  # filter data to just the chosen features for this examination

    clean_data['general rating_mean'] = rating_mean
    clean_data['general rating_median'] = rating_median
    clean_data['general total_votes'] = total_votes

    clean_data = normalize_features(clean_data)
    return clean_data


def analyse_data(num_clusters, clean_data, all_features, chosen_features, colors):

    clean_data = pick_features(clean_data, all_features, chosen_features)

    cluster_labels, cluster_centers, feature_map = examine_features(clean_data, num_clusters)

    clusters_data = create_clusters_list(clean_data, cluster_labels)

    # print all the mean features values of each cluster:
    print_feature_values_of_clusters(num_clusters, clusters_data)

    # plot and print each feature's cluster centers values
    plot_features_cluster_center_values(num_clusters, clusters_data, cluster_centers, feature_map, colors)

    return clusters_data


def run():
    # open data file:
    print('Loading data...')
    with tarfile.open('all_data_pickle.tar.gz', 'r:gz') as f:
        for member in f.getnames():
            data = pickle.loads(f.extractfile(member).read())

    print("Data about: ", len(data), " TV-shows was collected")

    # pick plotting colors:
    colors = ['red', 'black', 'blue', 'yellow', 'cyan', 'green', 'purple', 'grey', 'orange', 'olive', 'lime', 'lavender', 'teal', 'wheat']

    num_clusters = 10

    clean_data, all_features, general_features, genre_features = clean_and_filter_data(data)

    print("After setup we are left with data about: ", len(clean_data), " TV-shows")

    # take only general features:
    clusters_data = analyse_data(num_clusters, clean_data, all_features, general_features, colors)

    # display all the clustering data points for a given feature against the rating_mean:
    plot_cluster_points(num_clusters, clusters_data, colors)

    # take only genre features:
    analyse_data(num_clusters, clean_data, all_features, genre_features, colors)

    # take only most interesting/valuable genres:
    clean_data = clean_data[all_features]  # restart data with all the features
    rating_mean = clean_data['general rating_mean']
    clean_data = clean_data[['genres Drama', 'genres Crime', 'genres Mystery', 'genres Thriller']]
    clean_data['general rating_mean'] = rating_mean
    # normalize:
    clean_data = normalize_features(clean_data)
    # run kmeans:
    rating_mean = clean_data['general rating_mean']
    del clean_data['general rating_mean']
    cluster_labels, cluster_centers, feature_map = run_kmeans(clean_data, num_clusters)
    clean_data['general rating_mean'] = rating_mean
    # create clusters list:
    clusters_data = create_clusters_list(clean_data, cluster_labels)
    # plot drama vs crime / mystery / thriller:
    plot_feature_relationships(num_clusters, clusters_data, cluster_centers, feature_map, colors)


if __name__ == '__main__':
    run()
