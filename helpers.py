import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
import sklearn.preprocessing as pp
import sklearn.feature_selection as fs
import sklearn.cluster as ct
import sklearn.mixture as mx
sns.set()
import importlib
import time
import warnings
from kneed import KneeLocator
warnings.simplefilter(action='ignore', category=FutureWarning)

def find_cluster_features(centers, thresh='max'):
    k = centers.shape[0]
#     features = set()
    features = []
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            absdiff = np.abs(centers[c1] - centers[c2])
            if thresh == 'max':
                max_feature = np.argmax(absdiff)
                features.append(max_feature)
            else:
                found_features = np.where(absdiff > thresh)[0]
                for feat in found_features:
                    features.append(feat)
#             features.add(max_feature)
    return list(features)


def plot_cluster_features(X, clusters, centers, features, feature_names=None):
    k = centers.shape[0]
    for i in range(len(features)):
        for j in range (i + 1, len(features)):
            f1, f2 = features[i], features[j]
            plt.scatter(X[:, f1], X[:, f2], c=clusters, s=30, cmap='viridis')
            plt.scatter(centers[:, f1], centers[:, f2], c=np.arange(k), s=300, alpha=0.7, marker='o',
                        cmap='viridis', edgecolors='black', linewidths=2)
            xlabel = feature_names[f1] if feature_names is not None else 'feature %d' % f1
            ylabel = feature_names[f2] if feature_names is not None else 'feature %d' % f2
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

def plot_clusters_and_labels(X_or_data, f1, f2, feature_names=None, y=None, clusters=None, centers=None):
    if y is not None:
        data = get_full_cluster_data(X_or_data, y, clusters, centers)
    else:
        data = X_or_data
    g = sns.scatterplot(x=f1, y=f2, hue='cluster', style='label',
        size='cluster_center', sizes=[70, 200], data=data, legend='brief')
    if feature_names is not None:
        fnames = feature_names
    else:
        fnames = [str(f1), str(f2)]
    plt.xlabel(fnames[0])
    plt.ylabel(fnames[1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Clusters view from features %s vs %s' % (fnames[0], fnames[1]))
    plt.show()

def plot_clusters_labels_all_features(data, features, feature_names=None):
    for i in range(len(features)):
        for j in range (i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if feature_names is not None:
                fnames = [feature_names[f1], feature_names[f2]]
            else:
                fnames = None
            plot_clusters_and_labels(data, f1, f2, feature_names=fnames)

def get_full_cluster_data(X, y, clusters, centers):
    data = X.copy()
    data['label'] = y
    data['cluster'] = clusters
    data['cluster_center'] = 0
    df_centers = pd.DataFrame(centers)
    df_centers['label'] = 3
    df_centers['cluster'] = np.arange(centers.shape[0])
    df_centers['cluster_center'] = 1
    data = data.append(df_centers)
    return data

def get_cluster_label_ratios(data, n_clusters, pos_val=1, neg_val=-1):
    clusters = np.arange(n_clusters)
    out = {
        "cluster": [],
        "negative": [],
        "positive": [],
        "pos_over_neg": [],
        "total": []
    }
    for cluster in clusters:
        positive = data.query('cluster == %d and label == %d' % (cluster, pos_val)).shape[0]
        negative = data.query('cluster == %d and label == %d' % (cluster, neg_val)).shape[0]
        out["cluster"].append(cluster)
        out["negative"].append(negative)
        out["positive"].append(positive)
        out["pos_over_neg"].append(positive / float(negative))
        out["total"].append(positive + negative)
    df = pd.DataFrame(out)
    df['pct_neg'] = df['negative'] / df['negative'].sum()
    df['pct_pos'] = df['positive'] / df['positive'].sum()
    df['pct_total'] = df['total'] / df["total"].sum()
    return df

def plot_cluster_labels_ratios(cluster_label_data):
    plot_data = cluster_label_data.melt(id_vars=['cluster'],
        value_vars=['positive', 'negative'],
        var_name='label', value_name='count')
    sns.barplot(data=plot_data, x='cluster', y='count', hue='label')
    plt.title('Distribution of labels in each cluster')
    plt.show()

def get_kmeans_scores(X, ks=None, random_state=None):
    results = {
        "k": ks if ks is not None else np.arange(1, 11),
        "score": []
    }
    for k in results["k"]:
        kmeans = ct.KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        score = kmeans.score(X)
        results['score'].append(-score)
    return results

def plot_elbow_curve(centroids, errors, xlabel='Centroids', ylabel='SSE', title='Elbow Method showing optimal K', other_ks=None):
    kn = KneeLocator(centroids, errors, S=1.0, curve='convex', direction='decreasing')
    print("Knee", kn.knee)
    plt.plot(centroids, errors, 'bx-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    if other_ks is not None:
        for k in other_ks:
            plt.vlines(k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', linewidths=0.7)
    plt.show()
    
def kmeans_experiment(X, y, k, random_state=None, thresh=150, feature_names=None, pos_val=1, neg_val=-1):
    kmeans = ct.KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    features = find_cluster_features(centers, thresh=thresh)
    print("Features", features)
    print("Unique features", sorted(list(set(features))))
    data = get_full_cluster_data(X, y, clusters, centers)
    plot_clusters_labels_all_features(data, list(set(features)), feature_names=feature_names)
    cluster_label_counts = get_cluster_label_ratios(data, k, pos_val=pos_val, neg_val=neg_val)
    print(cluster_label_counts)
    plot_cluster_labels_ratios(cluster_label_counts)

def gmm_experiment(X, y, k, covariance='full', random_state=None, thresh=150, feature_names=None, pos_val=1, neg_val=-1):
    gmm = mx.GaussianMixture(n_components=k, covariance_type=covariance, random_state=random_state)
    gmm.fit(X)
    clusters = gmm.predict(X)
    centers = gmm.means_
    features = find_cluster_features(centers, thresh=thresh)
    print("Features", features)
    print("Unique features", sorted(list(set(features))))
    data = get_full_cluster_data(X, y, clusters, centers)
    # helpers.plot_cluster_features(X, y, centers, list(set(features)), feature_names=feature_names)
    plot_clusters_labels_all_features(data, list(set(features)), feature_names=feature_names)
    cluster_label_counts = get_cluster_label_ratios(data, k, pos_val=pos_val, neg_val=neg_val)
    print(cluster_label_counts)
    plot_cluster_labels_ratios(cluster_label_counts)

def plot_reconstructed(X_orig, X_restored, y_values, f1, f2, feature_names=None):
    X_orig = pd.DataFrame(X_orig)
    X_orig['class'] = y_values
    X_restored = pd.DataFrame(X_restored)
    X_restored['class'] = y_values
    if feature_names is not None:
        fnames = feature_names
    else:
        fnames = [str(f1), str(f2)]
    sns.relplot(x=f1, y=f2, hue='class', data=X_orig)
    plt.title('Original dataset features %s and %s' % (fnames[0], fnames[1]))
    plt.show()
    sns.relplot(x=f1, y=f2, hue='class', data=X_restored)
    plt.title('Reconstructred dataset features %s and %s' % (fnames[0], fnames[1]))
    plt.show()
    
