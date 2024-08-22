import pandas as pd
import time
import json
import numpy as np
import os
from flask import Flask, session, redirect, url_for, request
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
from dotenv import load_dotenv
import torch as tc
import matplotlib.pyplot as plt
import random


# define a function to normalize the audio features
def normalize(tensor):
    
    # use z-score normalization technique
    # compute the mean of the tensor along each column
    mean = tc.mean(tensor, dim=0)

    # compute the standard deviation of the tensor along each column
    st_dev = tc.std(tensor, dim=0)

    # compute the z-score
    z_score = (tensor - mean)/st_dev

    return z_score

# define the k-means clustering
def k_means(k, tensor_of_features): 
    
    # normalize the tensor_of_features
    tensor_of_features = normalize(tensor_of_features)

    # initialize tensor of centroids
    centroids = tc.zeros([k, tensor_of_features.size(1)])

    # implement k-means++
    # choose a random data point as initial centroid
    #random.seed(142)
    initial_centroid = tensor_of_features[random.randint(0, tensor_of_features.size(0)-1)]
    centroids[0] = initial_centroid

    # compute the distance of datapoints from the nearest centroid
    for i in range(1,k):    

        distances = tc.min(tc.cdist(tensor_of_features, centroids[:i]), dim=1)[0]
        probabilities = (distances ** 2 / tc.sum(distances ** 2)).numpy()
        new_centroid_index = np.random.choice(np.array(range(tensor_of_features.size(0))), p=probabilities)
        new_centroid = tensor_of_features[new_centroid_index]
        centroids[i] = new_centroid     

    # start the algorithm
    while True:

        # compute distances and label the data into clusters nearest to them
        distances = tc.cdist(tensor_of_features, centroids)
        labels = tc.argmin(distances, dim=1)
        
        # compute the mean of the data in a cluster, and 
        # set the mean as the new centroid for the cluster
        new_centroids = tc.zeros(k, tensor_of_features.size(1))
        for i in range(k):
            points_in_cluster = tensor_of_features[labels == i]
            if points_in_cluster.size(0) > 0: # if there is atleast 1 point in the cluster
                new_centroids[i] = points_in_cluster.mean(dim=0)
        
        # check if the centroids need any more adjusting
        if tc.equal(centroids, new_centroids):
            return new_centroids, tensor_of_features, labels
        else:
            centroids = new_centroids
        
# plot clusters for visualization purposes. Only for 2 features
def plot_clusters(x_tens, y_tens, labels, k): 

    # Define a list of colors manually
    color_list = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'cyan', 'magenta']

    # Ensure that the number of colors matches or exceeds the number of clusters
    if k > len(color_list):
        raise ValueError(f"Not enough colors defined for {k} clusters. Add more colors to the list.")
    
    # Plot each point with the corresponding color
    for i in range(len(x_tens)):
        cluster_label = labels[i].item()  # Convert tensor to a scalar
        plt.scatter(x_tens[i], y_tens[i], color=color_list[cluster_label], marker='o', label=f'Cluster {cluster_label}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Clusters')

# use elbow method to determine the number of clusters needed
def elbow_method(tensor_of_features,k_max):

    # initialize WCSS values (within-cluster sum of squares)
    WCSS = tc.zeros(k_max)

    for k in range(1, k_max+1):

        # run the k-means algorithm
        centroids, tensor_of_features, labels = k_means(k, tensor_of_features)
        total_WCSS = 0.0

        for i,centroid in enumerate(centroids):
            
            # compute the WCSS
            cluster_points = tensor_of_features[labels == i]
            distances = tc.sum((cluster_points - centroid) ** 2, dim=1) # sum along rows of the tensor

            # sum the distances and add to the WCSS tensor
            total_WCSS += tc.sum(distances).item()

        # add the value to the WCSS tensor    
        WCSS[k-1] = total_WCSS
    
    # plot the WCSS against k
    plt.plot(np.array(range(1,k_max + 1)), WCSS.numpy(), 'bx-')
    plt.title('WCSS against k-clusters')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()

def silhouette_method(tensor_of_features, k_max):

    average_silhouette_scores = []
    
    for k in range(3, k_max+1):

        # initialize average intra-cluster distance, and 
        # average inter-cluster distance
        a = tc.zeros(tensor_of_features.size(0))
        b = tc.zeros_like(a)

        # implement the k-means algorithm
        centroids, tensor_of_features, labels = k_means(k, tensor_of_features)
        
        # loop over the number of data points
        for i in range(tensor_of_features.size(0)):

            # find the average intra-cluster distance
            cluster_label = labels[i].item()
            cluster_points = tensor_of_features[labels == cluster_label]
            distance_a = tc.norm(tensor_of_features[i] - cluster_points, dim=1)
            a[i] = distance_a.mean()

            # find the clusters that the data point is not in
            other_labels = []
            average_distance = []
            for not_in in range(centroids.size(0)):
                if not_in != cluster_label:
                    other_labels.append(not_in)
            
            # find the distance to the nearest cluster 
            for label in other_labels:
                distance_b = tc.norm(tensor_of_features[i] - tensor_of_features[labels == label], dim=1)
                average_distance.append(distance_b.mean())
            b[i] = min(average_distance)
        
        s = (b - a) / tc.maximum(a, b)
        average_silhouette_scores.append(s.mean().item())

    # find the optimum number of clusters
    optimal_k = average_silhouette_scores.index(max(average_silhouette_scores)) + 3 # account for starting at k=3
    return optimal_k