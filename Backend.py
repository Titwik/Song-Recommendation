import pandas as pd
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# load the environment variables    
load_dotenv()

# load the dataset
dataset = '/home/titwik/Projects/Spotify Project/song_dataset/data/mpd.slice.0-999.json'

# set the client id, client secret, redirect_uri and scope for project
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
redirect_uri = os.getenv('redirect_uri')
scope = "user-library-read"

# intialize the spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

# define a function to load unique songs from the dataset
def track_uris(dataset):           
     
    # Load the JSON file
    with open(dataset, 'r') as f:
        data = json.load(f)

    # Extract playlists
    playlists = data['playlists']

    # Use a list to maintain order and a set for quick duplicate checking
    track_uri_list = []
    track_uri_set = set()

    # Loop through the tracks in each playlist 
    for playlist in playlists:
        for track in playlist['tracks']:
            uri = track['track_uri']
            if uri not in track_uri_set:
                track_uri_list.append(uri)
                track_uri_set.add(uri)
    
    return track_uri_list   

# define a function to display the track name and artist name from a track uri
def display_track_name(track_uri):      
    try:
        
        # Fetch the track's information
        track_info = sp.track(track_uri)

        # Extract and print the track name
        track_name = track_info['name']
        artist_name = track_info['artists'][0]['name']
        print(f"Track Name: {track_name} - {artist_name}")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# define a function to obtain the audio features of a list of tracks
def fetch_features_batch(sp, uri_batch):
    return sp.audio_features(uri_batch)

def get_audio_features(track_uri_list, batch_size=100):
    num_tracks = len(track_uri_list)
    features_tensor = tc.zeros(num_tracks, 5)   # change the second dimension for the number of features
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, num_tracks, batch_size):
            uri_batch = track_uri_list[i:i + batch_size]
            futures.append(executor.submit(fetch_features_batch, sp, uri_batch))

        for future in as_completed(futures):
            batch_features = future.result()
            if batch_features:
                for j, all_features in enumerate(batch_features):
                    index = (futures.index(future) * batch_size) + j
                    if all_features:
                        features_tensor[index, 0] = all_features['acousticness']
                        features_tensor[index, 1] = all_features['energy']
                        features_tensor[index, 2] = all_features['instrumentalness']
                        features_tensor[index, 3] = all_features['tempo']
                        features_tensor[index, 4] = all_features['loudness']

    return features_tensor

# define a function to get a sample list of song uri's for testing
def sample_list(num_of_songs, dataset):
    sample_list = []
    uri = track_uris(dataset)
    for i in range(num_of_songs):
        sample_list.append(uri[i])
    return sample_list

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

if __name__ == "__main__":
    example = sample_list(1000, dataset)
    example_features = get_audio_features(example)
    example_features = normalize(example_features)
    print(silhouette_method(example_features,10))
    #elbow_method(example_features,10)
    
    
    #k_list = list(range(5,6))
    
    #for k in k_list:
    #    result_cent, labels =  k_means(k, example_features)
    #    x_tens = example_features[:,0].numpy()
    #    y_tens = example_features[:,1].numpy()
    #    x_cen = result_cent[:,0].numpy()
    #    y_cen = result_cent[:,1].numpy()
    #    print(labels)

        # Create the scatter plot
    #    plt.figure(figsize=(8, 6))

        # Plot the data points
    #    plot_clusters(x_tens, y_tens, labels, k)

        # Plot the centroids
    #    plt.scatter(x_cen, y_cen, color='black', marker='x', s=100, label='Centroids')

        # Add labels and legend
    #    plt.xlabel('X-axis')
    #    plt.ylabel('Y-axis')
    #   plt.title('Scatter Plot of Data Points and Centroids')
    #    plt.show()