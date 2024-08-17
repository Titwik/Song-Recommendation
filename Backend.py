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
# Assuming `sp` is a Spotify API client instance
def fetch_features_batch(sp, uri_batch):
    return sp.audio_features(uri_batch)

def get_audio_features(track_uri_list, batch_size=100):
    num_tracks = len(track_uri_list)
    features_tensor = tc.zeros(num_tracks, 2)
    
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
                        #features_tensor[index, 2] = all_features['instrumentalness']
                        #features_tensor[index, 3] = all_features['tempo']
                        #features_tensor[index, 4] = all_features['loudness']

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
    random.seed(142)
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
    
    # Ensure only one label per cluster in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Clusters')

# use elbow method to determine the number of clusters needed
def  silhouette_method(data, centroids):
    
    # 
    
    pass


if __name__ == "__main__":
    k = 6
    example = sample_list(1000, dataset)
    example_features = get_audio_features(example)
    result_cent, tens, labels =  k_means(k, example_features)
    x_tens = tens[:,0].numpy()
    y_tens = tens[:,1].numpy()
    x_cen = result_cent[:,0].numpy()
    y_cen = result_cent[:,1].numpy()
    print(labels)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))

    # Plot the data points
    plot_clusters(x_tens, y_tens, labels, k)

    # Plot the centroids
    plt.scatter(x_cen, y_cen, color='black', marker='x', s=100, label='Centroids')

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Data Points and Centroids')
    plt.show()


