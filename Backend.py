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

# define a function to get the audio features of a track using the API
# MAKE A DIFF FILE FOR THIS!!!
def API_features(track_uri_list, batch_size=100, output_file = "audio_features.json"):

    # prepare to store audio features
    all_features = []

    # get the number of tracks to analyze
    num_tracks = len(track_uri_list)
    
    # check what tracks already exist in the output file
    try:
        with open(output_file, 'r') as file: 
            data = json.load(file)
            existing_uris = {item['uri'] for item in data}
            existing_features = data

    except FileNotFoundError:
        existing_uris = set()
        existing_features = []

    # find the new uri's that need song data
    uris_to_fetch = [uri for uri in track_uri_list if uri not in existing_uris]
    if len(uris_to_fetch) == 0:

        #print('All songs in the list have been added to the local file!')
        return

    # process tracks in batches
    for i in range(0, num_tracks, batch_size):

        # Get the batch of track URIs
        tracks_batch = uris_to_fetch[i:i + batch_size]

        # Make the API call to get audio features
        features = sp.audio_features(tracks=tracks_batch)

        # Filter out None values (in case some tracks have no available features)
        features = [f for f in features if f is not None]
            
        # Append the features to the list
        all_features.extend(features)

        # wait to avoid hitting the rate limit
        time.sleep(10)

    # Combine existing features with new features
    all_features.extend(existing_features)
        
    # Save the collected audio features to a JSON file
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=4)

    #print(f"Saved {len(uris_to_fetch)} audio features to {output_file}")
    
# define a function to get the audio features stored locally
# this allows faster reading of data + avoids hitting the rate limit
def get_audio_features(track_uri_list, features_file = "audio_features.json"):
    
    # use the API_features function
    API_features(track_uri_list)

    with open(features_file, 'r') as file: 
        features_json = json.load(file)

    # initialize tensor to store song data
    num_tracks = len(track_uri_list)
    features_tensor = tc.zeros(num_tracks, 5)   # change the second dimension for the number of features

    # map the track_uri to an index
    uri_to_index = {uri: index for index, uri in enumerate(track_uri_list)}

    # extract the song information from the .json file
    for item in features_json:
        uri = item.get('uri')
        if uri in uri_to_index:
            index = uri_to_index[uri]
            features_tensor[index, 0] = item.get("acousticness")
            features_tensor[index, 1] = item.get('energy')
            features_tensor[index, 2] = item.get('instrumentalness')
            features_tensor[index, 3] = item.get('tempo')
            features_tensor[index, 4] = item.get('loudness') 

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
    example = sample_list(8000, dataset)
    example_features = get_audio_features(example)
    example_features = normalize(example_features)
    #print(silhouette_method(example_features,10))
    elbow_method(example_features,10)
    
    
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