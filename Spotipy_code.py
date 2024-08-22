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

# define a function to get a sample list of song uri's for testing
def sample_list(num_of_songs, dataset):
    sample_list = []
    uri = track_uris(dataset)
    for i in range(num_of_songs):
        sample_list.append(uri[i])
    return sample_list

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