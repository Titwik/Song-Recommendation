from tqdm import tqdm
import time
import json
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import torch as tc

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

# use the API to get information about a list of songs
def API_information(track_uri_list, input_file = dataset, output_file = "audio_information.json"):
    
    # load the input JSON file
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    # extract playlists
    playlists = input_data['playlists']

    # prepare to store audio features
    all_features = []

    # check what tracks already exist in the output file
    try:
        with open(output_file, 'r') as file: 
            current_data = json.load(file)
            existing_uris = {item['uri'] for item in current_data}
            existing_features = current_data

    except FileNotFoundError:
        existing_uris = set()
        existing_features = []

    # find the new uri's that need song data
    uris_to_fetch = [uri for uri in track_uri_list if uri not in existing_uris]
    if len(uris_to_fetch) == 0:

        print('All songs in the list have been added to the local file!')
        return
    
    information = []

    # process tracks in batches
    batch_size = 100    
    for playlist in playlists:
        for track in playlist['tracks']:
            if track['track_uri'] in uris_to_fetch:
                
                song_info = {
                    'track_name': track['track_name'],
                    'artist_name': track['artist_name'],
                    'track_uri': track['track_uri']
                    #'genres' : sp.artist(artist_id)['genres']                    
                }

                # ensure duplicates are filtered out
                if not any(song['track_name'] == song_info['track_name'] and 
                           song['artist_name'] == song_info['artist_name'] 
                           for song in information):
                    information.append(song_info)

    

    return information

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


# define a function to get the audio features stored locally
# this allows faster reading of data + avoids hitting the rate limit
def get_audio_features(track_uri_list, features_file = "audio_features.json"):
    
    # use the API_information function
    API_information(track_uri_list)

    with open(features_file, 'r') as file: 
        features_json = json.load(file)

    # initialize tensor to store song data
    num_tracks = len(track_uri_list)
    features_tensor = tc.zeros(num_tracks, 6)   # change the second dimension for the number of features

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
            features_tensor[index, 3] = item.get('danceability')
            features_tensor[index, 4] = item.get('loudness') 
            features_tensor[index, 5] = item.get('valence')

    return features_tensor

if __name__ == "__main__":
    sample = sample_list(1, dataset)
    display_track_name(sample[0])
    