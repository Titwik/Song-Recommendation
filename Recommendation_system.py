import time
import os
import json
import random
import difflib
import spotipy
import Spotipy_code
import K_means_code
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

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

#----------------------------------------------------------------------------------------------------------------------------------

def recommend_songs(song_dataset = "no_bad_songs.json"):

    """Input a song and get a list of 5 song recommendations back, along with their spotify links"""

    song_name =  input("What song do you like?\n")
    print('')
    artist_name = input("Who is the artist of the song you like?\n")
    print('')
    print('Processing. Please wait...')
        
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1) 

    try:
        track = results['tracks']['items'][0]
        track_id = track['id']
        track_uri = track['uri']
        audio_features = sp.audio_features(track_uri)
        audio_features = audio_features[0]
        track_info = {
                    "track_name": song_name,
                    "track_id": track_id,
                    "artist_name": artist_name,
                    "danceability": audio_features['danceability'],
                    "energy": audio_features["energy"],
                    "key": audio_features["key"],
                    "loudness": audio_features['loudness'],
                    "mode": audio_features['mode'],
                    "speechiness": audio_features["speechiness"],
                    "acousticness": audio_features["acousticness"],
                    "instrumentalness": audio_features['instrumentalness'],
                    "liveness": audio_features['liveness'],
                    "valence": audio_features['valence'],
                    "tempo": audio_features['tempo'],
                    "duration_ms": audio_features['duration_ms'],
                    "time_signature": audio_features['time_signature']
                }

        # get artist id and genre information
        track_data = sp.track(track_id)
        artist_ids = []

        for artist in track_data["artists"]:
            artist_ids.append(artist["id"])
            break

        artists_data = sp.artists(artist_ids)

        genres = []

        for artist in artists_data["artists"]:
            genres += artist["genres"]

        genres = set(genres)
        genres = list(genres)

        track_info['genre'] = genres
        song_data = Spotipy_code.song_data(output_file=song_dataset)
        song_data.append(track_info)

        with open(song_dataset, 'w') as json_file:
            json.dump(song_data, json_file, indent=4)    
            #print("Song written to file\n")
            
        with open(song_dataset, 'r') as file:
            song_data = json.load(file)

        num_tracks = len(song_data)

        tensor_of_features = Spotipy_code.get_audio_features(song_dataset)

        _, tensor_of_features, labels = K_means_code.k_means(4, tensor_of_features)

        #print('Song clustering complete!')
        print('Almost there...')
        print('')   

        # keep track of indices of songs in the cluster
        indices_in_cluster = []
        for i, label in enumerate(labels):
            if label == labels[-1] and i != (len(song_data) - 1):
                indices_in_cluster.append(i)

        # get songs with similar genres 
        user_song = song_data[-1]
        user_song_genres = user_song['genre']

        # find songs with similar genres
        # if the user's song doesn't have genres, pick 5 random songs from the cluster
        if len(user_song_genres) == 0:
            print("Unfortunately, this song does not have any genres associated with it.")
            print('Here are some songs you may like:')
            print('')

            for i in range(5):
                #random.seed(234)
                recommendation_index = random.choice(indices_in_cluster)
                recommendation = song_data[recommendation_index]                
                name = recommendation['track_name']
                artist = recommendation['artist_name']
                print(f"{name} by {artist}")
                indices_in_cluster.remove(recommendation_index)
        
        else:
            # find the number of genres associated with the track
            no_of_genres = len(user_song_genres)
            potential_songs = []

            # search through the songs for all songs matching these genres
            for i in range(num_tracks):
                song = song_data[i]
                for j in range(no_of_genres):
                    genre = user_song_genres[j]
                    if genre in song['genre']:
                        potential_songs.append(song)
            
            # if there are not enough songs to recommend, use difflib to find possible matches
            if (len(potential_songs) < 5 or
                len(user_song_genres) < 2):
                for song in song_data:

                    # break out of the loop when potential_songs has atleast 5 songs
                    if len(potential_songs) >= 5:
                        break

                    for song_genre in song['genre']:
                        for user_song_genre in user_song_genres:
                            seq_match = difflib.SequenceMatcher(None, song_genre, user_song_genre)
                            ratio = seq_match.ratio()

                            if ratio > 0.5:
                                if song not in potential_songs:
                                    potential_songs.append(song)
                                    
                                    # stop further genre comparisons for this song to avoid duplicates          
                                    break  
        
            print("#------------------------------------------------------------------------")
            print('Here are some songs you may like:')
            print('')
            for i in range(5):
                recommendation = random.choice(potential_songs)                
                name = recommendation['track_name']
                artist = recommendation['artist_name']
                track_id = recommendation['track_id']
                print(f"Track name: {name}")
                print(f"Artist: {artist}")
                print(f"Link to song: https://open.spotify.com/track/{track_id}")
                print('')
            print("#------------------------------------------------------------------------")
            print('')
            
        # delete the user-entered song from the dataset 
        # if it exists in the dataset already. 
        new_track_id = song_data[-1]['track_id']
        del song_data[-1]       
        for song in song_data:
            if new_track_id == song['track_id']:
                with open(song_dataset, 'w') as json_file:
                    json.dump(song_data, json_file, indent=4)
                #print("Song removed from file\n")
                break

        
    except IndexError:
        print('No song or artist found. Please try again')

#----------------------------------------------------------------------------------------------------------------------------------

def meet_in_middle_songs(song_dataset = "no_bad_songs.json"):
    
    pass
#----------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    recommend_songs()
    