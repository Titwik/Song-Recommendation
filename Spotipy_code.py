import os
import time
import json
import spotipy
import torch as tc
import pandas as pd
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

# load the dataset 
def song_data(input_file = "dataset.csv", output_file = "audio_information_2.json"):

    """
    Imports the song data from the raw .csv file. Cleans the dataset and saves it to a .json file
    """

    # load the dataset
    df = pd.read_csv(input_file, low_memory=False)

    # add the information to a .json file
    try:
        with open(output_file, 'r') as file: 
            current_data = json.load(file)
            #print('Data exists')
            return current_data

    except FileNotFoundError:
        json_content = []

    errors = 0
    number = 1    
    duplicates = set()
    for i, row in df.iterrows():

        # check for duplicate songs
        if row['track_id'] in duplicates:
            print(f'Skipping song number {i} as it is a duplicate')
            errors+=1
            continue
        else:
            duplicates.add(row['track_id'])

        try:

            acousticness = float(row['acousticness'])
            danceability = float(row['danceability'])
            energy = float(row['energy'])
            key = float(row['key'])
            loudness = float(row['loudness'])
            mode = float(row['mode'])
            speechiness = float(row['speechiness'])
            instrumentalness = float(row['instrumentalness'])
            liveness = float(row['liveness'])
            valence = float(row['valence'])
            tempo = float(row['tempo'])

            # ensure values are consistent within their specified ranges
            if (acousticness > 1 or acousticness < 0 or
                danceability > 1 or danceability < 0 or
                valence > 1 or valence < 0 or
                energy > 1 or energy < 0 or
                instrumentalness > 1 or instrumentalness < 0 or
                liveness > 1 or liveness < 0 or
                speechiness > 1 or speechiness < 0):

                print(f"Skipping song number {i} due to numeric inconsistency")
                errors+=1
                continue

            # ensure that songs are atleast one minute long
            elif float(row['duration_ms']) < 60000.0:
                errors+= 1
                print(f'Skipping song number {i} as song is too short.')
                continue
            
            else:       
                
                track_info = {
                    "track_number" : number,
                    "track_name": row['track_name'],
                    "track_id": row["track_id"],
                    "artist_name": row["artist_name"],
                    "danceability": danceability,
                    "energy": energy,
                    "key": key,
                    "loudness": loudness,
                    "mode": mode,
                    "speechiness": speechiness,
                    "acousticness": acousticness,
                    "instrumentalness": instrumentalness,
                    "liveness": liveness,
                    "valence": valence,
                    "tempo": tempo,
                    "duration_ms": row['duration_ms'],
                    "time_signature": row['time_signature']
                }
                json_content.append(track_info)
                number+=1
        
        except ValueError:
            print(f"Skipping song number {i} due to conversion error")
            errors+=1
            continue

    # write JSON data to a file
    with open(output_file, 'w') as json_file:
        json.dump(json_content, json_file, indent=4)

    print(f"{130664 - errors-1} songs have been successfully written to {output_file}")

    with open(output_file, 'r') as file: 
        data = json.load(file)
        return data
    
#-------------------------------------------------------------------------------------

# get the audio features
def get_audio_features(features_file = "audio_information.json"):

    """Saves audio features into a Pytorch Tensor"""
    
    songs = song_data(output_file=features_file)
    num_tracks = len(songs)
    
    # initialize tensor to store song data
    features_tensor = tc.zeros(num_tracks, 5)   # change the second dimension for the number of features

    # extract the song information from the .json file
    for index in range(num_tracks):
        song = songs[index]
        features_tensor[index, 0] = song['energy'] 
        features_tensor[index, 1] = song['instrumentalness']
        features_tensor[index, 2] = song['loudness']
        features_tensor[index, 3] = song['valence']
        #features_tensor[index, 4] = song['tempo']
        features_tensor[index, 4] = song["acousticness"] 
        #features_tensor[index, 6] = song['speechiness']
    
    # get rid of rows with nan values
    nan_mask = ~tc.isnan(features_tensor).any(dim=1)
    features_tensor = features_tensor[nan_mask]

    return features_tensor  

#-------------------------------------------------------------------------------------

# use the API to get the genre of every artist in the dataset
def get_genre(num_tracks, song_dataset="audio_information.json"):

    """Retrieves genre information of each artist of a song in the dataset. Saves them to the .json file containing song information."""

    try:
        with open(song_dataset, 'r') as file: 
            songs = json.load(file)

    except FileNotFoundError:
        print("Generating dataset...")
        songs = song_data()
        print('')

    start = num_tracks[0]
    end = num_tracks[-1]  

    track_ids = [songs[i]["track_id"] for i in range(start, end+1, 1)] 

    batch_size = 50
    artist_ids = []

    # initialize API call counter
    api_call_count = 0

    # process track IDs in batches
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        track_data = sp.tracks(batch)

        # Extract artist IDs from each track in the batch
        for track in track_data['tracks']:
            for artist in track['artists']:
                artist_ids.append(artist['id'])
                break       

    genres = []

    for i in range(0, len(artist_ids), batch_size):
        batch = artist_ids[i:i+batch_size]
        artist_info = sp.artsts(batch)
        for artist in artist_info['artists']:
            genres.append(artist['genres'])

    # edit the dataset to have genres
    for genre_index, song_index in zip(range(end - start + 1), range(start, end+1, 1)):     
        song = songs[song_index]
        song['genre'] = genres[genre_index]           

    with open(song_dataset, 'w') as json_file:
        json.dump(songs, json_file, indent=4)       
    
    print(f"Genres added to songs {start} - {end}")
    print(f"Total API calls made: {api_call_count}")

#-------------------------------------------------------------------------------------
# delete the songs with no genre information
def delete_bad_songs(song_dataset = "audio_information.json"):

    """Deletes songs from the .json file that have no genre information."""

    with open(song_dataset, 'r') as file:   
        songs = json.load(file)

    num_tracks = len(songs)
    
    # create a new list with songs that have genres
    good_songs = [song for song in songs if len(song['genre']) > 0] 
    
    # calculate the number of deleted songs
    deleted = num_tracks - len(good_songs)
    
    # save the filtered dataset
    with open("no_bad_songs.json", 'w') as json_file:
        json.dump(good_songs, json_file, indent=4)

    print(f"Deleted {deleted} songs.")
    print(f"There are {num_tracks - deleted} songs in the dataset")
    print(len(good_songs))

#-------------------------------------------------------------------------------------

if __name__ == "__main__":  
    example_features = get_audio_features(features_file="no_bad_songs.json")
    print(example_features[-1])
    print(example_features[-1][0].item())

    

    
        
    

    

    
