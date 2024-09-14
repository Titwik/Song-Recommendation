# K-Means Song Recommendation Program

In this project, I have created a K-Means clustering algorithm from scratch in order to recommend similar songs to the user. The audio features used for clustering is generated using the [Spotify WebAPI](https://spotipy.readthedocs.io/en/2.24.0/). 

The dataset used comes from [this](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features?resource=download) Kaggle dataset, which contains the audio features for approximately 130K songs. 

## How To Run The Program

First, clone this repository into a directory of your choice locally using by first navigating to your chosen directory using `cd path/to/your/directory`, and then cloning the repository using 
```
git clone https://github.com/Titwik/Song-Recommendation.git
```
 You can then navigate to the repo using `cd Song-Recommendation`. 

In order to use the program, there are two things to be done:
1. You will need to download the required packages for the program
2. You will need to get a few config details from the [Spotify for Developers](https://developer.spotify.com/) website.
### 1. Installing the required packages

Begin by installing all the packages required to run the program by opening the terminal and running 
```
pip install -r requirements.txt
``` 

### 2. Get the Client ID, and Client Secret

1. Navigate to the [Spotify for Developers](https://developer.spotify.com/) and log into your Spotify account.
2. Go to your dashboard.
3. Click on 'Create an app'
4. Pick an ‘App name’ and ‘App description’ of your choice, set the 'redirect uri' as "http://localhost:5000/callback" and mark the 'Spotify WebAPI' checkbox.
5. After creation, you see your ‘Client Id’ and you can click on 'View client secret' to unhide your 'Client secret'.
6. Open the terminal and navigate to the repository
7. Paste the following lines into the terminal:
~~~
touch .env
nano .env
~~~
8. Paste the following into your .env file:
```
client_id = "YOUR CLIENT ID"
client_secret = "YOUR CLIENT SECRET"
redirect_uri = "http://localhost:5000/callback"
```
 
9. Copy your Client ID and Client Secret, and paste them in the .env file at the appropriate entries (make sure it's pasted within the "speech marks"). Save the file when done.

At this point, run 
```
python Recommendation_system.py
```
 in the terminal, and use the program!

![image](https://github.com/user-attachments/assets/25eaec15-f539-4d26-a2d7-eaf6ea8b02f9)


## How It Works

There are 3 main files that serve various purposes:
### `Spotipy_code.py`

`Spotipy_code.py` contains all the code pertaining to importing the dataset in its raw .csv file format, cleaning it, and appending the relevant data to the `audio_information.json`. Additionally, there is a function which uses the Spotipy WebAPI to retrieve genre information for each song's artist, and another function which filters out all the songs without genre information associated with it, saving the remainder in `no_bad_songs.json`.

The final function of note saves all the (relevant) audio features into an $n \times m$ tensor, where $n$ is the number of songs, and $m$ is the number of features we analyze. 

The features chosen for analysis are:

- Energy
- Instrumentalness
- Loudness
- Valence
- Acousticness
### `K_means_code.py`

`K_means_code.py` does all the heavy lifting when it comes to clustering the data. The tensor obtained from `Spotipy_code.py` is normalized and fed into the k-means algorithm. The elbow method is used to identify the ideal number of clusters. 

![image](https://github.com/user-attachments/assets/e54469a4-dd0a-4457-a79f-274ad6ef2754)


The choice of $k=5$ clusters is intended to capture a diverse range of audio features, enabling the formation of clusters that represent a variety of musical characteristics.

### `Recommendation_system.py`

The final module of the program. This file contains a single function that allows for a user to input their beloved song, and the artist of the song. The function then retrieves the audio features of this input, and adds it to `no_bad_songs.json`. 

The tensor of audio features is created using `Spotipy_code.py`, and then k-means clustering is done on this tensor using `K_means_code.py`. Song recommendations are chosen from the cluster in which the input song lies, and uses genre information to add a second layer of filtering. 

Songs from the same cluster and atleast one genre in common with the user's song are stored, and five are randomly chosen as recommendations for the user.
