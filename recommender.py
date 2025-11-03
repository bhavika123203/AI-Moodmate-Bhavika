import pandas as pd
import numpy as np

# Load and preprocess the music dataset once
def load_music_data():
    df = pd.read_csv("data/muse_v3.csv")
    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']
    df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
    df = df.sort_values(by=["emotional", "pleasant"])
    return df.reset_index(drop=True)

MUSIC_DATA = load_music_data()

# Split the dataframe into emotion-based sub-dataframes
df_sad = MUSIC_DATA[:18000]
df_fear = MUSIC_DATA[18000:36000]
df_angry = MUSIC_DATA[36000:54000]
df_neutral = MUSIC_DATA[54000:72000]
df_happy = MUSIC_DATA[72000:]

EMOTION_DFS = {
    'Sad': df_sad,
    'Fearful': df_fear,
    'Angry': df_angry,
    'Neutral': df_neutral,
    'Happy': df_happy,
    'Surprised': df_happy, # Map surprised to happy for music
    'Disgusted': df_angry # Map disgusted to angry
}

def get_recommendations(emotions, total_songs=30):
    """
    Refactored function to get a mix of songs based on a list of emotions.
    """
    if not emotions:
        return pd.DataFrame() # Return empty if no emotions
    
    data = pd.DataFrame()
    emotion_counts = len(emotions)

    # Distribute the total songs based on the number of emotions
    songs_per_emotion = total_songs // emotion_counts
    remainder = total_songs % emotion_counts

    for emotion in emotions:
        df = EMOTION_DFS.get(emotion)
        if df is not None:
            num_songs = songs_per_emotion + (1 if remainder > 0 else 0)
            data = pd.concat([data, df.sample(n=num_songs, replace=False)], ignore_index=True)
            if remainder > 0:
                remainder -= 1

    return data.sample(frac=1).reset_index(drop=True) # Shuffle final recommendations
