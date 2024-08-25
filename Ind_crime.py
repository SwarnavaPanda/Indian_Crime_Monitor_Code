#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install google-api-python-client youtube-transcript-api transformers torch sqlalchemy


# In[2]:


#pip install google-api-python-client


# In[3]:


#pip install google-auth google-auth-oauthlib google-auth-httplib2


# In[4]:


#pip install pytube


# In[1]:


#pip install google-cloud-speech


# In[2]:


#pip install pydub


# In[23]:


#pip install --upgrade pytube


# In[2]:


#pip install yt-dlp


# In[23]:


#pip install pytube assemblyai


# In[1]:


import sqlite3

# Define the database name
database_name = 'Indian_Crime_DB.db'

def create_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Create the table with the required attributes
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS Crimes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        date TEXT NOT NULL,
        category TEXT NOT NULL,
        state TEXT NOT NULL,
        actual_story TEXT
    )
    '''

    try:
        cursor.execute(create_table_query)
        conn.commit()
        print(f"Database '{database_name}' and table 'Crimes' created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

# Create the database and table
create_database()


# In[1]:


pip install sqlite3 transformers torch


# In[12]:


import sqlite3
import re
from transformers import BertTokenizer, BertModel
import torch

# Define the database name
database_name = 'Indian_Crime_DB.db'

# Load the Hugging Face model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def check_story_similarity(story1, story2):
    tokens1 = tokenizer(story1, return_tensors='pt', truncation=True, padding=True, max_length=512)
    tokens2 = tokenizer(story2, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        output1 = model(**tokens1)
        output2 = model(**tokens2)

    # Use the [CLS] token output for similarity
    vector1 = output1.last_hidden_state[:, 0, :]
    vector2 = output2.last_hidden_state[:, 0, :]

    similarity = torch.nn.functional.cosine_similarity(vector1, vector2)
    return similarity.item()

def extract_story_details(story):
    # Simple extraction logic based on keywords
    date = re.search(r'\b\d{4}-\d{2}-\d{2}\b', story)
    state = re.search(r'\b(?:state1|state2|state3)\b', story, re.IGNORECASE)  # Replace with actual states
    category = re.search(r'\b(?:theft|murder|assault)\b', story, re.IGNORECASE)  # Replace with actual categories

    date = date.group(0) if date else "Unknown"
    state = state.group(0) if state else "Unknown"
    category = category.group(0) if category else "Unknown"

    # Generate a title from the story
    title = story[:50] + '...' if len(story) > 50 else story

    return {
        'title': title,
        'date': date,
        'state': state,
        'category': category,
        'actual_story': story
    }

def insert_story_into_db(story_details):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    insert_query = '''
    INSERT INTO Crimes (title, date, category, state, actual_story)
    VALUES (?, ?, ?, ?, ?)
    '''

    try:
        cursor.execute(insert_query, (
            story_details['title'],
            story_details['date'],
            story_details['category'],
            story_details['state'],
            story_details['actual_story']
        ))
        conn.commit()
        print("Story details inserted into the database.")
    except Exception as e:
        print(f"An error occurred while inserting into the database: {e}")
    finally:
        conn.close()

def story_exists_in_db(story):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    cursor.execute("SELECT actual_story FROM Crimes")
    rows = cursor.fetchall()

    for row in rows:
        db_story = row[0]
        if check_story_similarity(story, db_story) > 0.9:  # Similarity threshold
            conn.close()
            return True

    conn.close()
    return False

def process_story(story):
    if story_exists_in_db(story):
        print("Story already exists in the database.")
    else:
        story_details = extract_story_details(story)
        insert_story_into_db(story_details)

# Example usage
'''new_story = """
On 2024-08-25 in State1, a case of theft was reported. 
The thief stole valuable items from a high-end store.
"""
process_story(new_story)'''


# In[5]:


import time
import os
from googleapiclient.discovery import build
import pytube 
from google.cloud import speech
from pydub import AudioSegment
from urllib.error import HTTPError


# In[6]:


def search_youtube(query):
    try:
        # Search for videos matching the query
        request = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            regionCode='IN',
            relevanceLanguage='en',
            maxResults=5
        )
        response = request.execute()
        return response['items']
    except HTTPError as e:
        print(f"HTTPError: {e.code} - {e.reason}")
        return []


# In[7]:


api_key = 'AIzaSyCM7lA26O7PXQbYhjwrxkZVzmVnu95Dt1A'
youtube=build('youtube', 'v3', developerKey=api_key)


# In[8]:


The_videos=search_youtube('Indian Crime News')
print(The_videos)


# In[9]:


import yt_dlp
import assemblyai as aai
import os

# Function to download audio from YouTube and transcribe it, then delete the audio file
def download_audio_text(youtube_url, output_path='.'):
    downloaded_file = None

    def my_hook(d):
        nonlocal downloaded_file
        if d['status'] == 'finished':
            downloaded_file = d['filename']

    # Options for downloading audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',  # You can change this to 'wav' or 'm4a' if preferred
            'preferredquality': '192',  # Audio quality (192 kbps)
        }],
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Save the file with the title as the name
        'progress_hooks': [my_hook]  # Hook to capture the filename
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print(f"Audio downloaded successfully to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    audio_file_path = downloaded_file
    
    s=audio_file_path
    se=''
    for i in range(0,len(s)-5):
        se=se+s[i]
    se=se+'.mp3'
    audio_file_path=se
    
    # Initialize AssemblyAI
    aai.settings.api_key = "c742248a1e3d417da76db06f2aafc679"
    transcriber = aai.Transcriber()
    
    # Transcribe the audio file
    transcript = transcriber.transcribe(audio_file_path)
    txt = transcript.text

    # Delete the audio file after transcription
    try:
        os.remove(audio_file_path)
        print(f"Audio file {audio_file_path} deleted successfully.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")

    return txt

# Example usage
#youtube_url = 'https://www.youtube.com/watch?v=wwRyEO0yjqM'  # Replace with your YouTube link
#output_path = './out_put_put'  # Optional: specify the path where you want to save the audio file

#transcript_text = download_audio_text(youtube_url, output_path)
#print(f"Transcript: {transcript_text}")


# In[ ]:


def monitor_youtube():
    while True:
        print("Searching for crime news...")
        videos = search_youtube("Indian crime news")
        audio_texts = []
        #print("These are : ",len(videos))

        for video in videos:
            video_url = f"https://www.youtube.com/watch?v={video['id']['videoId']}"
            print(video_url)
            print(f"Processing video: {video['snippet']['title']}")
            
            output_path = './out_put_put'
            
            T=download_audio_text(video_url, output_path)
            audio_texts.append(T)
            

           

        #print(f"Collected Transcripts: {audio_texts}")

        # Wait for 1 minute before repeating
        for i in audio_texts:
            process_story(i)
        time.sleep(60)
        audio_texts.clear()

if __name__ == "__main__":
    monitor_youtube()


# In[3]:


import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the database name
database_name = 'Indian_Crime_DB.db'

# Function to fetch data from the database
def fetch_data():
    conn = sqlite3.connect(database_name)
    query = "SELECT category FROM Crimes"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Function to plot pie chart for crime categories
def plot_pie_chart(data):
    category_counts = data['category'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Percentage of Each Type of Crime')
    st.pyplot(fig)

# Function to plot bar chart for crime categories
def plot_bar_chart(data):
    category_counts = data['category'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis", ax=ax)
    ax.set_title('Number of Crimes by Category')
    ax.set_xlabel('Crime Category')
    ax.set_ylabel('Number of Crimes')
    st.pyplot(fig)

# Streamlit application
st.title('Indian Crime Statistics Dashboard')

# Fetch data from the database
data = fetch_data()

if data.empty:
    st.write("No data available.")
else:
    st.write("### Crime Statistics Overview")
    
    # Pie chart
    st.write("#### Percentage of Each Type of Crime")
    plot_pie_chart(data)
    
    # Bar chart
    st.write("#### Number of Crimes by Category")
    plot_bar_chart(data)


# In[ ]:




