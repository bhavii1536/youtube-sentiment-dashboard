import streamlit as st
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from collections import defaultdict
from googleapiclient.discovery import build

# Your YouTube API Key here
API_KEY = 'YOUR_YOUTUBE_API_KEY'  # <-- Replace this with your own API key
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Initialize YouTube API
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Streamlit app
st.title("📊 YouTube Channel Insights + Sentiment Analysis")

# Input: YouTube Channel ID or username
channel_id = st.text_input("Enter YouTube Channel ID:")

# Function to fetch video IDs (limit to 50 recent)
def get_recent_video_ids(channel_id, max_results=50):
    res = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",
        type="video"
    ).execute()
    video_ids = [item['id']['videoId'] for item in res['items']]
    return video_ids

# Function to get comments for a video
def get_comments(video_id):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=50,
            textFormat='plainText'
        ).execute()

        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except:
        pass
    return comments

# Function to get video statistics
def get_video_details(video_ids):
    stats = []
    for i in range(0, len(video_ids), 50):  # API allows 50 at a time
        response = youtube.videos().list(
            part='statistics,snippet',
            id=','.join(video_ids[i:i+50])
        ).execute()

        for item in response['items']:
            video_id = item['id']
            title = item['snippet']['title']
            published_at = item['snippet']['publishedAt']
            views = int(item['statistics'].get('viewCount', 0))
            likes = int(item['statistics'].get('likeCount', 0))
            stats.append({
                'video_id': video_id,
                'title': title,
                'published_at': published_at,
                'views': views,
                'likes': likes
            })
    return pd.DataFrame(stats)

# Sentiment analyzer
def analyze_sentiment(comments):
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for comment in comments:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiments["Positive"] += 1
        elif polarity < 0:
            sentiments["Negative"] += 1
        else:
            sentiments["Neutral"] += 1
    return sentiments

# Month formatting function
def extract_month(published_at):
    return datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime('%B %Y')

# Once Channel ID is entered
if channel_id:
    st.info("Fetching data from YouTube...")
    video_ids = get_recent_video_ids(channel_id)
    video_data = get_video_details(video_ids)

    # Total views and likes
    total_views = video_data['views'].sum()
    total_likes = video_data['likes'].sum()

    st.success("✅ Data fetched successfully!")

    # Display total metrics
    st.markdown(f"### 📺 Total Views (last 50 videos): `{total_views}`")
    st.markdown(f"### 👍 Total Likes (last 50 videos): `{total_likes}`")

    # Sentiment analysis
    all_comments = []
    for vid in video_ids:
        all_comments.extend(get_comments(vid))

    sentiments = analyze_sentiment(all_comments)

    # Pie chart of sentiment
    st.markdown("## 🥧 Sentiment Analysis Summary")
    fig_pie = px.pie(
        names=list(sentiments.keys()),
        values=list(sentiments.values()),
        title="Sentiment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_pie)

    # Line chart of views over months
    video_data['month'] = video_data['published_at'].apply(extract_month)
    monthly_views = video_data.groupby('month')['views'].sum().reset_index()

    st.markdown("## 📈 Monthly Views (Last 50 Videos)")
    fig_line = px.line(
        monthly_views,
        x='month',
        y='views',
        title='Views per Month',
        markers=True,
        labels={'month': 'Month', 'views': 'Total Views'},
        line_shape='spline'
    )
    fig_line.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_line)
