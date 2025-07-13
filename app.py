import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from textblob import TextBlob
import time

# ⛓️ Setup API Key from secrets
api_key = st.secrets["YOUTUBE_API_KEY"]

# 🔌 Connect to YouTube API
youtube = build('youtube', 'v3', developerKey=api_key)

# 📦 Get channel info
def get_channel_stats(youtube, channel_id):
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=channel_id
    )
    response = request.execute()
    
    if 'items' not in response or not response['items']:
        return None  # Handle invalid channel ID
    
    data = dict(
        Channel_Name = response['items'][0]['snippet']['title'],
        Subscribers = int(response['items'][0]['statistics'].get('subscriberCount', 0)),
        Views = int(response['items'][0]['statistics'].get('viewCount', 0)),
        Total_Videos = int(response['items'][0]['statistics'].get('videoCount', 0)),
        Playlist_ID = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    )
    return data

# 📹 Get video IDs from playlist
def get_video_ids(youtube, playlist_id):
    video_ids = []
    next_page_token = None
    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return video_ids

# 🧠 Get video comments
def get_comments(youtube, video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50,
            textFormat="plainText"
        )
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except Exception as e:
        pass  # If comments are disabled
    return comments

# 💬 Analyze sentiments
def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        sentiments.append((comment, polarity, sentiment))
    return sentiments

# 🖼️ Streamlit UI
st.set_page_config(page_title="YouTube Channel Analyzer", page_icon="📊")
st.title("📊 YouTube Channel Analyzer")
st.write("Enter a **channel username or channel ID** below:")

channel_id = st.text_input("🔍 Channel Name or ID", placeholder="UCBR8-60-B28hp2BmDPdntcQ")

if channel_id:
    with st.spinner("Fetching channel data..."):
        channel_info = get_channel_stats(youtube, channel_id)
        if not channel_info:
            st.error("❌ Invalid channel ID or no data found.")
        else:
            st.subheader("📌 Channel Info")
            st.write(f"**Name:** {channel_info['Channel_Name']}")
            st.write(f"**Subscribers:** {channel_info['Subscribers']:,}")
            st.write(f"**Total Views:** {channel_info['Views']:,}")
            st.write(f"**Total Videos:** {channel_info['Total_Videos']:,}")

            st.subheader("📽️ Fetching Recent Videos...")
            video_ids = get_video_ids(youtube, channel_info['Playlist_ID'])

            if not video_ids:
                st.warning("No videos found.")
            else:
                all_sentiments = []
                for video_id in video_ids[:5]:  # Limit to first 5 videos for performance
                    comments = get_comments(youtube, video_id)
                    sentiments = analyze_sentiment(comments)
                    all_sentiments.extend(sentiments)
                    time.sleep(1)

                if all_sentiments:
                    st.subheader("💬 Sentiment Analysis Report")
                    df = pd.DataFrame(all_sentiments, columns=["Comment", "Polarity", "Sentiment"])
                    st.dataframe(df)

                    # 📊 Summary count
                    st.subheader("📈 Sentiment Summary")
                    st.bar_chart(df["Sentiment"].value_counts())

                else:
                    st.info("No comments found for recent videos.")

