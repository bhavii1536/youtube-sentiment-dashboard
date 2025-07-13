# 📦 Importing all required libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from collections import defaultdict
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 🔑 Load API Key securely from Streamlit secrets
API_KEY = st.secrets["YOUTUBE_API_KEY"]  # Your YouTube API key from secrets

# 🔧 YouTube API service initialization
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# 🧠 Load RoBERTa model & tokenizer only once to speed up performance
@st.cache_resource
def load_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")  # Tokenizer for input
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")  # Pretrained sentiment model
    return tokenizer, model

tokenizer, model = load_roberta_model()  # Load model and tokenizer

# 🖥️ Streamlit App Title
st.title("📊 YouTube Channel Insights + Sentiment Analysis")

# ✍️ User inputs Channel ID
channel_id = st.text_input("Enter YouTube Channel ID:")

# 🔍 Get channel name using the ID
def get_channel_name(channel_id):
    try:
        response = youtube.channels().list(part="snippet", id=channel_id).execute()
        return response['items'][0]['snippet']['title']  # Extract channel title
    except:
        return "Unknown Channel"

# 🎞️ Get recent video IDs (up to 50) from the channel
def get_recent_video_ids(channel_id, max_results=50):
    res = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",
        type="video"
    ).execute()
    return [item['id']['videoId'] for item in res['items']]  # Extract video IDs

# 💬 Get top-level comments from a video
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
            comments.append(comment)  # Add each comment to the list
    except:
        pass
    return comments

# 📊 Get video details like title, views, likes
def get_video_details(video_ids):
    stats = []
    for i in range(0, len(video_ids), 50):
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
    return pd.DataFrame(stats)  # Return as a DataFrame

# 🤖 Analyze sentiment of comments using RoBERTa
def analyze_sentiment(comments):
    sentiments = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}  # Initialize counters
    filtered_comments = [c for c in comments if len(c.strip()) > 5]  # Skip tiny comments

    if not filtered_comments:
        return sentiments

    labels_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}  # Index to label map

    for comment in filtered_comments[:300]:  # Limit to 300 for speed
        try:
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                label_id = torch.argmax(probs, dim=1).item()  # Get the predicted label
                sentiment = labels_map[label_id]
                sentiments[sentiment] += 1  # Increment count
        except Exception as e:
            st.warning(f"⚠ Skipping one comment due to error: {e}")
            continue

    return sentiments

# 📅 Extract the month from published date
def extract_month(published_at):
    return datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime('%b')

# 🚀 Main Streamlit logic
if channel_id:
    st.info("🔄 Fetching data from YouTube...")  # Let user know it's loading
    try:
        channel_name = get_channel_name(channel_id)  # Get channel title
        st.markdown(f"## 📺 Channel: {channel_name}")

        video_ids = get_recent_video_ids(channel_id)  # Get recent videos
        video_data = get_video_details(video_ids)  # Get their stats

        total_views = video_data['views'].sum()  # Sum of views
        total_likes = video_data['likes'].sum()  # Sum of likes

        st.success("✅ Data fetched successfully!")
        st.markdown(f"### 👁 Total Views (last 50 videos): {total_views}")
        st.markdown(f"### 👍 Total Likes (last 50 videos): {total_likes}")

        all_comments = []
        for vid in video_ids:
            all_comments.extend(get_comments(vid))  # Combine all comments

        sentiments = analyze_sentiment(all_comments)  # Run sentiment analysis
        sentiment_labels = {"POSITIVE": "😊 Positive", "NEGATIVE": "😡 Negative", "NEUTRAL": "😐 Neutral"}

        # 🥧 Pie chart: Sentiment Distribution (with color swap)
        st.markdown("## 🔍 Sentiment Analysis Summary")
        fig_pie = px.pie(
            names=[sentiment_labels[k] for k in sentiments.keys()],
            values=list(sentiments.values()),
            title="Comment Sentiment Distribution",
            color=[sentiment_labels[k] for k in sentiments.keys()],
            color_discrete_map={
                "😊 Positive": "#FDB462",  # Changed to Orange
                "😡 Negative": "#FB8072",  # Red stays same
                "😐 Neutral": "#80B1D3"   # Changed to Blue
            }
        )
        st.plotly_chart(fig_pie)  # Show pie chart

        # 📈 Line chart: Monthly views
        video_data['month'] = video_data['published_at'].apply(extract_month)  # Get month
        monthly_views = video_data.groupby('month')['views'].sum().reset_index()

        # 📅 Sort months in order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_views['month'] = pd.Categorical(monthly_views['month'], categories=month_order, ordered=True)
        monthly_views = monthly_views.sort_values('month')

        st.markdown("## 📈 Monthly Views (Last 50 Videos)")
        fig_line = px.line(
            monthly_views,
            x='month',
            y='views',
            title='Monthly Views Overview',
            markers=True,
            labels={'month': 'Month', 'views': 'Total Views'},
            line_shape='spline'
        )
        fig_line.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_line)  # Show line chart

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")  # Show error if any
