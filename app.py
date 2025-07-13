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

# Load API Key from Streamlit secrets
API_KEY = st.secrets["YOUTUBE_API_KEY"]
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

# Initialize YouTube API
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Load RoBERTa sentiment model
@st.cache_resource
def load_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_roberta_model()

# Streamlit App Title
st.title("📊 YouTube Channel Insights + Sentiment Analysis")

# Input: YouTube Channel ID
channel_id = st.text_input("Enter YouTube Channel ID:")

# Get Channel Name
def get_channel_name(channel_id):
    try:
        response = youtube.channels().list(part="snippet", id=channel_id).execute()
        return response['items'][0]['snippet']['title']
    except:
        return "Unknown Channel"

# Get Recent Video IDs
def get_recent_video_ids(channel_id, max_results=50):
    res = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",
        type="video"
    ).execute()
    return [item['id']['videoId'] for item in res['items']]

# Get Comments for a Video (up to 100)
def get_comments(video_id, max_comments=100):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except:
            break

    return comments

# Get Video Details
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
    return pd.DataFrame(stats)

# Sentiment Analysis using RoBERTa (batch)
def analyze_sentiment(comments):
    sentiments = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    filtered_comments = [c for c in comments if len(c.strip()) > 5][:500]  # Limit to 500 comments

    if not filtered_comments:
        return sentiments

    labels_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    try:
        with torch.no_grad():
            inputs = tokenizer(filtered_comments, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            labels = torch.argmax(probs, dim=1).tolist()

        for label in labels:
            sentiments[labels_map[label]] += 1

    except Exception as e:
        st.warning(f"⚠️ Sentiment analysis error: {e}")

    return sentiments

# Extract Month Name
def extract_month(published_at):
    return datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime('%b')

# Main App Logic
if channel_id:
    st.info("🔄 Fetching data from YouTube...")
    try:
        channel_name = get_channel_name(channel_id)
        st.markdown(f"## 📺 Channel: *{channel_name}*")

        video_ids = get_recent_video_ids(channel_id)
        video_data = get_video_details(video_ids)

        total_views = video_data['views'].sum()
        total_likes = video_data['likes'].sum()

        st.success("✅ Data fetched successfully!")
        st.markdown(f"### 👁️ Total Views (last 50 videos): {total_views}")
        st.markdown(f"### 👍 Total Likes (last 50 videos): {total_likes}")

        # Collect comments
        all_comments = []
        for vid in video_ids:
            all_comments.extend(get_comments(vid, max_comments=100))

        # Sentiment analysis
        sentiments = analyze_sentiment(all_comments)
        sentiment_labels = {"POSITIVE": "😊 Positive", "NEGATIVE": "😡 Negative", "NEUTRAL": "😐 Neutral"}
        sentiment_display = [sentiment_labels.get(k, k) for k in sentiments.keys()]

        st.markdown("## 🥧 Sentiment Analysis Summary")
        fig_pie = px.pie(
            names=sentiment_display,
            values=list(sentiments.values()),
            title="Comment Sentiment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie)

        # Monthly Views Chart
        video_data['month'] = video_data['published_at'].apply(extract_month)
        monthly_views = video_data.groupby('month')['views'].sum().reset_index()
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
        st.plotly_chart(fig_line)

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
