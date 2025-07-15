import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from calendar import month_abbr
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 🔐 Load API Key securely
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# Initialize YouTube API
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Load RoBERTa model & tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Title
st.title("📊 YouTube Sentiment Insights")

# Input Channel ID
channel_id = st.text_input("Enter YouTube Channel ID:")

# Fetch Channel Name
def get_channel_name(cid):
    try:
        data = youtube.channels().list(part="snippet", id=cid).execute()
        return data['items'][0]['snippet']['title']
    except:
        return "Unknown Channel"

# Fetch Video IDs
def get_video_ids(cid, max_results=50):
    res = youtube.search().list(
        part="snippet",
        channelId=cid,
        maxResults=max_results,
        order="date",
        type="video"
    ).execute()
    return [item['id']['videoId'] for item in res['items']]

# Fetch Comments
def get_comments(vid):
    comments = []
    try:
        res = youtube.commentThreads().list(
            part="snippet",
            videoId=vid,
            maxResults=50,
            textFormat="plainText"
        ).execute()
        for item in res['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except:
        pass
    return comments

# Get video stats
def get_video_details(video_ids):
    stats = []
    for i in range(0, len(video_ids), 50):
        res = youtube.videos().list(
            part='statistics,snippet',
            id=','.join(video_ids[i:i+50])
        ).execute()
        for item in res['items']:
            stats.append({
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'published_at': item['snippet']['publishedAt'],
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0))
            })
    return pd.DataFrame(stats)

# Sentiment analysis
def analyze_sentiment(comments):
    results = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    filtered = [c for c in comments if len(c.strip()) > 5]
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    for comment in filtered[:300]:
        try:
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            with torch.no_grad():
                output = model(**inputs)
                probs = F.softmax(output.logits, dim=1)
                label_id = torch.argmax(probs).item()
                sentiment = label_map[label_id]
                results[sentiment] += 1
        except:
            continue
    return results

# Month extractor
def extract_month(date_str):
    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").strftime('%b')

# MAIN LOGIC
if channel_id:
    st.info("🔄 Fetching data...")

    try:
        channel_name = get_channel_name(channel_id)
        st.subheader(f"📺 Channel: {channel_name}")

        video_ids = get_video_ids(channel_id)
        video_df = get_video_details(video_ids)

        total_views = video_df['views'].sum()
        total_likes = video_df['likes'].sum()

        st.markdown(f"**👀 Total Views (Last 50 videos):** `{total_views}`")
        st.markdown(f"**❤️ Total Likes (Last 50 videos):** `{total_likes}`")

        # Comments & Sentiment
        all_comments = []
        for vid in video_ids:
            all_comments.extend(get_comments(vid))

        sentiment_result = analyze_sentiment(all_comments)
        label_map = {
            "POSITIVE": "😊 Positive",
            "NEUTRAL": "😐 Neutral",
            "NEGATIVE": "😡 Negative"
        }

        st.subheader("🔍 Sentiment Distribution")

        pie = px.pie(
            names=[label_map[k] for k in sentiment_result],
            values=list(sentiment_result.values()),
            color=[label_map[k] for k in sentiment_result],
            color_discrete_map={
                "😊 Positive": "#80B1D3",
                "😐 Neutral": "#FDB462",
                "😡 Negative": "#FB8072"
            },
            title="Comment Sentiment"
        )
        st.plotly_chart(pie)

        # Monthly Views
        st.subheader("📈 Monthly Views")

        video_df['month'] = video_df['published_at'].apply(extract_month)

        # Dynamically build month list up to current month
        current_month_num = datetime.now().month
        month_order = list(month_abbr)[1:current_month_num + 1]  # ['Jan', ..., current month]
        video_df['month'] = pd.Categorical(video_df['month'], categories=month_order, ordered=True)

        monthly_views = video_df.groupby('month')['views'].sum().reset_index().sort_values('month')

        line = px.line(
            monthly_views,
            x='month',
            y='views',
            markers=True,
            title="Monthly Views Overview"
        )
        st.plotly_chart(line)

    except Exception as e:
        st.error(f"❌ Error: {e}")
