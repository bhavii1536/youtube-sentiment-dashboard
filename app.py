import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Set cache directory for HuggingFace Transformers
os.environ["TRANSFORMERS_CACHE"] = "./cache"

# ✅ Load API key securely from Streamlit secrets
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# ✅ Load RoBERTa model and tokenizer with Streamlit cache
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_roberta_model():
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return model, tokenizer

model, tokenizer = load_roberta_model()

# ✅ Analyze sentiment using RoBERTa
def analyze_sentiment_roberta(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    scores = probs.detach().numpy()[0]
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[np.argmax(scores)], scores

# ✅ Get comments from YouTube
def get_comments(youtube, video_id, max_comments=100):
    comments = []
    next_page_token = None
    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# ✅ Extract Video ID from URL
def extract_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[-1]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return None

# ✅ Streamlit UI
st.set_page_config(page_title="YouTube Sentiment Dashboard", layout="wide")
st.title("🎥 YouTube Comment Sentiment Dashboard")
st.markdown("Enter a YouTube video URL to analyze the sentiments of its comments.")

video_url = st.text_input("📺 Paste YouTube Video URL:")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please try again.")
    else:
        st.success("✅ Valid YouTube URL detected. Fetching comments...")
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        comments = get_comments(youtube, video_id, max_comments=100)

        if not comments:
            st.warning("😕 No comments found for this video.")
        else:
            st.info(f"💬 Fetched {len(comments)} comments. Running sentiment analysis...")

            sentiments = []
            for comment in comments:
                label, scores = analyze_sentiment_roberta(comment)
                sentiments.append({
                    "Comment": comment,
                    "Sentiment": label,
                    "Negative": scores[0],
                    "Neutral": scores[1],
                    "Positive": scores[2]
                })

            df = pd.DataFrame(sentiments)
            st.dataframe(df[["Comment", "Sentiment"]], use_container_width=True)

            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Breakdown')
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📈 Show raw scores (optional)"):
                st.dataframe(df, use_container_width=True)
