import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from textblob import TextBlob
import time

# Load API Key securely
api_key = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=api_key)

# =========================
# FUNCTIONS
# =========================

def get_channel_stats(channel_id):
    request = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id
    )
    response = request.execute()

    if 'items' not in response or not response['items']:
        return None

    data = response['items'][0]
    stats = {
        "Channel Name": data['snippet']['title'],
        "Subscribers": int(data['statistics'].get('subscriberCount', 0)),
        "Total Views": int(data['statistics'].get('viewCount', 0)),
        "Total Videos": int(data['statistics'].get('videoCount', 0)),
        "Channel Created": data['snippet']['publishedAt'].split("T")[0]
    }
    return stats

def get_latest_videos(channel_id, max_results=5):
    # Get uploads playlist
    response = youtube.channels().list(part="contentDetails", id=channel_id).execute()
    if 'items' not in response or not response['items']:
        return []
    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Get latest video IDs
    playlist_response = youtube.playlistItems().list(
        part="snippet",
        playlistId=uploads_playlist_id,
        maxResults=max_results
    ).execute()

    videos = []
    for item in playlist_response['items']:
        video_id = item['snippet']['resourceId']['videoId']
        title = item['snippet']['title']
        published = item['snippet']['publishedAt']

        video_response = youtube.videos().list(
            part="statistics,snippet",
            id=video_id
        ).execute()

        if 'items' in video_response and video_response['items']:
            stats = video_response['items'][0]['statistics']
            videos.append({
                "Title": title,
                "Video ID": video_id,
                "Published Date": published.split("T")[0],
                "Views": int(stats.get("viewCount", 0)),
                "Likes": int(stats.get("likeCount", 0)),
                "Comments": int(stats.get("commentCount", 0))
            })
    return videos

def get_video_comments(video_id, max_comments=50):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            textFormat="plainText"
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    except Exception as e:
        print("Error:", e)
    return comments

def analyze_sentiment(comment):
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "😊 Positive"
    elif polarity < -0.1:
        return "😞 Negative"
    else:
        return "😐 Neutral"

# =========================
# UI
# =========================

st.set_page_config(page_title="📊 YouTube Channel Analyzer with Sentiment", layout="centered")
st.title("📊 YouTube Channel Analyzer + 🧠 Sentiment Analysis")

channel_id = st.text_input("🔍 Enter YouTube **Channel ID** (not username):")

if channel_id:
    with st.spinner("📡 Fetching channel info..."):
        channel_info = get_channel_stats(channel_id)

    if not channel_info:
        st.error("❌ Channel not found. Please check the ID.")
    else:
        st.success(f"✅ Fetched details for **{channel_info['Channel Name']}**")

        # Overview
        st.markdown("### 📌 Channel Overview")
        col1, col2 = st.columns(2)
        col1.metric("👥 Subscribers", f"{channel_info['Subscribers']:,}")
        col2.metric("📺 Total Videos", f"{channel_info['Total Videos']:,}")
        st.metric("👁️ Total Views", f"{channel_info['Total Views']:,}")
        st.markdown(f"📅 **Channel Created on:** `{channel_info['Channel Created']}`")

        st.markdown("---")

        # Latest Videos
        st.markdown("### 🎥 Latest Videos")
        with st.spinner("📡 Loading videos..."):
            latest_videos = get_latest_videos(channel_id, max_results=5)

        if latest_videos:
            video_df = pd.DataFrame(latest_videos)
            st.dataframe(video_df)

            # Choose a video to analyze
            selected_video = st.selectbox("🧠 Select a video to analyze comments:", video_df["Title"].tolist())
            video_id = video_df[video_df["Title"] == selected_video]["Video ID"].values[0]

            st.markdown(f"#### 💬 Sentiment Analysis for: **{selected_video}**")
            with st.spinner("Analyzing comments..."):
                comments = get_video_comments(video_id)
                if comments:
                    sentiment_data = [{"Comment": c, "Sentiment": analyze_sentiment(c)} for c in comments]
                    sentiment_df = pd.DataFrame(sentiment_data)

                    # Show results
                    st.dataframe(sentiment_df)

                    # Pie Chart
                    pie = sentiment_df['Sentiment'].value_counts().reset_index()
                    pie.columns = ['Sentiment', 'Count']
                    st.markdown("#### 🥧 Sentiment Distribution")
                    st.plotly_chart({
                        "data": [{
                            "labels": pie["Sentiment"],
                            "values": pie["Count"],
                            "type": "pie",
                            "hole": .4
                        }],
                        "layout": {"title": "Comment Sentiment Breakdown"}
                    })

                else:
                    st.warning("No comments found or comments are disabled for this video.")
        else:
            st.warning("No videos found or uploads unavailable.")
else:
    st.info("👆 Enter a **valid Channel ID** to start analysis.")
