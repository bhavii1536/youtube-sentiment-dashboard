import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from textblob import TextBlob
import time
import matplotlib.pyplot as plt

API_KEY = "AIzaSyBClkFCKIDKI4eAL79bNzAIpHRTlT58uuM"  # 🔁 Replace with your real API key
CHANNEL_ID = "UCrU83y4nqxHQOkmBIid8IBg"  # 🔁 Replace with the YouTube channel ID

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_ids(channel_id):
    video_ids = []
    next_page_token = None
    while True:
        res = youtube.search().list(
            part='id',
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type='video'
        ).execute()
        for item in res['items']:
            video_ids.append(item['id']['videoId'])
        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break
    return video_ids

def get_comments(video_id):
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        ).execute()
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except Exception as e:
        st.warning(f"Error fetching comments for video {video_id}: {e}")
    return comments

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_video_stats(video_ids):
    stats = {}
    for i in range(0, len(video_ids), 50):
        res = youtube.videos().list(
            part='statistics',
            id=','.join(video_ids[i:i+50])
        ).execute()
        for item in res['items']:
            vid = item['id']
            stats[vid] = item['statistics']
    return stats

def main():
    st.title("📊 YouTube Channel Sentiment & Engagement Dashboard")

    with st.spinner("Fetching video IDs..."):
        video_ids = get_video_ids(CHANNEL_ID)
    st.success(f"🎥 Found {len(video_ids)} videos.")

    limit = st.slider("Number of videos to analyze", 1, min(50, len(video_ids)), 10)

    if st.button("Analyze Comments"):
        final_data = []
        limited_video_ids = video_ids[:limit]

        progress_bar = st.progress(0)
        for idx, vid in enumerate(limited_video_ids):
            comments = get_comments(vid)
            for comment in comments:
                sentiment = analyze_sentiment(comment)
                final_data.append({
                    "video_id": vid,
                    "comment": comment,
                    "sentiment": sentiment
                })
            progress_bar.progress((idx + 1) / limit)
            time.sleep(0.2)

        df = pd.DataFrame(final_data)
        st.write(f"💬 Total comments analyzed: {len(df)}")

        # Sentiment Pie Chart
        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c', '#95a5a6']
        )
        st.subheader("📈 Sentiment Distribution")
        st.pyplot(fig)

        # Engagement stats
        video_stats = get_video_stats(limited_video_ids)
        total_views = 0
        total_likes = 0
        for vid in limited_video_ids:
            stats = video_stats.get(vid, {})
            total_views += int(stats.get('viewCount', 0))
            total_likes += int(stats.get('likeCount', 0))

        st.subheader("📺 Overall Engagement")
        st.write(f"👁️ Total Views: **{total_views:,}**")
        st.write(f"👍 Total Likes: **{total_likes:,}**")

        if st.checkbox("🔍 Show detailed comment data"):
            st.dataframe(df)

if __name__ == "__main__":
    main()
