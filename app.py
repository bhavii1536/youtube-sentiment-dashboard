import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime
import calendar

# Load API key securely from Streamlit secrets
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# Set up YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Page config
st.set_page_config(page_title="YouTube Channel Analyzer 💡", layout="wide")

st.title("📺 YouTube Channel Analysis Dashboard")

# Input field for channel ID
channel_id = st.text_input("🔎 Enter YouTube Channel ID:")

if channel_id:
    # Function to get uploads playlist ID
    def get_uploads_playlist_id(channel_id):
        response = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        return response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Get video IDs from uploads playlist
    def get_video_ids(playlist_id, max_videos=50):
        video_ids = []
        next_page_token = None
        while len(video_ids) < max_videos:
            res = youtube.playlistItems().list(
                part='snippet',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            for item in res['items']:
                video_ids.append(item['snippet']['resourceId']['videoId'])
                if len(video_ids) >= max_videos:
                    break
            next_page_token = res.get('nextPageToken')
            if not next_page_token:
                break
        return video_ids

    # Get video statistics and comments
    def get_video_stats_and_comments(video_ids):
        stats_list = []
        sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for vid in video_ids:
            response = youtube.videos().list(
                part='snippet,statistics',
                id=vid
            ).execute()

            item = response['items'][0]
            snippet = item['snippet']
            stats = item['statistics']

            published_at = snippet['publishedAt']
            view_count = int(stats.get('viewCount', 0))
            like_count = int(stats.get('likeCount', 0))

            # Comments
            try:
                comments_res = youtube.commentThreads().list(
                    part='snippet',
                    videoId=vid,
                    maxResults=20,
                    textFormat="plainText"
                ).execute()

                for comment in comments_res['items']:
                    text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    analysis = TextBlob(text)
                    polarity = analysis.sentiment.polarity
                    if polarity > 0:
                        sentiment_scores['Positive'] += 1
                    elif polarity < 0:
                        sentiment_scores['Negative'] += 1
                    else:
                        sentiment_scores['Neutral'] += 1
            except:
                pass

            stats_list.append({
                'Video ID': vid,
                'Published At': published_at,
                'Views': view_count,
                'Likes': like_count
            })
        return stats_list, sentiment_scores

    try:
        playlist_id = get_uploads_playlist_id(channel_id)
        video_ids = get_video_ids(playlist_id)
        stats, sentiments = get_video_stats_and_comments(video_ids)

        df = pd.DataFrame(stats)
        df['Published At'] = pd.to_datetime(df['Published At'])
        df['Month'] = df['Published At'].dt.month
        df['Month Name'] = df['Month'].apply(lambda x: calendar.month_abbr[x])

        # Group by month
        monthly_views = df.groupby('Month Name')['Views'].sum().reindex(
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']).dropna()

        st.markdown("### 📊 Monthly Views Overview")
        fig, ax = plt.subplots(figsize=(10, 4))
        monthly_views.plot(kind='line', marker='o', color='green', ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Views")
        ax.set_title("Views per Month 📈")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("### 💬 Sentiment Analysis (from recent comments)")
        sentiment_labels = list(sentiments.keys())
        sentiment_values = list(sentiments.values())

        pie_colors = ['#2ecc71', '#e74c3c', '#f1c40f']
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%', colors=pie_colors, startangle=140)
        ax2.axis('equal')
        st.pyplot(fig2)

        st.markdown("### 📌 Summary Metrics")
        total_views = df['Views'].sum()
        total_likes = df['Likes'].sum()

        col1, col2 = st.columns(2)
        col1.metric("👍 Total Likes", f"{total_likes:,}")
        col2.metric("👁️ Total Views", f"{total_views:,}")

        with st.expander("📄 Raw Data Table (Latest 50 Videos)"):
            st.dataframe(df[['Video ID', 'Published At', 'Views', 'Likes']])

    except Exception as e:
        st.error(f"Something went wrong! 😢\n\n{str(e)}")
