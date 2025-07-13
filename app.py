import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 🔐 Use secret key from Streamlit secrets
api_key = st.secrets["api_key"]

# 📺 Initialize YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

# 🚀 Get channel details
def get_channel_stats(youtube, channel_username):
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        forUsername=channel_username
    )
    response = request.execute()

    if not response['items']:
        request = youtube.channels().list(
            part='snippet,contentDetails,statistics',
            id=channel_username
        )
        response = request.execute()

    if not response['items']:
        return None

    return response['items'][0]

# 📦 Get uploads playlist ID
def get_uploads_playlist_id(channel_id):
    res = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()
    return res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

# 🎞️ Get all video IDs
def get_all_video_ids_from_playlist(playlist_id):
    video_ids = []
    next_page_token = None
    while True:
        res = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        video_ids += [item['contentDetails']['videoId'] for item in res['items']]
        next_page_token = res.get('nextPageToken')
        if not next_page_token:
            break
    return video_ids

# 📊 Get video details
def get_video_details(video_ids):
    all_data = []
    for i in range(0, len(video_ids), 50):
        response = youtube.videos().list(
            part="snippet,statistics",
            id=','.join(video_ids[i:i+50])
        ).execute()
        for video in response['items']:
            published_at = video['snippet']['publishedAt']
            published_month = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m")
            data = {
                'title': video['snippet']['title'],
                'views': int(video['statistics'].get('viewCount', 0)),
                'likes': int(video['statistics'].get('likeCount', 0)),
                'comments': int(video['statistics'].get('commentCount', 0)),
                'month': published_month
            }
            all_data.append(data)
    return pd.DataFrame(all_data)

# 🌟 Streamlit UI
st.set_page_config(page_title="YouTube Channel Analyzer 🎬", layout="centered")
st.title("📊 YouTube Channel Analyzer")
st.write("Enter a **channel username** or **channel ID** below:")

channel_input = st.text_input("🔍 Channel Name or ID")

if channel_input:
    channel_data = get_channel_stats(youtube, channel_input)
    if not channel_data:
        st.error("❌ Channel not found. Please check the name or ID.")
    else:
        st.success("✅ Channel found!")
        st.subheader(f"📺 {channel_data['snippet']['title']}")
        
        subscribers = int(channel_data['statistics'].get('subscriberCount', 0))
        st.write(f"👥 Subscribers: **{subscribers:,}**")

        channel_id = channel_data['id']
        uploads_playlist_id = get_uploads_playlist_id(channel_id)
        all_video_ids = get_all_video_ids_from_playlist(uploads_playlist_id)

        st.write(f"🎞️ Total Videos: **{len(all_video_ids)}**")

        df_videos = get_video_details(all_video_ids)

        total_views = df_videos['views'].sum()
        total_likes = df_videos['likes'].sum()
        total_comments = df_videos['comments'].sum()

        # 🍰 Total Performance Pie Chart
        st.markdown("### 📌 Overall Performance")
        chart_data = {
            '👁️ Views': total_views,
            '👍 Likes': total_likes,
            '💬 Comments': total_comments
        }

        fig1, ax1 = plt.subplots()
        ax1.pie(chart_data.values(), labels=chart_data.keys(), autopct='%1.1f%%', startangle=90,
                colors=['#87CEEB', '#90EE90', '#FFB6C1'])
        ax1.axis('equal')
        st.pyplot(fig1)

        # 📅 Month-wise Views Chart
        st.markdown("### 📆 Monthly Views Trend")

        monthly_views = df_videos.groupby('month')['views'].sum().sort_index()
        monthly_df = monthly_views.reset_index()
        monthly_df['pct_change'] = monthly_df['views'].pct_change().fillna(0) * 100

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=monthly_df, x='month', y='views', marker='o', ax=ax2)
        ax2.set_ylabel("Views")
        ax2.set_xlabel("Month")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # 🔺🔻 View Changes by Month
        st.markdown("### 📈 View Changes by Month")
        for i in range(1, len(monthly_df)):
            current = monthly_df.iloc[i]
            previous = monthly_df.iloc[i - 1]
            diff = current['views'] - previous['views']
            arrow = "🔺" if diff > 0 else "🔻"
            percent = current['pct_change']
            st.write(f"{current['month']}: {arrow} {abs(percent):.2f}% ({current['views']:,} views)")

        # 🍰 Monthly Views Pie Chart
        st.markdown("### 🧁 Monthly Views Contribution")
        fig3, ax3 = plt.subplots()
        ax3.pie(monthly_views, labels=monthly_views.index, autopct='%1.1f%%', startangle=90,
                colors=plt.cm.Pastel1.colors)
        ax3.axis('equal')
        st.pyplot(fig3)

        # 📋 Video table
        with st.expander("📋 Show All Video Stats"):
            st.dataframe(df_videos)
