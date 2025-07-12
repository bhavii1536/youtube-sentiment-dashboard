import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

# --- BEAUTIFY with CSS: gradient bg + fonts + input styling ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #fceabb, #f8b500);
    }
    .reportview-container {
        background: linear-gradient(135deg, #e0c3fc, #8ec5fc);
        color: #222;
        font-family: 'Quicksand', sans-serif;
    }
    header, footer {visibility: hidden;}
    h1, h2, h3, h4, h5 {
        font-family: 'Quicksand', sans-serif;
        color: #222;
    }
    .stTextInput>div>div>input {
        background-color: #fff0db;
        border: 2px solid #f8b500;
        border-radius: 10px;
        padding: 8px;
        font-size: 18px;
        font-weight: 600;
        color: #222;
    }
    </style>
""", unsafe_allow_html=True)

API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_name(channel_id):
    try:
        res = youtube.channels().list(part='snippet', id=channel_id).execute()
        return res['items'][0]['snippet']['title'] if res['items'] else "Unknown Channel"
    except:
        return "Unknown Channel"

def get_video_ids(channel_id):
    video_ids = []
    next_page_token = None
    try:
        while True:
            res = youtube.search().list(
                part='id',
                channelId=channel_id,
                maxResults=50,
                pageToken=next_page_token,
                type='video',
                order='date'
            ).execute()
            for item in res['items']:
                video_ids.append(item['id']['videoId'])
            next_page_token = res.get('nextPageToken')
            if not next_page_token:
                break
    except:
        return []
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
    except:
        pass
    return comments

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def plot_pie_chart(sentiment_counts):
    colors = {
        'Positive': '#6BCB77',
        'Negative': '#FF6B6B',
        'Neutral': '#A0AEC0'
    }

    df = sentiment_counts.reset_index()
    df.columns = ['Sentiment', 'Count']

    fig = px.pie(df, values='Count', names='Sentiment',
                 color='Sentiment', color_discrete_map=colors,
                 hole=0, title="Sentiment Distribution")
    fig.update_traces(
        textinfo='percent+label',
        marker=dict(line=dict(color='white', width=2)),
        hovertemplate='<b>%{label}</b><br>Comments: %{value}<br>Percent: %{percent}'
    )
    return fig

def main():
    st.title("📊 YouTube Sentiment & Engagement Dashboard")

    channel_id = st.text_input("📺 Enter YouTube Channel ID:")

    if channel_id:
        channel_name = get_channel_name(channel_id)
        st.subheader(f"✨ Channel: {channel_name}")

        all_video_ids = get_video_ids(channel_id)
        total_videos_found = len(all_video_ids)
        st.success(f"🎬 Found {total_videos_found} videos.")

        if not all_video_ids:
            st.error("No videos found.")
            return

        num_videos = st.slider("📌 Videos to analyze", 1, min(50, total_videos_found), 10)
        selected_video_ids = all_video_ids[:num_videos]

        st.info("Analyzing comments...")

        final_data = []
        total_views = 0
        total_likes = 0

        for vid in selected_video_ids:
            comments = get_comments(vid)
            for comment in comments:
                sentiment = analyze_sentiment(comment)
                final_data.append({"video_id": vid, "comment": comment, "sentiment": sentiment})
            try:
                stats = youtube.videos().list(part="statistics", id=vid).execute()
                if stats["items"]:
                    stat = stats["items"][0]["statistics"]
                    total_views += int(stat.get("viewCount", 0))
                    total_likes += int(stat.get("likeCount", 0))
            except:
                pass

        if not final_data:
            st.warning("No comments found.")
            return

        df = pd.DataFrame(final_data)
        sentiment_counts = df['sentiment'].value_counts()
        total_comments = len(df)
        positive = sentiment_counts.get("Positive", 0)
        negative = sentiment_counts.get("Negative", 0)
        neutral = sentiment_counts.get("Neutral", 0)

        def percent(part): return f"{(part / total_comments * 100):.2f}%" if total_comments > 0 else "0.00%"

        # Show pie chart
        st.plotly_chart(plot_pie_chart(sentiment_counts), use_container_width=True)

        # Print summary text below chart
        st.markdown("### 📑 Final Summary Report")
        st.text(f"===== Overall Sentiment Summary =====")
        st.text(f"Total Comments Analyzed: {total_comments}")
        st.text(f"Positive: {positive} ({percent(positive)})")
        st.text(f"Negative: {negative} ({percent(negative)})")
        st.text(f"Neutral: {neutral} ({percent(neutral)})")

        st.text(f"===== Overall Channel Engagement =====")
        st.text(f"Total Views on {num_videos} videos: {total_views}")
        st.text(f"Total Likes on {num_videos} videos: {total_likes}")

if __name__ == "__main__":
    main()
