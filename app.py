import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import traceback

# Initialize YouTube API client using API key from Streamlit secrets
API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_channel_name(channel_id):
    try:
        res = youtube.channels().list(
            part='snippet',
            id=channel_id
        ).execute()
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
    except HttpError as e:
        st.error(f"Google API error: HTTP {e.resp.status}\n{e.content.decode('utf-8') if hasattr(e.content, 'decode') else e.content}")
        st.text(traceback.format_exc())
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.text(traceback.format_exc())
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
    except HttpError as e:
        st.warning(f"Could not fetch comments for video {video_id}. Error: {e}")
    except Exception as e:
        st.warning(f"Unexpected error fetching comments: {e}")
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

def plot_sentiment_chart(sentiment_counts, chart_type):
    colors = {
        'Positive': '#6BCB77',   # Pastel green
        'Negative': '#FF6B6B',   # Coral red
        'Neutral': '#A0AEC0'     # Cool gray-blue
    }

    df = sentiment_counts.reset_index()
    df.columns = ['Sentiment', 'Count']

    if chart_type == 'Pie Chart':
        fig = px.pie(df, values='Count', names='Sentiment',
                     color='Sentiment',
                     color_discrete_map=colors,
                     hole=0,
                     title="Sentiment Distribution")
        fig.update_traces(
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2)),
            hovertemplate='<b>%{label}</b><br>Comments: %{value}<br>Percent: %{percent}'
        )

    elif chart_type == 'Donut Chart':
        fig = px.pie(df, values='Count', names='Sentiment',
                     color='Sentiment',
                     color_discrete_map=colors,
                     hole=0.5,
                     title="Sentiment Distribution")
        fig.update_traces(
            textinfo='percent+label',
            marker=dict(line=dict(color='white', width=2)),
            hovertemplate='<b>%{label}</b><br>Comments: %{value}<br>Percent: %{percent}'
        )

    elif chart_type == 'Bar Chart':
        fig = px.bar(df, x='Sentiment', y='Count',
                     color='Sentiment',
                     color_discrete_map=colors,
                     text='Count',
                     title="Sentiment Counts")
        fig.update_traces(
            textposition='outside',
            marker_line_color='black',
            marker_line_width=1.5,
            opacity=0.85
        )
        fig.update_layout(bargap=0.4)

    fig.update_layout(
        title_font=dict(size=22, family='Arial'),
        font=dict(size=14, family='Arial'),
        legend_title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(255,255,255,1)'
    )

    return fig

def main():
    st.title("📊 YouTube Sentiment & Engagement Dashboard")

    channel_id = st.text_input("Enter YouTube Channel ID (e.g. UC_x5XG1OV2P6uZZ5FSM9Ttw):")

    if channel_id:
        channel_name = get_channel_name(channel_id)
        st.subheader(f"📺 Channel: {channel_name}")

        st.info("Fetching videos and comments, please wait...")
        video_ids = get_video_ids(channel_id)

        if not video_ids:
            st.error("No videos found or API error occurred.")
            return

        num_videos = st.slider("Number of videos to analyze", 1, min(50, len(video_ids)), 10)
        video_ids = video_ids[:num_videos]

        final_data = []

        for vid in video_ids:
            comments = get_comments(vid)
            for comment in comments:
                sentiment = analyze_sentiment(comment)
                final_data.append({"video_id": vid, "comment": comment, "sentiment": sentiment})

        if not final_data:
            st.warning("No comments found for the selected videos.")
            return

        df = pd.DataFrame(final_data)

        st.write(f"💬 Total comments analyzed: {len(df)}")

        sentiment_counts = df['sentiment'].value_counts()

        chart_type = st.selectbox("Choose sentiment chart type", ['Pie Chart', 'Donut Chart', 'Bar Chart'])

        fig = plot_sentiment_chart(sentiment_counts, chart_type)
        st.plotly_chart(fig, use_container_width=True)

        st.write("📊 Overall Engagement")
        st.write(f"👁️ Total Videos Analyzed: {num_videos}")

if __name__ == "__main__":
    main()
