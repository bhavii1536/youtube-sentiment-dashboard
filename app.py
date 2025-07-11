import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import concurrent.futures
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="YouTube Sentiment & Engagement Dashboard", layout="wide", initial_sidebar_state='expanded')

# API key from Streamlit secrets (store your key safely in Streamlit Cloud)
API_KEY = st.secrets["AIzaSyBClkFCKIDKI4eAL79bNzAIpHRTlT58uuM"]
youtube = build('youtube', 'v3', developerKey=API_KEY)
analyzer = SentimentIntensityAnalyzer()

@st.cache_data(show_spinner=False)
def get_video_ids(channel_id):
    video_ids = []
    next_page_token = None
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
    return video_ids

@st.cache_data(show_spinner=False)
def get_comments(video_id, max_comments):
    comments = []
    try:
        results = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=max_comments
        ).execute()
        for item in results.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except:
        pass
    return comments

@st.cache_data(show_spinner=False)
def get_video_stats(video_ids):
    stats = {}
    for vid in video_ids:
        try:
            response = youtube.videos().list(part='statistics', id=vid).execute()
            if response['items']:
                stats[vid] = response['items'][0]['statistics']
        except:
            stats[vid] = {}
    return stats

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def plot_sentiment_chart(sentiment_counts, chart_type):
    # Use RGBA for gradient-like transparency effect
    colors = {
        'Positive': 'rgba(74, 144, 226, 0.9)',   # Deep Blue (semi-transparent)
        'Negative': 'rgba(255, 111, 97, 0.9)',   # Coral Red
        'Neutral': 'rgba(169, 204, 227, 0.9)'    # Light Sky Blue
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
            opacity=0.85  # simulate gradient transparency
        )
        fig.update_layout(bargap=0.4)

    # Shared layout tweaks for all charts
    fig.update_layout(
        title_font=dict(size=22, family='Arial'),
        font=dict(size=14, family='Arial'),
        legend_title=None,
        plot_bgcolor='rgba(0,0,0,0)',      # transparent background
        paper_bgcolor='rgba(255,255,255,1)' # white paper background
    )

    return fig

def generate_wordcloud(text, title):
    if not text.strip():
        st.write(f"No {title.lower()} comments to display.")
        return
    wc = WordCloud(width=500, height=250,
                   background_color='white',
                   colormap='Blues').generate(text)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.subheader(title)
    st.pyplot(fig)

def main():
    st.markdown("<h1 style='text-align:center; color:#1E40AF; font-weight:bold;'>📊 YouTube Sentiment & Engagement Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    channel_id = st.text_input("Enter YouTube Channel ID (e.g. UCrU83y4nqkmBIid8IBg):")

    if channel_id:
        with st.spinner("Fetching videos..."):
            video_ids = get_video_ids(channel_id)
        st.success(f"🎥 Found {len(video_ids)} videos.")

        max_videos = st.slider("Number of videos to analyze", 1, min(len(video_ids), 100), 10)
        max_comments = st.slider("Number of comments per video", 10, 100, 50)

        chart_type = st.selectbox("Select Sentiment Chart Type", ['Pie Chart', 'Donut Chart', 'Bar Chart'])

        if st.button("Analyze Comments"):
            limited_video_ids = video_ids[:max_videos]

            progress_bar = st.progress(0)
            final_data = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(get_comments, vid, max_comments): vid for vid in limited_video_ids}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    vid = futures[future]
                    comments = future.result()
                    for comment in comments:
                        sentiment = analyze_sentiment(comment)
                        final_data.append({
                            "video_id": vid,
                            "comment": comment,
                            "sentiment": sentiment
                        })
                    progress_bar.progress((i + 1) / len(limited_video_ids))

            df = pd.DataFrame(final_data)
            total_comments = len(df)

            video_stats = get_video_stats(limited_video_ids)
            total_views = sum(int(stats.get('viewCount', 0)) for stats in video_stats.values())
            total_likes = sum(int(stats.get('likeCount', 0)) for stats in video_stats.values())

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 💬 Total comments analyzed: **{total_comments:,}**")
                st.markdown(f"### 👁️ Total Views: **{total_views:,}**")
                st.markdown(f"### 👍 Total Likes: **{total_likes:,}**")

                sentiment_counts = df['sentiment'].value_counts()
                sentiment_percent = (sentiment_counts / sentiment_counts.sum() * 100).round(2)

                fig = plot_sentiment_chart(sentiment_counts, chart_type)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**Positive:** {sentiment_percent.get('Positive', 0)}%  |  "
                            f"**Negative:** {sentiment_percent.get('Negative', 0)}%  |  "
                            f"**Neutral:** {sentiment_percent.get('Neutral', 0)}%")

                engagement_rate = (total_likes / total_views * 100) if total_views > 0 else 0
                st.markdown(f"### 📊 Engagement Rate (Likes / Views): **{engagement_rate:.2f}%**")

            with col2:
                positive_comments = " ".join(df[df.sentiment == 'Positive']['comment'])
                negative_comments = " ".join(df[df.sentiment == 'Negative']['comment'])

                generate_wordcloud(positive_comments, "Positive Comments Word Cloud")
                generate_wordcloud(negative_comments, "Negative Comments Word Cloud")

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Comments & Sentiments CSV",
                data=csv,
                file_name='youtube_sentiment.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
