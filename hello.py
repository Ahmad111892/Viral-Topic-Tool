import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import networkx as nx
from datetime import datetime, timedelta
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# --- Mathematical Constants and Functions (Euler/Gauss/Newton Inspired) ---
PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio

def fibonacci_growth_model(n):
    """Fibonacci-based growth prediction model"""
    # Clamping n to a reasonable upper bound to prevent overflow
    n = min(n, 90)
    return int((PHI**n - (1-PHI)**n) / np.sqrt(5))

def gaussian_viral_probability(engagement_rate, mean_engagement=2.0, std_engagement=1.0):
    """Calculate viral probability using Gaussian distribution"""
    return stats.norm.cdf(engagement_rate, mean_engagement, std_engagement)

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 Enhanced YouTube Growth Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a more polished interface ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background-color: #e8f5e8;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #4CAF50;
    }
    .einstein-quote {
        font-style: italic;
        color: #666;
        border-left: 3px solid #2196F3;
        padding-left: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# --- Advanced Analytics Classes ---
class ViralPatternAnalyzer:
    """Advanced mathematical analysis of viral patterns"""

    def __init__(self):
        self.growth_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def analyze_growth_patterns(self, channel_data):
        """Analyze growth patterns using machine learning"""
        if len(channel_data) < 5:
            return {"pattern": "insufficient_data", "prediction": 0}

        features, targets = [], []
        for channel in channel_data:
            features.append([
                channel.get('Subscribers', 0),
                channel.get('Total Views', 0),
                channel.get('Video Count', 0),
                channel.get('Engagement Score', 0),
                channel.get('Content Velocity', 0),
                self._days_since_creation(channel.get('Creation Date', ''))
            ])
            targets.append(channel.get('Subscribers', 0))

        features, targets = np.array(features), np.array(targets)
        if len(features) < 2:
            return {"pattern": "insufficient_data", "prediction": 0}

        try:
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)
            if len(X_train) == 0:
                return {"pattern": "insufficient_data", "prediction": 0}

            self.growth_model.fit(X_train, y_train)
            y_pred = self.growth_model.predict(X_test) if len(X_test) > 0 else [0]
            r2 = r2_score(y_test, y_pred) if len(X_test) > 0 else 0

            return {
                "pattern": "exponential" if r2 > 0.7 else "linear" if r2 > 0.4 else "irregular",
                "prediction_accuracy": r2,
                "feature_importance": dict(zip([
                    "subscribers", "views", "videos", "engagement", "velocity", "age"
                ], self.growth_model.feature_importances_))
            }
        except Exception as e:
            st.error(f"Error during pattern analysis: {e}")
            return {"pattern": "analysis_error", "error": str(e)}

    def _days_since_creation(self, creation_date_str):
        try:
            creation_date = datetime.strptime(creation_date_str, "%Y-%m-%d")
            return (datetime.now() - creation_date).days
        except (ValueError, TypeError):
            return 0

    def predict_viral_potential(self, channel_metrics):
        engagement_score = channel_metrics.get('Engagement Score', 0)
        velocity = channel_metrics.get('Content Velocity', 0)
        viral_prob = gaussian_viral_probability(engagement_score)
        growth_factor = fibonacci_growth_model(min(velocity, 10)) / 1000.0
        einstein_factor = engagement_score * (velocity ** 2) / 100.0
        combined_score = (viral_prob * 0.4 + growth_factor * 0.3 + einstein_factor * 0.3)
        return {
            "viral_probability": min(viral_prob, 1.0),
            "growth_prediction": growth_factor,
            "einstein_engagement": einstein_factor,
            "combined_viral_score": min(combined_score, 1.0)
        }

class NetworkAnalyzer:
    """Network analysis of YouTube channels"""
    def __init__(self):
        self.graph = nx.Graph()

    def build_channel_network(self, channels_data):
        self.graph.clear()
        for i, channel in enumerate(channels_data):
            self.graph.add_node(i, **channel)
        for i in range(len(channels_data)):
            for j in range(i + 1, len(channels_data)):
                similarity = self._calculate_similarity(channels_data[i], channels_data[j])
                if similarity > 0.5:
                    self.graph.add_edge(i, j, weight=similarity)

    def _calculate_similarity(self, ch1, ch2):
        m1 = np.array([ch1.get(k, 0) for k in ['Subscribers', 'Engagement Score', 'Content Velocity']])
        m2 = np.array([ch2.get(k, 0) for k in ['Subscribers', 'Engagement Score', 'Content Velocity']])
        norm1, norm2 = np.linalg.norm(m1), np.linalg.norm(m2)
        if norm1 == 0 or norm2 == 0: return 0
        return np.dot(m1, m2) / (norm1 * norm2)

    def get_network_insights(self):
        if len(self.graph.nodes()) < 2:
            return {"clusters": 0, "density": 0, "total_connections": 0}
        try:
            communities = list(nx.connected_components(self.graph))
            return {
                "clusters": len(communities),
                "density": nx.density(self.graph),
                "total_connections": self.graph.number_of_edges()
            }
        except Exception as e:
            st.warning(f"Could not compute network insights: {e}")
            return {"clusters": 0, "density": 0, "total_connections": 0}

@st.cache_data(ttl=3600)
def fetch_youtube_data(url, params):
    """A cached function to fetch data from YouTube API."""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def enhanced_deep_dive_analysis(api_key, channel_id, channel_desc):
    """Performs a deep-dive analysis of a single channel."""
    results = {"Engagement Score": 0, "Content Velocity": 0, "Weekly Frequency": 0, "Sentiment Score": 0, "Mathematical_Growth_Rate": 0}
    
    # Get recent videos
    video_search_params = {"part": "snippet", "channelId": channel_id, "order": "date", "maxResults": 10, "key": api_key}
    video_data = fetch_youtube_data(YOUTUBE_SEARCH_URL, video_search_params)
    if not video_data or not video_data.get("items"): return results

    video_ids = [item["id"]["videoId"] for item in video_data["items"] if "videoId" in item.get("id", {})]
    if not video_ids: return results
    
    # Get video statistics
    video_details_params = {"part": "statistics,snippet", "id": ",".join(video_ids), "key": api_key}
    details_data = fetch_youtube_data(YOUTUBE_VIDEO_URL, video_details_params)
    if not details_data or not details_data.get("items"): return results

    total_likes, total_comments, total_views, view_counts, published_dates, titles = 0, 0, 0, [], [], []
    for item in details_data["items"]:
        stats = item.get("statistics", {})
        total_likes += int(stats.get("likeCount", 0))
        total_comments += int(stats.get("commentCount", 0))
        views = int(stats.get("viewCount", 0))
        total_views += views
        view_counts.append(views)
        published_dates.append(datetime.fromisoformat(item["snippet"]["publishedAt"].replace("Z", "+00:00")))
        titles.append(item["snippet"]["title"])

    if total_views > 0: results["Engagement Score"] = round(((total_likes + total_comments) / total_views) * 100, 2)
    if len(view_counts) > 1:
        x, y = np.arange(len(view_counts)), np.log(np.maximum(view_counts, 1))
        if np.var(x) > 0:
            slope, _ = np.polyfit(x, y, 1)
            results["Mathematical_Growth_Rate"] = round(slope, 4)
    if len(published_dates) > 1:
        time_span_days = (max(published_dates) - min(published_dates)).days
        if time_span_days > 0:
            results["Weekly Frequency"] = round((len(published_dates) / time_span_days) * 7, 1)

    if titles: results["Sentiment Score"] = round(np.mean([TextBlob(t).sentiment.polarity for t in titles]), 3)

    # Monetization clues
    keywords = ["affiliate", "merch", "patreon", "course", "sponsor"]
    results["Monetization Clues"] = [k for k in keywords if k in channel_desc.lower()]

    # Content velocity
    velocity_params = {"part": "id", "channelId": channel_id, "publishedAfter": (datetime.utcnow() - timedelta(days=30)).isoformat("T") + "Z", "maxResults": 50, "key": api_key}
    velocity_data = fetch_youtube_data(YOUTUBE_SEARCH_URL, velocity_params)
    if velocity_data: results["Content Velocity"] = len(velocity_data.get("items", []))
    
    return results

def find_viral_channels(api_key, niches, video_type, use_ai):
    """Finds viral channels based on a list of niches."""
    viral_channels, processed_ids = [], set()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, niche in enumerate(niches):
        status_text.text(f"🔍 Analyzing niche: '{niche}' ({i + 1}/{len(niches)})")
        progress_bar.progress((i + 1) / len(niches))

        search_params = {"part": "snippet", "q": niche, "type": "video", "order": "viewCount", "publishedAfter": (datetime.utcnow() - timedelta(days=90)).isoformat("T") + "Z", "maxResults": 20, "key": api_key}
        if video_type != 'Any':
            search_params['videoDuration'] = 'short' if video_type == 'Shorts Channel' else 'long'
        
        search_results = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_params)
        if not search_results: continue

        channel_ids = {item["snippet"]["channelId"] for item in search_results.get("items", [])}
        new_ids = list(channel_ids - processed_ids)
        if not new_ids: continue

        channel_params = {"part": "snippet,statistics", "id": ",".join(new_ids), "key": api_key}
        channel_data = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
        if not channel_data: continue

        for channel in channel_data.get("items", []):
            published_date = datetime.fromisoformat(channel["snippet"]["publishedAt"].replace("Z", "+00:00"))
            if published_date.year == datetime.now().year:
                stats = channel.get("statistics", {})
                subs, views, vids = int(stats.get("subscriberCount", 0)), int(stats.get("viewCount", 0)), int(stats.get("videoCount", 0))
                if subs > 1000 and views > 50000 and 5 < vids < 100:
                    channel_id = channel['id']
                    analysis_data = enhanced_deep_dive_analysis(api_key, channel_id, channel["snippet"].get("description", ""))
                    viral_channels.append({
                        "Channel Name": channel["snippet"]["title"], "URL": f"https://youtube.com/channel/{channel_id}",
                        "Subscribers": subs, "Total Views": views, "Video Count": vids,
                        "Creation Date": published_date.strftime("%Y-%m-%d"), "Found Via Niche": niche,
                        **analysis_data
                    })
                    processed_ids.add(channel_id)
    
    progress_bar.empty()
    status_text.empty()
    
    if viral_channels and use_ai:
        analyzer = ViralPatternAnalyzer()
        st.session_state.pattern_analysis = analyzer.analyze_growth_patterns(viral_channels)
        network_analyzer = NetworkAnalyzer()
        network_analyzer.build_channel_network(viral_channels)
        st.session_state.network_analysis = network_analyzer.get_network_insights()
        for ch in viral_channels:
            ch.update(analyzer.predict_viral_potential(ch))

    return viral_channels

def display_channel_metrics(channel):
    """Displays metrics for a single channel in an expander."""
    with st.expander(f"{channel['Channel Name']} - {channel['Subscribers']:,} Subscribers", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("👀 Total Views", f"{channel['Total Views']:,}")
        col2.metric("📺 Videos", channel['Video Count'])
        col3.metric("📈 Engagement", f"{channel.get('Engagement Score', 0)}%")
        
        st.markdown(f"🔗 **[Visit Channel]({channel['URL']})**")
        if 'AI_Combined_Score' in channel:
            st.progress(channel['AI_Combined_Score'])
            st.caption(f"AI Viral Score: {channel['AI_Combined_Score']:.3f}")

def main():
    """Main function to run the Streamlit application."""
    st.title("🧠 Enhanced YouTube Growth Toolkit")
    st.markdown("""
    <div class="einstein-quote">
        "The measure of intelligence is the ability to change." - Albert Einstein
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Configuration")
        api_key = st.text_input("YouTube API Key:", type="password", value=st.secrets.get("YOUTUBE_API_KEY", ""))
        use_ai = st.checkbox("Enable AI Predictions", value=True)

    tab1, tab2 = st.tabs(["🔍 Channel Discovery", "🧠 AI Insights Dashboard"])

    with tab1:
        st.header("🚀 Find Viral Channels")
        video_type = st.radio("Channel Type:", ('Any', 'Shorts Channel', 'Long Video Channel'), horizontal=True)
        default_niches = "AI Tools\nPersonal Finance for Gen Z\nSustainable Living\nSide Hustles"
        niche_input = st.text_area("🎯 Enter Niches (one per line):", default_niches, height=120)
        
        if st.button("Start Discovery", type="primary"):
            if not api_key:
                st.error("Please enter your YouTube API key in the sidebar.")
            else:
                niches = [n.strip() for n in niche_input.split('\n') if n.strip()]
                if not niches:
                    st.warning("Please enter at least one niche.")
                else:
                    with st.spinner("AI is analyzing the YouTube-verse..."):
                        channels = find_viral_channels(api_key, niches, video_type, use_ai)
                    st.session_state.discovered_channels = channels
                    if channels:
                        st.success(f"Found {len(channels)} promising channels!")
                        sort_key = "AI_Combined_Score" if use_ai and channels and 'AI_Combined_Score' in channels[0] else "Subscribers"
                        st.session_state.discovered_channels.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
                    else:
                        st.warning("No rapidly growing channels found. Try different niches.")
        
        if 'discovered_channels' in st.session_state and st.session_state.discovered_channels:
            st.markdown("---")
            st.subheader("Discovered Channels")
            for channel in st.session_state.discovered_channels:
                display_channel_metrics(channel)

    with tab2:
        st.header("💡 AI-Powered Insights")
        if 'discovered_channels' not in st.session_state or not st.session_state.discovered_channels:
            st.info("Discover channels in the first tab to see AI insights.")
        else:
            channels = st.session_state.discovered_channels
            if 'pattern_analysis' in st.session_state:
                st.subheader("🔬 Growth Pattern Analysis")
                pattern_data = st.session_state.pattern_analysis
                col1, col2 = st.columns(2)
                col1.metric("Dominant Growth Pattern", pattern_data.get('pattern', 'N/A').title())
                col2.metric("Prediction Accuracy (R²)", f"{pattern_data.get('prediction_accuracy', 0):.1%}")
                
                if 'feature_importance' in pattern_data:
                    st.subheader("🎯 Key Success Factors")
                    imp_df = pd.DataFrame(list(pattern_data['feature_importance'].items()), columns=['Factor', 'Importance'])
                    fig = px.bar(imp_df.sort_values('Importance'), x='Importance', y='Factor', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'network_analysis' in st.session_state:
                st.subheader("🌐 Niche Network Analysis")
                net_data = st.session_state.network_analysis
                col1, col2, col3 = st.columns(3)
                col1.metric("🔗 Clusters Found", net_data.get("clusters", 0))
                col2.metric("Density", f"{net_data.get('density', 0):.3f}")
                col3.metric("Connections", net_data.get("total_connections", 0))

if __name__ == "__main__":
    main()

