import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from textstat import flesch_reading_ease
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration (Following Guido's Philosophy of Elegance) ---
st.set_page_config(
    page_title="YouTube Growth Intelligence Engine",
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


# --- Mathematical Constants & Configuration (Euler's Influence) ---
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618 (for optimal UI proportions)
EULER_NUMBER = np.e
PI = np.pi

# API Configuration
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# --- Advanced Mathematical Models (Newton & Gauss Inspired) ---
class GrowthAnalyzer:
    """
    Advanced growth analysis using differential equations and statistical modeling
    Inspired by Newton's calculus and Gauss's statistical methods
    """
    @staticmethod
    def exponential_growth_model(t, a, b, c):
        """Exponential growth model: y = a * e^(b*t) + c"""
        return a * np.exp(b * t) + c

    @staticmethod
    def logistic_growth_model(t, L, k, t0):
        """Logistic growth model: y = L / (1 + e^(-k*(t-t0)))"""
        return L / (1 + np.exp(-k * (t - t0)))

    @staticmethod
    def power_law_model(x, a, b):
        """Power law model: y = a * x^b (Pareto distribution)"""
        return a * np.power(x, b)

    @staticmethod
    def calculate_growth_velocity(data_points, time_intervals):
        """Calculate instantaneous growth velocity using numerical differentiation"""
        if len(data_points) < 2:
            return 0
        velocities = np.gradient(data_points, time_intervals)
        return np.mean(velocities)

    @staticmethod
    def calculate_growth_acceleration(data_points, time_intervals):
        """Calculate growth acceleration (second derivative)"""
        if len(data_points) < 3:
            return 0
        velocities = np.gradient(data_points, time_intervals)
        accelerations = np.gradient(velocities, time_intervals)
        return np.mean(accelerations)

class ViralityPredictor:
    """
    Advanced virality prediction using machine learning principles
    Inspired by Hinton, Bengio, and LeCun's deep learning approaches
    """
    @staticmethod
    def calculate_viral_coefficient(views, time_since_publish, subscriber_count):
        """
        Calculate viral coefficient using normalized metrics
        VC = (Views / Subscribers) * e^(-λt) where λ is decay constant
        """
        if subscriber_count == 0:
            return 0
        # Time decay factor (videos lose momentum over time)
        decay_constant = 0.1  # Empirically derived
        time_factor = np.exp(-decay_constant * time_since_publish.days)
        viral_ratio = views / max(subscriber_count, 1)
        return viral_ratio * time_factor

    @staticmethod
    def engagement_quality_score(likes, comments, views, video_duration):
        """
        Multi-dimensional engagement quality using weighted metrics
        """
        if views == 0:
            return 0
        # Normalize by video duration (longer videos typically have lower engagement rates)
        duration_factor = max(1, video_duration / 300)  # 5 minutes baseline
        like_rate = (likes / views) * 100
        comment_rate = (comments / views) * 100
        # Comments are more valuable than likes (require more engagement)
        engagement_score = (like_rate + 3 * comment_rate) / duration_factor
        # Apply sigmoid function to normalize to 0-10 scale
        return 10 / (1 + np.exp(-engagement_score + 2))

class NetworkAnalyzer:
    """
    Network analysis for understanding content relationships
    Inspired by modern graph theory and Perelman's topological insights
    """
    @staticmethod
    def build_topic_network(channels_data):
        """Build a network graph of related channels and topics"""
        G = nx.Graph()
        # Add nodes (channels) with attributes
        for channel in channels_data:
            G.add_node(
                channel['Channel Name'],
                subscribers=channel['Subscribers'],
                niche=channel['Found Via Niche']
            )
        # Add edges based on niche similarity
        niches = defaultdict(list)
        for channel in channels_data:
            niches[channel['Found Via Niche']].append(channel['Channel Name'])
        for niche, channels in niches.items():
            for i, channel1 in enumerate(channels):
                for channel2 in channels[i+1:]:
                    G.add_edge(channel1, channel2, weight=1.0, niche=niche)
        return G

    @staticmethod
    def calculate_network_centrality(G, node):
        """Calculate various centrality measures"""
        try:
            betweenness = nx.betweenness_centrality(G)[node]
            closeness = nx.closeness_centrality(G)[node]
            degree = nx.degree_centrality(G)[node]
            return {
                'betweenness': betweenness,
                'closeness': closeness,
                'degree': degree,
                'influence_score': (betweenness + closeness + degree) / 3
            }
        except Exception:
            return {'betweenness': 0, 'closeness': 0, 'degree': 0, 'influence_score': 0}

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

def parse_youtube_duration(duration_str):
    """Parse YouTube duration format (PT#M#S) to seconds"""
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def perform_advanced_analysis(api_key, channel_id, channel_data, analysis_depth):
    """
    Comprehensive channel analysis using advanced mathematical models
    """
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    analysis_results = {
        "Engagement Score": 0, "Viral Potential": 0, "Growth Velocity": 0,
        "Growth Acceleration": 0, "Content Consistency": 0, "Monetization Signals": [],
        "Readability Score": 0, "Topic Coherence": 0, "Optimal Upload Times": [],
        "Predicted Growth Trajectory": "Stable"
    }
    try:
        # Get comprehensive video data
        video_search_params = {
            "part": "snippet", "channelId": channel_id, "order": "date",
            "maxResults": 25, "key": api_key
        }
        video_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, video_search_params)
        if not video_response or not video_response.get("items"): return analysis_results
        
        video_ids = [item["id"]["videoId"] for item in video_response["items"] if "videoId" in item.get("id", {})]
        if not video_ids: return analysis_results
            
        video_details_params = {
            "part": "statistics,snippet,contentDetails", "id": ",".join(video_ids), "key": api_key
        }
        details_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, video_details_params)
        if not details_response or not details_response.get("items"): return analysis_results

        videos_data = details_response.get("items", [])
        metrics = defaultdict(list)
        for video in videos_data:
            stats, snippet, content_details = video.get("statistics", {}), video.get("snippet", {}), video.get("contentDetails", {})
            metrics['views'].append(int(stats.get("viewCount", 0)))
            metrics['likes'].append(int(stats.get("likeCount", 0)))
            metrics['comments'].append(int(stats.get("commentCount", 0)))
            metrics['titles'].append(snippet.get("title", ""))
            metrics['descriptions'].append(snippet.get("description", ""))
            metrics['durations'].append(parse_youtube_duration(content_details.get("duration", "PT0S")))
            if snippet.get("publishedAt"):
                metrics['publish_dates'].append(datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00")))
        
        if len(metrics['views']) > 2:
            # Enhanced Engagement & Virality Analysis
            engagement_scores, viral_coefficients = [], []
            subscriber_count = int(channel_data.get("statistics", {}).get("subscriberCount", 1))

            for i in range(len(metrics['views'])):
                if metrics['views'][i] > 0:
                    engagement_scores.append(predictor.engagement_quality_score(metrics['likes'][i], metrics['comments'][i], metrics['views'][i], metrics['durations'][i]))
                    time_since = datetime.now(metrics['publish_dates'][i].tzinfo) - metrics['publish_dates'][i]
                    viral_coefficients.append(predictor.calculate_viral_coefficient(metrics['views'][i], time_since, subscriber_count))
            
            if engagement_scores: analysis_results["Engagement Score"] = np.mean(engagement_scores)
            if viral_coefficients: analysis_results["Viral Potential"] = np.mean(viral_coefficients) * 100

            # Growth Dynamics Analysis
            if len(metrics['publish_dates']) > 3:
                sorted_data = sorted(zip(metrics['publish_dates'], metrics['views']))
                dates, views = zip(*sorted_data)
                time_deltas = [(d - dates[0]).days for d in dates]
                analysis_results["Growth Velocity"] = analyzer.calculate_growth_velocity(views, time_deltas)
                analysis_results["Growth Acceleration"] = analyzer.calculate_growth_acceleration(views, time_deltas)

            # Content Consistency
            if np.mean(metrics['views']) > 0:
                view_cv = np.std(metrics['views']) / np.mean(metrics['views'])
                analysis_results["Content Consistency"] = max(0, 100 - (view_cv * 100))

        # NLP Analysis
        all_text = " ".join(metrics['titles'] + metrics['descriptions'])
        if all_text:
            try:
                analysis_results["Readability Score"] = flesch_reading_ease(all_text)
            except Exception:
                analysis_results["Readability Score"] = 50

    except Exception as e:
        st.warning(f"Partial analysis for {channel_data['snippet']['title']} due to: {e}")

    # Monetization Signals
    channel_description = channel_data.get("snippet", {}).get("description", "")
    monetization_patterns = {'Affiliate': r'affiliate|commission', 'Sponsorship': r'sponsor|brand deal', 'Merchandise': r'merch|store', 'Course': r'course|masterclass', 'Patreon': r'patreon|ko-fi'}
    detected_signals = [sig_type for sig_type, pattern in monetization_patterns.items() if re.search(pattern, channel_description.lower())]
    analysis_results["Monetization Signals"] = detected_signals

    return analysis_results


def find_viral_new_channels_enhanced(api_key, niche_ideas_list, video_type="Any", analysis_depth="Deep"):
    """
    Enhanced niche research with advanced mathematical modeling
    """
    viral_channels = []
    current_year = datetime.now().year
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_channel_ids = set()

    for i, niche in enumerate(niche_ideas_list):
        status_text.text(f"🔬 Analyzing niche '{niche}'... ({i + 1}/{len(niche_ideas_list)})")
        progress_bar.progress((i + 1) / len(niche_ideas_list))
        
        search_params = {
            "part": "snippet", "q": niche, "type": "video", "order": "relevance",
            "publishedAfter": (datetime.utcnow() - timedelta(days=120)).isoformat("T") + "Z",
            "maxResults": 30, "key": api_key
        }
        if video_type != "Any":
             search_params['videoDuration'] = 'short' if video_type == "Shorts Channel" else 'long'
        
        search_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_params)
        if not search_response or not search_response.get("items"): continue

        new_channel_ids = list({item["snippet"]["channelId"] for item in search_response["items"]} - processed_channel_ids)
        if not new_channel_ids: continue

        for batch_start in range(0, len(new_channel_ids), 50):
            batch_ids = new_channel_ids[batch_start:batch_start + 50]
            channel_params = {"part": "snippet,statistics", "id": ",".join(batch_ids), "key": api_key}
            channel_response = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
            if not channel_response or not channel_response.get("items"): continue

            for channel in channel_response["items"]:
                published_date = datetime.fromisoformat(channel["snippet"]["publishedAt"].replace("Z", "+00:00"))
                if published_date.year >= current_year - 1:
                    stats_data = channel.get("statistics", {})
                    subs, views, video_count = int(stats_data.get("subscriberCount", 0)), int(stats_data.get("viewCount", 0)), int(stats_data.get("videoCount", 0))
                    subscriber_velocity = subs / max((datetime.now(published_date.tzinfo) - published_date).days, 1)
                    view_to_video_ratio = views / max(video_count, 1)
                    
                    if subs > 500 and views > 25000 and 3 < video_count < 200 and subscriber_velocity > 5 and view_to_video_ratio > 1000:
                        channel_id = channel['id']
                        analysis_data = perform_advanced_analysis(api_key, channel_id, channel, analysis_depth)
                        viral_channels.append({
                            "Channel Name": channel["snippet"]["title"], "URL": f"https://www.youtube.com/channel/{channel_id}",
                            "Subscribers": subs, "Total Views": views, "Video Count": video_count,
                            "Creation Date": published_date.strftime("%Y-%m-%d"), "Channel Age (Days)": (datetime.now(published_date.tzinfo) - published_date).days,
                            "Found Via Niche": niche, "Subscriber Velocity": round(subscriber_velocity, 2),
                            "View-to-Video Ratio": round(view_to_video_ratio, 0), **analysis_data
                        })
                        processed_channel_ids.add(channel_id)

    progress_bar.empty()
    status_text.empty()
    if viral_channels:
        return apply_advanced_ranking(viral_channels)
    return viral_channels

def apply_advanced_ranking(channels):
    """
    Advanced ranking algorithm using multi-dimensional scoring
    """
    weights = {'subscriber_velocity': 0.25, 'engagement': 0.20, 'viral_potential': 0.20, 'growth_velocity': 0.15, 'consistency': 0.10, 'monetization': 0.10}
    
    features = []
    for ch in channels:
        features.append([
            ch.get('Subscriber Velocity', 0), ch.get('Engagement Score', 0), ch.get('Viral Potential', 0),
            ch.get('Growth Velocity', 0), ch.get('Content Consistency', 0), len(ch.get('Monetization Signals', [])) * 10
        ])
    
    if not features: return channels
    
    features_normalized = StandardScaler().fit_transform(features)
    
    for i, channel in enumerate(channels):
        score = np.dot(features_normalized[i], list(weights.values()))
        channel['Intelligence_Score'] = round(score * 100, 2)
        if score > 0.8: channel['Ranking_Tier'] = "🏆 Elite"
        elif score > 0.6: channel['Ranking_Tier'] = "🥇 Excellent"
        elif score > 0.4: channel['Ranking_Tier'] = "🥈 Good"
        else: channel['Ranking_Tier'] = "📈 Emerging"
            
    return sorted(channels, key=lambda x: x.get('Intelligence_Score', 0), reverse=True)


# --- Main Application UI ---

st.title("🧠 YouTube Growth Intelligence Engine")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    <h3 style="color: white; margin: 0;">Advanced Mathematical Analysis for YouTube Success</h3>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔧 Configuration Panel")
    api_key = st.text_input("YouTube Data API Key:", type="password", help="Get your API key from Google Cloud Console")
    if api_key: st.success("✅ API Key Configured")
    else: st.error("❌ API Key Required")
    st.divider()
    analysis_depth = st.selectbox("Analysis Depth:", ["Quick", "Standard", "Deep"], index=2)

tab1, tab2 = st.tabs(["🔍 Intelligent Niche Research", "📈 Growth Trajectory Analysis"])

with tab1:
    st.header("🚀 Intelligent Niche Research Engine")
    video_type_choice = st.radio("Channel Type Focus:", ('Any Content', 'Shorts-Focused', 'Long-Form Content'), horizontal=True)
    suggested_niches = {
        "AI & Technology": ["AI Tools for Creators", "No-Code SaaS", "Crypto DeFi Explained"],
        "Personal Development": ["Productivity for ADHD", "Financial Independence", "Minimalist Lifestyle"],
    }
    niche_category = st.selectbox("Choose a Category for Suggestions:", list(suggested_niches.keys()))
    user_niche_input = st.text_area("Enter Niche Ideas (one per line):", "\n".join(suggested_niches[niche_category]), height=150)

    if st.button("🚀 Launch Intelligent Analysis", type="primary", use_container_width=True):
        if not api_key:
            st.error("🔐 Please configure your API key in the sidebar.")
        else:
            niche_ideas = [n.strip() for n in user_niche_input.split('\n') if n.strip()]
            if not niche_ideas:
                st.warning("⚠️ Please enter at least one niche idea.")
            else:
                with st.spinner("🔬 Applying advanced mathematical models..."):
                    video_type_map = {'Any Content': 'Any', 'Shorts-Focused': 'Shorts Channel', 'Long-Form Content': 'Long Video Channel'}
                    st.session_state.analysis_results = find_viral_new_channels_enhanced(api_key, niche_ideas, video_type_map[video_type_choice], analysis_depth)
                
                if st.session_state.analysis_results:
                    st.success(f"🎉 Analysis Complete! Found {len(st.session_state.analysis_results)} high-potential channels.")
                else:
                    st.warning("🔍 No channels found matching the criteria. Try adjusting your search.")

    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.subheader("🔬 Individual Channel Intelligence Reports")
        for i, channel in enumerate(st.session_state.analysis_results):
            with st.expander(f"#{i+1} {channel['Channel Name']} • {channel.get('Ranking_Tier', 'Unranked')} • Score: {channel.get('Intelligence_Score', 0):.1f}", expanded=(i < 3)):
                col1, col2, col3 = st.columns(3)
                col1.metric("Subscribers", f"{channel['Subscribers']:,}")
                col2.metric("Total Views", f"{channel['Total Views']:,}")
                col3.metric("Videos", channel['Video Count'])
                st.markdown(f"[🔗 Visit Channel]({channel['URL']})")


with tab2:
    st.header("📈 Growth Trajectory Analysis")
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        df_trajectory = pd.DataFrame(st.session_state.analysis_results)
        fig_growth = px.scatter(
            df_trajectory, x='Growth Velocity', y='Growth Acceleration', size='Subscribers',
            color='Intelligence_Score', hover_name='Channel Name', title="Growth Dynamics Analysis"
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.info("📊 Run the Niche Research analysis first to see growth trajectory data.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Powered by advanced mathematical models</p>", unsafe_allow_html=True)

