#!/usr/bin/env python3
"""
YouTube Analytics Platform
A comprehensive tool combining channel discovery, niche research, growth analysis, and viral video finding
Integrates advanced mathematical models and real-time YouTube Data API analytics
"""

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

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Analytics Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Combined Theme) ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
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
    .result-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-highlight {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 5px;
        display: inline-block;
        font-weight: bold;
    }
    .optional-section {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
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

# --- Mathematical Constants & API Configuration ---
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
EULER_NUMBER = np.e
PI = np.pi
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# --- Advanced Analytics Classes ---
class GrowthAnalyzer:
    @staticmethod
    def exponential_growth_model(t, a, b, c):
        return a * np.exp(b * t) + c

    @staticmethod
    def logistic_growth_model(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))

    @staticmethod
    def power_law_model(x, a, b):
        return a * np.power(x, b)

    @staticmethod
    def calculate_growth_velocity(data_points, time_intervals):
        if len(data_points) < 2:
            return 0
        velocities = np.gradient(data_points, time_intervals)
        return np.mean(velocities)

    @staticmethod
    def calculate_growth_acceleration(data_points, time_intervals):
        if len(data_points) < 3:
            return 0
        velocities = np.gradient(data_points, time_intervals)
        accelerations = np.gradient(velocities, time_intervals)
        return np.mean(accelerations)

class ViralityPredictor:
    @staticmethod
    def calculate_viral_coefficient(views, time_since_publish, subscriber_count):
        if subscriber_count == 0:
            return 0
        decay_constant = 0.1
        time_factor = np.exp(-decay_constant * time_since_publish.days)
        viral_ratio = views / max(subscriber_count, 1)
        return viral_ratio * time_factor

    @staticmethod
    def engagement_quality_score(likes, comments, views, video_duration):
        if views == 0:
            return 0
        duration_factor = max(1, video_duration / 300)
        like_rate = (likes / views) * 100
        comment_rate = (comments / views) * 100
        engagement_score = (like_rate + 3 * comment_rate) / duration_factor
        return 10 / (1 + np.exp(-engagement_score + 2))

class NetworkAnalyzer:
    @staticmethod
    def build_topic_network(channels_data):
        G = nx.Graph()
        for channel in channels_data:
            G.add_node(
                channel['Channel Name'],
                subscribers=channel['Subscribers'],
                niche=channel['Found Via Niche']
            )
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

# --- Utility Functions ---
@st.cache_data(ttl=3600)
def fetch_youtube_data(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def parse_youtube_duration(duration_str):
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def format_number(num):
    if num >= 1000000000:
        return f"{num/1000000000:.1f}B"
    elif num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)

# --- Analysis Functions ---
def perform_advanced_analysis(api_key, channel_id, channel_data, analysis_depth):
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    analysis_results = {
        "Engagement Score": 0, "Viral Potential": 0, "Growth Velocity": 0,
        "Growth Acceleration": 0, "Content Consistency": 0, "Monetization Signals": [],
        "Readability Score": 0, "Topic Coherence": 0, "Optimal Upload Times": [],
        "Predicted Growth Trajectory": "Stable"
    }
    try:
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
            engagement_scores, viral_coefficients = [], []
            subscriber_count = int(channel_data.get("statistics", {}).get("subscriberCount", 1))

            for i in range(len(metrics['views'])):
                if metrics['views'][i] > 0:
                    engagement_scores.append(predictor.engagement_quality_score(metrics['likes'][i], metrics['comments'][i], metrics['views'][i], metrics['durations'][i]))
                    time_since = datetime.now(metrics['publish_dates'][i].tzinfo) - metrics['publish_dates'][i]
                    viral_coefficients.append(predictor.calculate_viral_coefficient(metrics['views'][i], time_since, subscriber_count))
            
            if engagement_scores: analysis_results["Engagement Score"] = np.mean(engagement_scores)
            if viral_coefficients: analysis_results["Viral Potential"] = np.mean(viral_coefficients) * 100

            if len(metrics['publish_dates']) > 3:
                sorted_data = sorted(zip(metrics['publish_dates'], metrics['views']))
                dates, views = zip(*sorted_data)
                time_deltas = [(d - dates[0]).days for d in dates]
                analysis_results["Growth Velocity"] = analyzer.calculate_growth_velocity(views, time_deltas)
                analysis_results["Growth Acceleration"] = analyzer.calculate_growth_acceleration(views, time_deltas)

            if np.mean(metrics['views']) > 0:
                view_cv = np.std(metrics['views']) / np.mean(metrics['views'])
                analysis_results["Content Consistency"] = max(0, 100 - (view_cv * 100))

        all_text = " ".join(metrics['titles'] + metrics['descriptions'])
        if all_text:
            try:
                analysis_results["Readability Score"] = flesch_reading_ease(all_text)
            except Exception:
                analysis_results["Readability Score"] = 50

    except Exception as e:
        st.warning(f"Partial analysis for {channel_data['snippet']['title']} due to: {e}")

    channel_description = channel_data.get("snippet", {}).get("description", "")
    monetization_patterns = {'Affiliate': r'affiliate|commission', 'Sponsorship': r'sponsor|brand deal', 'Merchandise': r'merch|store', 'Course': r'course|masterclass', 'Patreon': r'patreon|ko-fi'}
    detected_signals = [sig_type for sig_type, pattern in monetization_patterns.items() if re.search(pattern, channel_description.lower())]
    analysis_results["Monetization Signals"] = detected_signals

    return analysis_results

def find_viral_new_channels_enhanced(api_key, niche_ideas_list, video_type="Any", analysis_depth="Deep"):
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

def find_channels_with_criteria(api_key, search_params):
    keywords = search_params.get('keywords', '')
    channel_type = search_params.get('channel_type', 'Any')
    creation_year = search_params.get('creation_year', None)
    max_channels = search_params.get('max_channels', 50)
    description_keyword = search_params.get('description_keyword', '')
    min_subscribers = search_params.get('min_subscribers', 0)
    max_subscribers = search_params.get('max_subscribers', float('inf'))
    min_videos = search_params.get('min_videos', 0)
    max_videos = search_params.get('max_videos', float('inf'))
    
    found_channels = []
    processed_channel_ids = set()
    progress_bar = st.progress(0)
    status_text = st.empty()
    search_terms = [term.strip() for term in keywords.split(',') if term.strip()]
    
    for i, term in enumerate(search_terms):
        status_text.text(f"🔍 Searching for: '{term}' ({i + 1}/{len(search_terms)})")
        progress_bar.progress((i + 1) / len(search_terms))
        
        search_query_params = {
            "part": "snippet",
            "q": term,
            "type": "video",
            "order": "relevance",
            "maxResults": 50,
            "key": api_key
        }
        
        if channel_type == "Short":
            search_query_params['videoDuration'] = 'short'
        elif channel_type == "Long":
            search_query_params['videoDuration'] = 'long'
        
        if creation_year and creation_year > 1900:
            start_date = f"{creation_year}-01-01T00:00:00Z"
            end_date = f"{creation_year + 1}-01-01T00:00:00Z"
            search_query_params['publishedAfter'] = start_date
            search_query_params['publishedBefore'] = end_date
        
        search_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_query_params)
        if not search_response or not search_response.get("items"):
            continue
        
        channel_ids = list(set([item["snippet"]["channelId"] for item in search_response["items"] if item["snippet"]["channelId"] not in processed_channel_ids]))
        
        if not channel_ids:
            continue
        
        for batch_start in range(0, len(channel_ids), 50):
            batch_ids = channel_ids[batch_start:batch_start + 50]
            channel_params = {
                "part": "snippet,statistics",
                "id": ",".join(batch_ids),
                "key": api_key
            }
            
            channel_response = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
            if not channel_response or not channel_response.get("items"):
                continue
            
            for channel in channel_response["items"]:
                try:
                    snippet = channel.get("snippet", {})
                    stats = channel.get("statistics", {})
                    channel_name = snippet.get("title", "Unknown")
                    channel_description = snippet.get("description", "")
                    published_date = datetime.fromisoformat(snippet.get("publishedAt", "").replace("Z", "+00:00"))
                    
                    subscribers = int(stats.get("subscriberCount", 0))
                    total_views = int(stats.get("viewCount", 0))
                    video_count = int(stats.get("videoCount", 0))
                    
                    if creation_year and published_date.year != creation_year:
                        continue
                    
                    if not (min_subscribers <= subscribers <= max_subscribers):
                        continue
                    
                    if not (min_videos <= video_count <= max_videos):
                        continue
                    
                    if description_keyword and description_keyword.lower() not in channel_description.lower():
                        continue
                    
                    channel_age_days = (datetime.now(published_date.tzinfo) - published_date).days
                    avg_views_per_video = total_views / max(video_count, 1)
                    subscriber_velocity = subscribers / max(channel_age_days, 1)
                    
                    channel_data = {
                        "Channel Name": channel_name,
                        "URL": f"https://www.youtube.com/channel/{channel['id']}",
                        "Subscribers": subscribers,
                        "Total Views": total_views,
                        "Video Count": video_count,
                        "Creation Date": published_date.strftime("%Y-%m-%d"),
                        "Channel Age (Days)": channel_age_days,
                        "Found Via Keyword": term,
                        "Subscriber Velocity": round(subscriber_velocity, 4),
                        "Avg Views per Video": round(avg_views_per_video, 0),
                        "Description": channel_description[:200] + "..." if len(channel_description) > 200 else channel_description
                    }
                    
                    found_channels.append(channel_data)
                    processed_channel_ids.add(channel['id'])
                    
                    if len(found_channels) >= max_channels:
                        break
                        
                except (ValueError, KeyError):
                    continue
            
            if len(found_channels) >= max_channels:
                break
        
        if len(found_channels) >= max_channels:
            break
    
    progress_bar.empty()
    status_text.empty()
    
    return found_channels

def perform_advanced_channel_analysis(api_key, channels_data):
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    enhanced_channels = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, channel_data in enumerate(channels_data):
        status_text.text(f"🧠 Analyzing: {channel_data['Channel Name']} ({i + 1}/{len(channels_data)})")
        progress_bar.progress((i + 1) / len(channels_data))
        
        try:
            channel_id = channel_data['URL'].split('/')[-1]
            video_search_params = {
                "part": "snippet",
                "channelId": channel_id,
                "order": "date",
                "maxResults": 10,
                "key": api_key
            }
            
            video_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, video_search_params)
            if video_response and video_response.get("items"):
                video_ids = [item["id"]["videoId"] for item in video_response["items"] if "videoId" in item.get("id", {})]
                
                if video_ids:
                    video_details_params = {
                        "part": "statistics,snippet,contentDetails",
                        "id": ",".join(video_ids[:5]),
                        "key": api_key
                    }
                    
                    details_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, video_details_params)
                    if details_response and details_response.get("items"):
                        videos_data = details_response.get("items", [])
                        total_views = sum([int(v.get("statistics", {}).get("viewCount", 0)) for v in videos_data])
                        total_likes = sum([int(v.get("statistics", {}).get("likeCount", 0)) for v in videos_data])
                        total_comments = sum([int(v.get("statistics", {}).get("commentCount", 0)) for v in videos_data])
                        
                        if total_views > 0:
                            engagement_rate = ((total_likes + total_comments) / total_views) * 100
                        else:
                            engagement_rate = 0
                        
                        channel_data["Recent Engagement Rate"] = round(engagement_rate, 3)
                        channel_data["Recent Avg Views"] = round(total_views / len(videos_data), 0)
                        channel_data["Analysis Status"] = "✅ Complete"
                    else:
                        channel_data["Analysis Status"] = "⚠️ Limited Data"
                else:
                    channel_data["Analysis Status"] = "❌ No Videos"
            else:
                channel_data["Analysis Status"] = "❌ Access Denied"
                
        except Exception as e:
            channel_data["Analysis Status"] = f"❌ Error: {str(e)[:30]}"
        
        enhanced_channels.append(channel_data)
    
    progress_bar.empty()
    status_text.empty()
    
    return enhanced_channels

# --- Main Application UI ---
st.markdown("""
<div class="main-header">
    <h1>🧠 YouTube Analytics Platform</h1>
    <h3>Advanced Channel Discovery & Growth Intelligence</h3>
    <p>Discover trending channels, analyze growth patterns, and find viral videos with AI-powered analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration Panel")
    api_key = st.text_input("YouTube Data API Key:", type="password", help="Get your API key from Google Cloud Console")
    if api_key:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Required")
    st.divider()
    analysis_mode = st.selectbox("Analysis Mode:", ["Basic Search", "Advanced Analytics"], index=1)
    analysis_depth = st.selectbox("Analysis Depth:", ["Quick", "Standard", "Deep"], index=2)
    st.divider()
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    st.subheader("📊 API Usage")
    if 'api_calls_made' not in st.session_state:
        st.session_state.api_calls_made = 0
    st.metric("API Calls Today", st.session_state.api_calls_made)
    st.progress(min(st.session_state.api_calls_made / 100, 1.0))
    if st.session_state.api_calls_made > 80:
        st.warning("⚠️ Approaching API limit!")

# Tabs for Tools
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Channel Finder", "🔍 Intelligent Niche Research", "📈 Growth Trajectory Analysis", "🔥 Viral Video Finder"])

with tab1:
    st.header("🎯 Channel Finder")
    st.markdown("<div class='optional-section'><h4>📋 Search Criteria</h4></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        keywords = st.text_input("🔤 Keywords (e.g. 'AI, gaming, cooking'):", placeholder="Separate with commas")
        channel_type = st.selectbox("📺 Channel Type:", ["Any", "Long", "Short"])
    with col2:
        creation_year = st.number_input("📅 Creation Year:", min_value=2005, max_value=2025, value=2023)
        max_channels = st.number_input("🔢 Max Channels:", min_value=1, max_value=500, value=50)
    
    st.markdown("<div class='optional-section'><h4>🎛️ Optional Filters</h4></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        description_keyword = st.text_input("📝 Description Keyword:", placeholder="Optional")
        min_subscribers = st.number_input("👥 Min Subscribers:", min_value=0, value=0)
        min_videos = st.number_input("🎬 Min Videos:", min_value=0, value=0)
    with col4:
        max_subscribers = st.number_input("👥 Max Subscribers:", min_value=1, value=1000000)
        max_videos = st.number_input("🎬 Max Videos:", min_value=1, value=10000)
    
    if st.button("🚀 Start Channel Discovery", type="primary", use_container_width=True):
        if not api_key:
            st.error("🔐 Please configure your YouTube API key in the sidebar!")
        elif not keywords:
            st.error("🔤 Please enter at least one keyword!")
        else:
            search_params = {
                'keywords': keywords, 'channel_type': channel_type, 'creation_year': creation_year,
                'max_channels': max_channels, 'description_keyword': description_keyword,
                'min_subscribers': min_subscribers, 'max_subscribers': max_subscribers,
                'min_videos': min_videos, 'max_videos': max_videos
            }
            with st.spinner("🔍 Searching for channels..."):
                channels = find_channels_with_criteria(api_key, search_params)
                if analysis_mode == "Advanced Analytics":
                    channels = perform_advanced_channel_analysis(api_key, channels)
                st.session_state.found_channels = channels
                st.session_state.api_calls_made += 1
            if channels:
                st.success(f"🎉 Found {len(channels)} channels!")
                st.subheader("📋 Preview")
                st.dataframe(pd.DataFrame(channels)[['Channel Name', 'Subscribers', 'Video Count', 'Creation Date']].head(10), use_container_width=True)
            else:
                st.warning("😔 No channels found. Try adjusting filters.")

with tab2:
    st.header("🚀 Intelligent Niche Research Engine")
    video_type_choice = st.radio("Channel Type Focus:", ('Any Content', 'Shorts-Focused', 'Long-Form Content'), horizontal=True)
    suggested_niches = {
        "AI & Technology": ["AI Tools for Creators", "No-Code SaaS", "Crypto DeFi Explained"],
        "Personal Development": ["Productivity for ADHD", "Financial Independence", "Minimalist Lifestyle"],
    }
    niche_category = st.selectbox("Choose a Category:", list(suggested_niches.keys()))
    user_niche_input = st.text_area("Niche Ideas (one per line):", "\n".join(suggested_niches[niche_category]), height=150)

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
                    st.session_state.api_calls_made += 1
                if st.session_state.analysis_results:
                    st.success(f"🎉 Found {len(st.session_state.analysis_results)} high-potential channels!")
                else:
                    st.warning("🔍 No channels found. Try adjusting your search.")

    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.subheader("🔬 Channel Intelligence Reports")
        for i, channel in enumerate(st.session_state.analysis_results):
            with st.expander(f"#{i+1} {channel['Channel Name']} • {channel.get('Ranking_Tier', 'Unranked')} • Score: {channel.get('Intelligence_Score', 0):.1f}", expanded=(i < 3)):
                col1, col2, col3 = st.columns(3)
                col1.metric("Subscribers", f"{channel['Subscribers']:,}")
                col2.metric("Total Views", f"{channel['Total Views']:,}")
                col3.metric("Videos", channel['Video Count'])
                col4, col5, col6 = st.columns(3)
                col4.metric("Engagement Score", f"{channel['Engagement Score']:.2f}")
                col5.metric("Viral Potential", f"{channel['Viral Potential']:.2f}%")
                col6.metric("Growth Velocity", f"{channel['Growth Velocity']:.2f}")
                st.markdown(f"[🔗 Visit Channel]({channel['URL']})")

with tab3:
    st.header("📈 Growth Trajectory Analysis")
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        df_trajectory = pd.DataFrame(st.session_state.analysis_results)
        fig_growth = px.scatter(
            df_trajectory, x='Growth Velocity', y='Growth Acceleration', size='Subscribers',
            color='Intelligence_Score', hover_name='Channel Name', title="Growth Dynamics Analysis"
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    if 'found_channels' in st.session_state and st.session_state.found_channels:
        df = pd.DataFrame(st.session_state.found_channels)
        fig_scatter = px.scatter(
            df, x='Subscribers', y='Total Views', size='Video Count', hover_name='Channel Name',
            title="Views vs Subscribers Analysis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        fig_age = px.bar(
            df.head(20), x='Channel Name', y='Channel Age (Days)', title="Channel Age Analysis (Top 20 Channels)"
        )
        fig_age.update_xaxes(tickangle=45)
        st.plotly_chart(fig_age, use_container_width=True)
    if not ('analysis_results' in st.session_state or 'found_channels' in st.session_state):
        st.info("📊 Run a search in Channel Finder or Intelligent Niche Research to see growth trajectory data.")

with tab4:
    st.header("🔥 Viral Video Finder")
    days = st.number_input("Days to Search (1-30):", min_value=1, max_value=30, value=5)
    keywords = st.text_area("Keywords (one per line):", "Affair Relationship Stories\nReddit Update\nReddit Relationship Advice")

    if st.button("🔍 Find Viral Videos"):
        if not api_key:
            st.error("🔐 Please configure your API key in the sidebar.")
        else:
            with st.spinner("🔍 Searching for viral videos..."):
                start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
                all_results = []
                for keyword in keywords.split("\n"):
                    st.write(f"Searching for keyword: {keyword}")
                    search_params = {
                        "part": "snippet",
                        "q": keyword,
                        "type": "video",
                        "order": "viewCount",
                        "publishedAfter": start_date,
                        "maxResults": 5,
                        "key": api_key,
                    }
                    response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_params)
                    if not response or not response.get("items"):
                        st.warning(f"No videos found for keyword: {keyword}")
                        continue
                    videos = response["items"]
                    video_ids = [video["id"]["videoId"] for video in videos if "id" in video and "videoId" in video["id"]]
                    channel_ids = [video["snippet"]["channelId"] for video in videos if "snippet" in video and "channelId" in video["snippet"]]
                    if not video_ids or not channel_ids:
                        st.warning(f"Skipping keyword: {keyword} due to missing data.")
                        continue
                    stats_params = {"part": "statistics", "id": ",".join(video_ids), "key": api_key}
                    stats_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, stats_params)
                    if not stats_response or not stats_response.get("items"):
                        st.warning(f"Failed to fetch video statistics for keyword: {keyword}")
                        continue
                    channel_params = {"part": "statistics", "id": ",".join(channel_ids), "key": api_key}
                    channel_response = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
                    if not channel_response or not channel_response.get("items"):
                        st.warning(f"Failed to fetch channel statistics for keyword: {keyword}")
                        continue
                    stats = stats_response["items"]
                    channels = channel_response["items"]
                    for video, stat, channel in zip(videos, stats, channels):
                        title = video["snippet"].get("title", "N/A")
                        description = video["snippet"].get("description", "")[:200]
                        video_url = f"https://www.youtube.com/watch?v={video['id']['videoId']}"
                        views = int(stat["statistics"].get("viewCount", 0))
                        subs = int(channel["statistics"].get("subscriberCount", 0))
                        if subs < 3000:
                            all_results.append({
                                "Title": title,
                                "Description": description,
                                "URL": video_url,
                                "Views": views,
                                "Subscribers": subs
                            })
                    st.session_state.api_calls_made += 1
                if all_results:
                    st.success(f"Found {len(all_results)} results!")
                    for result in all_results:
                        st.markdown(
                            f"**Title:** {result['Title']}  \n"
                            f"**Description:** {result['Description']}  \n"
                            f"**URL:** [Watch Video]({result['URL']})  \n"
                            f"**Views:** {result['Views']}  \n"
                            f"**Subscribers:** {result['Subscribers']}"
                        )
                        st.write("---")
                else:
                    st.warning("No results found for channels with fewer than 3,000 subscribers.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>📝 YouTube Analytics Platform</h4>
    <p>Discover trending channels, analyze growth patterns, and find viral videos with advanced analytics powered by YouTube Data API v3 and mathematical models.</p>
    <p><strong>🚀 Features:</strong> Channel discovery • Niche research • Growth tracking • Viral video detection • Real-time visualizations</p>
    <p>YouTube Analytics Platform v1.0 | Last Updated: September 2025</p>
</div>
""", unsafe_allow_html=True)
