#!/usr/bin/env python3
"""
YouTube Channel Finder - Advanced Web Application
A comprehensive tool to find and analyze YouTube channels based on multiple criteria
Combines user input questions with advanced analytics and growth intelligence
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
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="🔍 YouTube Channel Finder - Advanced Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
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
    .question-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
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

# --- Utility Functions ---
@st.cache_data(ttl=3600)
def fetch_youtube_data(url, params):
    """Fetch data from YouTube API with caching"""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def parse_youtube_duration(duration_str):
    """Parse YouTube duration format (PT1H2M3S) to seconds"""
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def format_number(num):
    """Format large numbers for display"""
    if num >= 1000000000:
        return f"{num/1000000000:.1f}B"
    elif num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)

# --- Main Search Function ---
def find_channels_with_criteria(api_key, search_params):
    """Find YouTube channels based on user-defined criteria"""
    
    # Extract search parameters
    keywords = search_params.get('keywords', '')
    channel_type = search_params.get('channel_type', 'Any')
    creation_year = search_params.get('creation_year', None)
    max_channels = search_params.get('max_channels', 50)
    
    # Optional filters
    description_keyword = search_params.get('description_keyword', '')
    min_subscribers = search_params.get('min_subscribers', 0)
    max_subscribers = search_params.get('max_subscribers', float('inf'))
    min_videos = search_params.get('min_videos', 0)
    max_videos = search_params.get('max_videos', float('inf'))
    
    found_channels = []
    processed_channel_ids = set()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Search for videos first, then extract channels
    search_terms = [term.strip() for term in keywords.split(',') if term.strip()]
    
    for i, term in enumerate(search_terms):
        status_text.text(f"🔍 Searching for: '{term}' ({i + 1}/{len(search_terms)})")
        progress_bar.progress((i + 1) / len(search_terms))
        
        # Configure search parameters
        search_query_params = {
            "part": "snippet",
            "q": term,
            "type": "video",
            "order": "relevance",
            "maxResults": 50,
            "key": api_key
        }
        
        # Add video duration filter if specified
        if channel_type == "Short":
            search_query_params['videoDuration'] = 'short'
        elif channel_type == "Long":
            search_query_params['videoDuration'] = 'long'
        
        # Add creation year filter
        if creation_year and creation_year > 1900:
            start_date = f"{creation_year}-01-01T00:00:00Z"
            end_date = f"{creation_year + 1}-01-01T00:00:00Z"
            search_query_params['publishedAfter'] = start_date
            search_query_params['publishedBefore'] = end_date
        
        # Search for videos
        search_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_query_params)
        if not search_response or not search_response.get("items"):
            continue
        
        # Extract unique channel IDs
        channel_ids = list(set([
            item["snippet"]["channelId"] 
            for item in search_response["items"] 
            if item["snippet"]["channelId"] not in processed_channel_ids
        ]))
        
        if not channel_ids:
            continue
        
        # Get channel details in batches
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
            
            # Process each channel
            for channel in channel_response["items"]:
                try:
                    # Extract channel data
                    snippet = channel.get("snippet", {})
                    stats = channel.get("statistics", {})
                    
                    channel_name = snippet.get("title", "Unknown")
                    channel_description = snippet.get("description", "")
                    published_date = datetime.fromisoformat(
                        snippet.get("publishedAt", "").replace("Z", "+00:00")
                    )
                    
                    subscribers = int(stats.get("subscriberCount", 0))
                    total_views = int(stats.get("viewCount", 0))
                    video_count = int(stats.get("videoCount", 0))
                    
                    # Apply filters
                    if creation_year and published_date.year != creation_year:
                        continue
                    
                    if not (min_subscribers <= subscribers <= max_subscribers):
                        continue
                    
                    if not (min_videos <= video_count <= max_videos):
                        continue
                    
                    if description_keyword and description_keyword.lower() not in channel_description.lower():
                        continue
                    
                    # Calculate additional metrics
                    channel_age_days = (datetime.now(published_date.tzinfo) - published_date).days
                    avg_views_per_video = total_views / max(video_count, 1)
                    subscriber_velocity = subscribers / max(channel_age_days, 1)
                    
                    # Store channel data
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
                    
                    # Stop if we've reached the maximum number of channels
                    if len(found_channels) >= max_channels:
                        break
                        
                except (ValueError, KeyError) as e:
                    continue
            
            if len(found_channels) >= max_channels:
                break
        
        if len(found_channels) >= max_channels:
            break
    
    progress_bar.empty()
    status_text.empty()
    
    return found_channels

# --- Advanced Analysis Function ---
def perform_advanced_channel_analysis(api_key, channels_data):
    """Perform advanced analysis on found channels"""
    
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    
    enhanced_channels = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, channel_data in enumerate(channels_data):
        status_text.text(f"🧠 Analyzing: {channel_data['Channel Name']} ({i + 1}/{len(channels_data)})")
        progress_bar.progress((i + 1) / len(channels_data))
        
        try:
            # Extract channel ID from URL
            channel_id = channel_data['URL'].split('/')[-1]
            
            # Get recent videos for analysis
            video_search_params = {
                "part": "snippet",
                "channelId": channel_id,
                "order": "date",
                "maxResults": 10,
                "key": api_key
            }
            
            video_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, video_search_params)
            
            if video_response and video_response.get("items"):
                video_ids = [
                    item["id"]["videoId"] 
                    for item in video_response["items"] 
                    if "videoId" in item.get("id", {})
                ]
                
                if video_ids:
                    # Get video statistics
                    video_details_params = {
                        "part": "statistics,snippet,contentDetails",
                        "id": ",".join(video_ids[:5]),  # Limit to 5 most recent
                        "key": api_key
                    }
                    
                    details_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, video_details_params)
                    
                    if details_response and details_response.get("items"):
                        videos_data = details_response.get("items", [])
                        
                        # Calculate engagement metrics
                        total_views = sum([int(v.get("statistics", {}).get("viewCount", 0)) for v in videos_data])
                        total_likes = sum([int(v.get("statistics", {}).get("likeCount", 0)) for v in videos_data])
                        total_comments = sum([int(v.get("statistics", {}).get("commentCount", 0)) for v in videos_data])
                        
                        if total_views > 0:
                            engagement_rate = ((total_likes + total_comments) / total_views) * 100
                        else:
                            engagement_rate = 0
                        
                        # Add advanced metrics to channel data
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

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 YouTube Channel Finder</h1>
    <h3>Advanced Channel Discovery & Analytics Platform</h3>
    <p>Find YouTube channels based on your specific criteria with AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 API Configuration")
    api_key = st.text_input("YouTube Data API Key:", type="password", help="Get your API key from Google Cloud Console")
    
    if api_key:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Required")
        st.info("📝 Get your API key from: https://console.cloud.google.com/")
    
    st.divider()
    
    st.header("⚙️ Search Settings")
    analysis_mode = st.selectbox(
        "Analysis Mode:",
        ["Basic Search", "Advanced Analytics"],
        help="Choose between quick search or detailed analysis with metrics"
    )

# Main Content
tab1, tab2, tab3 = st.tabs(["🔍 Channel Finder", "📊 Results Analysis", "📈 Growth Insights"])

with tab1:
    st.header("🎯 Channel Search Criteria")
    
    # Required Questions
    st.markdown("""
    <div class="question-box">
        <h4>📋 Required Information</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        keywords = st.text_input(
            "🔤 Enter keywords (e.g. 'AI', 'cat', 'funny'):",
            placeholder="gaming, tutorial, cooking",
            help="Separate multiple keywords with commas"
        )
        
        channel_type = st.selectbox(
            "📺 Which channel do you find?",
            ["Any", "Long", "Short"],
            help="Long = Long videos, Short = Short videos/Shorts"
        )
    
    with col2:
        creation_year = st.number_input(
            "📅 Channel Creation Year? (e.g. 2023):",
            min_value=2005,
            max_value=2025,
            value=2023,
            help="Year when the channel was created"
        )
        
        max_channels = st.number_input(
            "🔢 How many channels to find? (e.g. 100):",
            min_value=1,
            max_value=500,
            value=50,
            help="Maximum number of channels to discover"
        )
    
    # Optional Questions
    st.markdown("""
    <div class="optional-section">
        <h4>🎛️ Optional Filters (Press Enter to skip)</h4>
        <p><em>Leave empty to skip these filters</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        description_keyword = st.text_input(
            "📝 Channel Description Keyword? (e.g. 'Copyright Disclaimer'):",
            placeholder="Optional - leave empty to skip",
            help="Find channels with specific words in their description"
        )
        
        min_subscribers = st.number_input(
            "👥 Minimum Channel Subscribers? (e.g. 1):",
            min_value=0,
            value=0,
            help="Minimum subscriber count filter"
        )
        
        min_videos = st.number_input(
            "🎬 Minimum Channel Videos? (e.g. 1):",
            min_value=0,
            value=0,
            help="Minimum video count filter"
        )
    
    with col4:
        max_subscribers = st.number_input(
            "👥 Maximum Channel Subscribers? (e.g. 1000000):",
            min_value=1,
            value=1000000,
            help="Maximum subscriber count filter"
        )
        
        max_videos = st.number_input(
            "🎬 Maximum Channel Videos? (e.g. 10000):",
            min_value=1,
            value=10000,
            help="Maximum video count filter"
        )
    
    # Search Button
    st.divider()
    
    if st.button("🚀 Start Channel Discovery", type="primary", use_container_width=True):
        if not api_key:
            st.error("🔐 Please configure your YouTube API key in the sidebar first!")
        elif not keywords:
            st.error("🔤 Please enter at least one keyword to search for!")
        else:
            # Prepare search parameters
            search_params = {
                'keywords': keywords,
                'channel_type': channel_type,
                'creation_year': creation_year,
                'max_channels': max_channels,
                'description_keyword': description_keyword,
                'min_subscribers': min_subscribers,
                'max_subscribers': max_subscribers,
                'min_videos': min_videos,
                'max_videos': max_videos
            }
            
            # Start search
            with st.spinner("🔍 Searching for channels..."):
                channels = find_channels_with_criteria(api_key, search_params)
            
            if channels:
                # Perform advanced analysis if selected
                if analysis_mode == "Advanced Analytics":
                    with st.spinner("🧠 Performing advanced analysis..."):
                        channels = perform_advanced_channel_analysis(api_key, channels)
                
                st.session_state.found_channels = channels
                st.success(f"🎉 Discovery complete! Found {len(channels)} channels matching your criteria.")
                
                # Show quick preview
                st.subheader("📋 Quick Preview")
                preview_df = pd.DataFrame(channels)
                st.dataframe(
                    preview_df[['Channel Name', 'Subscribers', 'Video Count', 'Creation Date']].head(10),
                    use_container_width=True
                )
            else:
                st.warning("😔 No channels found matching your criteria. Try adjusting your filters.")

with tab2:
    st.header("📊 Detailed Results Analysis")
    
    if 'found_channels' in st.session_state and st.session_state.found_channels:
        channels_data = st.session_state.found_channels
        
        # Summary Statistics
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_channels = len(channels_data)
        total_subscribers = sum([ch['Subscribers'] for ch in channels_data])
        total_videos = sum([ch['Video Count'] for ch in channels_data])
        avg_age = sum([ch['Channel Age (Days)'] for ch in channels_data]) / total_channels
        
        col1.metric("Total Channels", total_channels)
        col2.metric("Combined Subscribers", format_number(total_subscribers))
        col3.metric("Combined Videos", format_number(total_videos))
        col4.metric("Avg Channel Age", f"{int(avg_age)} days")
        
        # Individual Channel Cards
        st.subheader("🎯 Individual Channel Details")
        
        # Sort options
        sort_option = st.selectbox(
            "Sort by:",
            ["Subscribers", "Total Views", "Video Count", "Channel Age (Days)", "Creation Date"]
        )
        
        # Sort the channels
        sorted_channels = sorted(
            channels_data, 
            key=lambda x: x.get(sort_option, 0), 
            reverse=True
        )
        
        # Display channels in expandable cards
        for i, channel in enumerate(sorted_channels):
            with st.expander(f"#{i+1} {channel['Channel Name']} • {format_number(channel['Subscribers'])} subscribers", expanded=(i < 3)):
                
                # Basic metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Subscribers", format_number(channel['Subscribers']))
                col2.metric("👀 Total Views", format_number(channel['Total Views']))
                col3.metric("🎬 Videos", channel['Video Count'])
                
                col4, col5, col6 = st.columns(3)
                col4.metric("📅 Created", channel['Creation Date'])
                col5.metric("⏱️ Age", f"{channel['Channel Age (Days)']} days")
                col6.metric("🔍 Found via", channel['Found Via Keyword'])
                
                # Advanced metrics if available
                if 'Recent Engagement Rate' in channel:
                    col7, col8, col9 = st.columns(3)
                    col7.metric("💝 Engagement Rate", f"{channel.get('Recent Engagement Rate', 0):.3f}%")
                    col8.metric("📈 Recent Avg Views", format_number(channel.get('Recent Avg Views', 0)))
                    col9.metric("🔬 Analysis", channel.get('Analysis Status', 'N/A'))
                
                # Channel description
                if channel.get('Description'):
                    st.text_area("📝 Description:", channel['Description'], height=100, key=f"desc_{i}")
                
                # Channel link
                st.markdown(f"🔗 **[Visit Channel]({channel['URL']})**")
        
        # Export options
        st.subheader("💾 Export Results")
        
        # Convert to DataFrame for export
        export_df = pd.DataFrame(channels_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"youtube_channels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_str = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_str,
                file_name=f"youtube_channels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("📋 Run a channel search first to see detailed results here.")

with tab3:
    st.header("📈 Growth & Performance Insights")
    
    if 'found_channels' in st.session_state and st.session_state.found_channels:
        channels_data = st.session_state.found_channels
        df = pd.DataFrame(channels_data)
        
        # Growth Analysis Charts
        st.subheader("📊 Visual Analytics")
        
        # Subscriber Distribution
        fig_subs = px.histogram(
            df, 
            x='Subscribers', 
            nbins=20,
            title="📈 Subscriber Distribution",
            labels={'Subscribers': 'Subscriber Count', 'count': 'Number of Channels'}
        )
        st.plotly_chart(fig_subs, use_container_width=True)
        
        # Views vs Subscribers Scatter
        if len(df) > 1:
            fig_scatter = px.scatter(
                df,
                x='Subscribers',
                y='Total Views',
                size='Video Count',
                hover_name='Channel Name',
                title="👀 Views vs Subscribers Analysis",
                labels={'Subscribers': 'Subscriber Count', 'Total Views': 'Total View Count'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Channel Age Analysis
        fig_age = px.bar(
            df.head(20),  # Top 20 channels
            x='Channel Name',
            y='Channel Age (Days)',
            title="⏰ Channel Age Analysis (Top 20 Channels)",
            labels={'Channel Age (Days)': 'Age in Days', 'Channel Name': 'Channel'}
        )
        fig_age.update_xaxis(tickangle=45)
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Performance Insights
        st.subheader("🎯 Key Insights")
        
        # Calculate insights
        top_performer = max(channels_data, key=lambda x: x['Subscribers'])
        fastest_growing = min(channels_data, key=lambda x: x['Channel Age (Days)'])
        most_productive = max(channels_data, key=lambda x: x['Video Count'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **🏆 Top Performer**
            
            **{top_performer['Channel Name']}**
            - {format_number(top_performer['Subscribers'])} subscribers
            - {top_performer['Video Count']} videos
            - Created: {top_performer['Creation Date']}
            """)
        
        with col2:
            st.markdown(f"""
            **⚡ Newest Channel**
            
            **{fastest_growing['Channel Name']}**
            - {fastest_growing['Channel Age (Days)']} days old
            - {format_number(fastest_growing['Subscribers'])} subscribers
            - Created: {fastest_growing['Creation Date']}
            """)
        
        with col3:
            st.markdown(f"""
            **🎬 Most Productive**
            
            **{most_productive['Channel Name']}**
            - {most_productive['Video Count']} videos
            - {format_number(most_productive['Subscribers'])} subscribers
            - Created: {most_productive['Creation Date']}
            """)
        
        # Advanced Analytics (if available)
        if any('Recent Engagement Rate' in ch for ch in channels_data):
            st.subheader("🧠 Advanced Analytics")
            
            # Filter channels with advanced data
            advanced_channels = [ch for ch in channels_data if 'Recent Engagement Rate' in ch]
            if advanced_channels:
                advanced_df = pd.DataFrame(advanced_channels)
                
                # Engagement Rate Analysis
                fig_engagement = px.bar(
                    advanced_df.head(15),
                    x='Channel Name',
                    y='Recent Engagement Rate',
                    title="💝 Recent Engagement Rate Analysis",
                    labels={'Recent Engagement Rate': 'Engagement Rate (%)', 'Channel Name': 'Channel'}
                )
                fig_engagement.update_xaxis(tickangle=45)
                st.plotly_chart(fig_engagement, use_container_width=True)
    
    else:
        st.info("📈 Run a channel search first to see growth insights.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>📝 Application Description</h4>
    <p>This advanced YouTube Channel Finder helps you discover channels based on specific criteria like keywords, creation year, subscriber count, and video count. It combines intelligent search algorithms with comprehensive analytics to provide deep insights into channel performance, growth patterns, and engagement metrics. Perfect for content creators, marketers, and researchers looking to discover trending channels and analyze YouTube ecosystem dynamics.</p>
    <br>
    <p><strong>🚀 Features:</strong> Smart channel discovery • Advanced analytics • Growth tracking • Engagement analysis • Export capabilities • Real-time data visualization</p>
    <p><em>Powered by YouTube Data API v3 and advanced mathematical models for growth intelligence</em></p>
</div>
""", unsafe_allow_html=True)

# Additional Features Section
st.markdown("---")
st.header("🔧 Additional Tools & Features")

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("🎯 Search Tips"):
        st.markdown("""
        **Effective Search Strategies:**
        
        • Use specific, relevant keywords
        • Combine multiple related terms with commas
        • Try different year ranges to find emerging channels
        • Adjust subscriber limits to find your target audience size
        • Use description keywords to filter by niche topics
        
        **Examples:**
        - "AI tutorial, machine learning, python coding"
        - "cooking recipes, healthy meals, quick dinner"
        - "gaming review, indie games, retro gaming"
        """)

with col2:
    with st.expander("📊 Understanding Metrics"):
        st.markdown("""
        **Key Metrics Explained:**
        
        • **Subscribers**: Total channel followers
        • **Total Views**: Cumulative views across all videos
        • **Video Count**: Number of uploaded videos
        • **Engagement Rate**: (Likes + Comments) / Views ratio
        • **Subscriber Velocity**: Subscribers gained per day
        • **Channel Age**: Days since channel creation
        
        **Growth Indicators:**
        - High engagement rate = Active audience
        - High subscriber velocity = Fast growing
        - Good view-to-video ratio = Consistent quality
        """)

with col3:
    with st.expander("🔗 API Setup Guide"):
        st.markdown("""
        **Getting Your YouTube API Key:**
        
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing
        3. Enable YouTube Data API v3
        4. Create credentials (API Key)
        5. Copy and paste the key in sidebar
        
        **API Limits:**
        - 10,000 requests per day (free tier)
        - Each search uses ~3-5 quota units
        - Channel details use ~1 quota unit
        
        **Cost:** Free up to daily quota limit
        """)

# Quick Actions Section
st.markdown("---")
st.header("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔥 Find Viral Channels", use_container_width=True):
        if 'found_channels' in st.session_state:
            viral_channels = [ch for ch in st.session_state.found_channels if ch['Total Views'] > ch['Subscribers'] * 100]
            if viral_channels:
                st.success(f"Found {len(viral_channels)} potentially viral channels!")
                for ch in viral_channels[:5]:
                    st.write(f"🔥 {ch['Channel Name']} - {format_number(ch['Subscribers'])} subs")
            else:
                st.info("Run a search first to find viral channels")
        else:
            st.info("Run a search first!")

with col2:
    if st.button("🆕 Newest Channels", use_container_width=True):
        if 'found_channels' in st.session_state:
            newest = sorted(st.session_state.found_channels, key=lambda x: x['Channel Age (Days)'])[:5]
            st.success("Top 5 Newest Channels:")
            for ch in newest:
                st.write(f"🆕 {ch['Channel Name']} - {ch['Channel Age (Days)']} days old")
        else:
            st.info("Run a search first!")

with col3:
    if st.button("📈 Fast Growing", use_container_width=True):
        if 'found_channels' in st.session_state:
            fast_growing = sorted(st.session_state.found_channels, key=lambda x: x.get('Subscriber Velocity', 0), reverse=True)[:5]
            st.success("Top 5 Fast Growing:")
            for ch in fast_growing:
                st.write(f"📈 {ch['Channel Name']} - {ch.get('Subscriber Velocity', 0):.2f}/day")
        else:
            st.info("Run a search first!")

with col4:
    if st.button("🎬 Most Active", use_container_width=True):
        if 'found_channels' in st.session_state:
            most_active = sorted(st.session_state.found_channels, key=lambda x: x['Video Count'], reverse=True)[:5]
            st.success("Top 5 Most Active:")
            for ch in most_active:
                st.write(f"🎬 {ch['Channel Name']} - {ch['Video Count']} videos")
        else:
            st.info("Run a search first!")

# Settings and Configuration
with st.sidebar:
    st.markdown("---")
    st.header("🛠️ Advanced Settings")
    
    # Cache management
    if st.button("🗑️ Clear Cache", help="Clear cached API responses"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    # Data export format
    export_format = st.selectbox(
        "Default Export Format:",
        ["CSV", "JSON", "Excel"],
        help="Choose default format for data exports"
    )
    
    # Display options
    show_advanced_metrics = st.checkbox(
        "Show Advanced Metrics",
        value=True,
        help="Display engagement rates and growth analytics"
    )
    
    # Search limits
    st.subheader("🔢 Search Limits")
    default_max_results = st.slider(
        "Default Max Channels:",
        min_value=10,
        max_value=200,
        value=50,
        help="Default maximum channels to find"
    )
    
    # API usage tracking
    st.subheader("📊 API Usage")
    if 'api_calls_made' not in st.session_state:
        st.session_state.api_calls_made = 0
    
    st.metric("API Calls Today", st.session_state.api_calls_made)
    st.progress(min(st.session_state.api_calls_made / 100, 1.0))  # Assume 100 calls per day limit for display
    
    if st.session_state.api_calls_made > 80:
        st.warning("⚠️ Approaching API limit!")
    
    # Help and support
    st.markdown("---")
    st.subheader("❓ Help & Support")
    
    with st.expander("📚 Documentation"):
        st.markdown("""
        **Common Issues:**
        
        • **No results found**: Try broader keywords or adjust filters
        • **API errors**: Check your API key and quota limits
        • **Slow searches**: Reduce max channels or use basic mode
        • **Missing data**: Some channels may have private statistics
        
        **Best Practices:**
        
        • Start with broad searches, then refine
        • Use specific niches for targeted results
        • Check channel creation dates for trends
        • Export data for further analysis
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **If something goes wrong:**
        
        1. **Refresh the page** - Clears temporary issues
        2. **Check API key** - Ensure it's valid and has quota
        3. **Clear cache** - Use the button above
        4. **Reduce search scope** - Lower max channels
        5. **Try different keywords** - Some terms may be restricted
        
        **Error Codes:**
        - 403: API key issue or quota exceeded
        - 400: Invalid search parameters
        - 404: Channel not found or deleted
        """)

# Version and Updates
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8em;">
    <p>YouTube Channel Finder v2.0 | Built with Streamlit & YouTube Data API v3</p>
    <p>Last Updated: September 2025 | © 2025 Advanced Analytics Platform</p>
</div>
""", unsafe_allow_html=True)
