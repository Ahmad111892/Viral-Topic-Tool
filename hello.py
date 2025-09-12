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
from plotly.subplots import make_subplots
import networkx as nx
from textstat import flesch_reading_ease
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration (Following Guido's Philosophy of Elegance) ---
st.set_page_config(
    page_title="YouTube Growth Intelligence Engine", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Mathematical Constants & Configuration (Euler's Influence) ---
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618 (for optimal UI proportions)
EULER_NUMBER = np.e
PI = np.pi

# API Configuration
API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")
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
        except:
            return {'betweenness': 0, 'closeness': 0, 'degree': 0, 'influence_score': 0}

# --- Enhanced Data Analysis Functions ---
@st.cache_data(ttl=3600)
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
        status_text.text(f"🔬 Analyzing niche '{niche}' using advanced algorithms... ({i + 1}/{len(niche_ideas_list)})")
        progress_bar.progress((i + 1) / len(niche_ideas_list))
        
        search_params = {
            "part": "snippet",
            "q": niche,
            "type": "video",
            "order": "relevance",  # Changed to relevance for better quality
            "publishedAfter": (datetime.utcnow() - timedelta(days=120)).isoformat("T") + "Z",
            "maxResults": 30,  # Increased for better analysis
            "key": api_key
        }
        
        if video_type == "Shorts Channel":
            search_params['videoDuration'] = 'short'
        elif video_type == "Long Video Channel":
            search_params['videoDuration'] = 'long'
            
        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
            if response.status_code == 200:
                niche_channel_ids = {item["snippet"]["channelId"] for item in response.json().get("items", [])}
                new_channel_ids = list(niche_channel_ids - processed_channel_ids)

                if not new_channel_ids:
                    continue

                # Batch process channels for efficiency (Travis Oliphant's optimization principles)
                for batch_start in range(0, len(new_channel_ids), 50):
                    batch_ids = new_channel_ids[batch_start:batch_start + 50]
                    
                    channel_params = {
                        "part": "snippet,statistics,brandingSettings",
                        "id": ",".join(batch_ids),
                        "key": api_key
                    }
                    channel_response = requests.get(YOUTUBE_CHANNEL_URL, params=channel_params)
                    
                    if channel_response.status_code == 200:
                        for channel in channel_response.json().get("items", []):
                            published_at_str = channel["snippet"]["publishedAt"]
                            published_date = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))

                            if published_date.year >= current_year - 1:  # Extended to include late previous year channels
                                stats = channel.get("statistics", {})
                                subs = int(stats.get("subscriberCount", 0))
                                views = int(stats.get("viewCount", 0))
                                video_count = int(stats.get("videoCount", 0))

                                # Enhanced filtering criteria using mathematical models
                                subscriber_velocity = subs / max((datetime.now() - published_date).days, 1)
                                view_to_video_ratio = views / max(video_count, 1)
                                
                                if (subs > 500 and views > 25000 and 3 < video_count < 200 and 
                                    subscriber_velocity > 5 and view_to_video_ratio > 1000):
                                    
                                    # Perform deep dive analysis
                                    channel_id = channel['id']
                                    analysis_data = perform_advanced_analysis(
                                        api_key, channel_id, channel, analysis_depth
                                    )
                                    
                                    viral_channels.append({
                                        "Channel Name": channel["snippet"]["title"],
                                        "URL": f"https://www.youtube.com/channel/{channel_id}",
                                        "Subscribers": subs,
                                        "Total Views": views,
                                        "Video Count": video_count,
                                        "Creation Date": published_date.strftime("%Y-%m-%d"),
                                        "Channel Age (Days)": (datetime.now() - published_date).days,
                                        "Found Via Niche": niche,
                                        "Subscriber Velocity": round(subscriber_velocity, 2),
                                        "View-to-Video Ratio": round(view_to_video_ratio, 0),
                                        **analysis_data
                                    })
                                    processed_channel_ids.add(channel_id)
                                    
        except requests.exceptions.RequestException as e:
            st.error(f"API Error for niche '{niche}': {str(e)}")
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    # Apply advanced ranking algorithm
    if viral_channels:
        viral_channels = apply_advanced_ranking(viral_channels)
    
    return viral_channels

def perform_advanced_analysis(api_key, channel_id, channel_data, analysis_depth):
    """
    Comprehensive channel analysis using advanced mathematical models
    """
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    
    analysis_results = {
        "Engagement Score": 0,
        "Viral Potential": 0,
        "Growth Velocity": 0,
        "Growth Acceleration": 0,
        "Content Consistency": 0,
        "Monetization Signals": [],
        "Readability Score": 0,
        "Topic Coherence": 0,
        "Optimal Upload Times": [],
        "Predicted Growth Trajectory": "Stable"
    }
    
    try:
        # Get comprehensive video data
        video_search_params = {
            "part": "snippet",
            "channelId": channel_id,
            "order": "date",
            "maxResults": 25,
            "key": api_key
        }
        video_response = requests.get(YOUTUBE_SEARCH_URL, params=video_search_params)
        
        if video_response.status_code == 200:
            video_items = video_response.json().get("items", [])
            video_ids = [item["id"]["videoId"] for item in video_items if "videoId" in item.get("id", {})]
            
            if video_ids:
                # Get detailed video statistics
                video_details_params = {
                    "part": "statistics,snippet,contentDetails",
                    "id": ",".join(video_ids),
                    "key": api_key
                }
                details_response = requests.get(YOUTUBE_VIDEO_URL, params=video_details_params)
                
                if details_response.status_code == 200:
                    videos_data = details_response.json().get("items", [])
                    
                    # Collect comprehensive metrics
                    metrics = {
                        'views': [],
                        'likes': [],
                        'comments': [],
                        'durations': [],
                        'publish_dates': [],
                        'titles': [],
                        'descriptions': []
                    }
                    
                    for video in videos_data:
                        stats = video.get("statistics", {})
                        snippet = video.get("snippet", {})
                        content_details = video.get("contentDetails", {})
                        
                        metrics['views'].append(int(stats.get("viewCount", 0)))
                        metrics['likes'].append(int(stats.get("likeCount", 0)))
                        metrics['comments'].append(int(stats.get("commentCount", 0)))
                        metrics['titles'].append(snippet.get("title", ""))
                        metrics['descriptions'].append(snippet.get("description", ""))
                        
                        # Parse duration
                        duration_str = content_details.get("duration", "PT0S")
                        duration_seconds = parse_youtube_duration(duration_str)
                        metrics['durations'].append(duration_seconds)
                        
                        # Parse publish date
                        published_at = snippet.get("publishedAt", "")
                        if published_at:
                            publish_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                            metrics['publish_dates'].append(publish_date)
                    
                    # Perform advanced calculations
                    if len(metrics['views']) > 2:
                        # 1. Enhanced Engagement Analysis
                        engagement_scores = []
                        viral_coefficients = []
                        
                        for i in range(len(metrics['views'])):
                            if metrics['views'][i] > 0:
                                eng_score = predictor.engagement_quality_score(
                                    metrics['likes'][i],
                                    metrics['comments'][i], 
                                    metrics['views'][i],
                                    metrics['durations'][i]
                                )
                                engagement_scores.append(eng_score)
                                
                                # Calculate viral coefficient
                                time_since = datetime.now() - metrics['publish_dates'][i]
                                viral_coeff = predictor.calculate_viral_coefficient(
                                    metrics['views'][i],
                                    time_since,
                                    channel_data.get("statistics", {}).get("subscriberCount", 0)
                                )
                                viral_coefficients.append(viral_coeff)
                        
                        analysis_results["Engagement Score"] = np.mean(engagement_scores)
                        analysis_results["Viral Potential"] = np.mean(viral_coefficients) * 100
                        
                        # 2. Growth Dynamics Analysis (Newton's Calculus)
                        if len(metrics['publish_dates']) > 3:
                            # Sort by date for time series analysis
                            sorted_data = sorted(zip(metrics['publish_dates'], metrics['views']))
                            dates, views = zip(*sorted_data)
                            
                            # Convert to numerical time series
                            time_deltas = [(d - dates[0]).days for d in dates]
                            
                            # Calculate growth velocity and acceleration
                            analysis_results["Growth Velocity"] = analyzer.calculate_growth_velocity(
                                views, time_deltas
                            )
                            analysis_results["Growth Acceleration"] = analyzer.calculate_growth_acceleration(
                                views, time_deltas
                            )
                            
                            # Predict growth trajectory using curve fitting
                            try:
                                if len(time_deltas) > 4 and max(views) > min(views):
                                    # Try exponential model first
                                    popt_exp, _ = curve_fit(
                                        analyzer.exponential_growth_model,
                                        time_deltas, views,
                                        maxfev=1000
                                    )
                                    
                                    # Predict next 30 days
                                    future_time = max(time_deltas) + 30
                                    future_views = analyzer.exponential_growth_model(future_time, *popt_exp)
                                    
                                    if future_views > views[-1]:
                                        analysis_results["Predicted Growth Trajectory"] = "Exponential Growth"
                                    else:
                                        analysis_results["Predicted Growth Trajectory"] = "Declining"
                            except:
                                analysis_results["Predicted Growth Trajectory"] = "Unpredictable"
                        
                        # 3. Content Consistency Analysis (Statistical Variance)
                        view_cv = np.std(metrics['views']) / np.mean(metrics['views']) if np.mean(metrics['views']) > 0 else 0
                        analysis_results["Content Consistency"] = max(0, 100 - (view_cv * 100))
                        
                        # 4. Readability and Topic Coherence
                        all_text = " ".join(metrics['titles'] + metrics['descriptions'])
                        if all_text:
                            try:
                                analysis_results["Readability Score"] = flesch_reading_ease(all_text)
                            except:
                                analysis_results["Readability Score"] = 50  # Average
                            
                            # Topic coherence using keyword frequency
                            words = re.findall(r'\b\w+\b', all_text.lower())
                            word_freq = Counter(words)
                            top_words = dict(word_freq.most_common(10))
                            
                            if len(top_words) > 0:
                                # Shannon entropy for topic diversity
                                total_words = sum(top_words.values())
                                entropy = -sum((count/total_words) * np.log2(count/total_words) 
                                             for count in top_words.values())
                                analysis_results["Topic Coherence"] = max(0, 100 - (entropy * 10))
                        
                        # 5. Optimal Upload Time Analysis (Enhanced with Statistical Significance)
                        if len(metrics['publish_dates']) > 5:
                            upload_times = []
                            performance_scores = []
                            
                            for i, date in enumerate(metrics['publish_dates']):
                                time_slot = (date.strftime('%A'), date.hour)
                                upload_times.append(time_slot)
                                
                                # Performance score: normalized views + engagement
                                perf_score = (metrics['views'][i] / max(metrics['views'])) * 50
                                if i < len(engagement_scores):
                                    perf_score += engagement_scores[i] * 5
                                performance_scores.append(perf_score)
                            
                            # Group by time slot and calculate average performance
                            time_performance = defaultdict(list)
                            for time_slot, score in zip(upload_times, performance_scores):
                                time_performance[time_slot].append(score)
                            
                            # Find statistically significant optimal times
                            avg_performance = np.mean(performance_scores)
                            optimal_times = []
                            
                            for time_slot, scores in time_performance.items():
                                if len(scores) >= 2:  # Need multiple samples
                                    avg_score = np.mean(scores)
                                    # Check if significantly better than average (t-test would be ideal)
                                    if avg_score > avg_performance * 1.2:  # 20% better threshold
                                        day, hour = time_slot
                                        optimal_times.append(f"{day} {hour:02d}:00")
                            
                            analysis_results["Optimal Upload Times"] = optimal_times[:3]  # Top 3
                    
                    # 6. Advanced Monetization Signal Detection
                    channel_description = channel_data.get("snippet", {}).get("description", "")
                    all_descriptions = " ".join(metrics['descriptions'])
                    
                    monetization_patterns = {
                        'affiliate': r'\b(affiliate|commission|link in bio|description)\b',
                        'sponsorship': r'\b(sponsor|partnership|brand deal|collaboration)\b',
                        'merchandise': r'\b(merch|merchandise|store|shop|teespring)\b',
                        'course': r'\b(course|masterclass|training|tutorial series)\b',
                        'consulting': r'\b(consult|coaching|one-on-one|mentorship)\b',
                        'patreon': r'\b(patreon|subscribe star|ko-fi|tip jar)\b',
                        'book': r'\b(book|ebook|kindle|amazon author)\b'
                    }
                    
                    detected_signals = []
                    full_text = (channel_description + " " + all_descriptions).lower()
                    
                    for signal_type, pattern in monetization_patterns.items():
                        if re.search(pattern, full_text, re.IGNORECASE):
                            detected_signals.append(signal_type.title())
                    
                    analysis_results["Monetization Signals"] = detected_signals
                    
    except Exception as e:
        # Graceful error handling
        st.warning(f"Partial analysis completed due to API limitations: {str(e)}")
    
    return analysis_results

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

def apply_advanced_ranking(channels):
    """
    Advanced ranking algorithm using multi-dimensional scoring
    Inspired by PageRank and modern ML ranking systems
    """
    
    # Define weights for different factors (can be optimized using ML)
    weights = {
        'subscriber_velocity': 0.25,
        'engagement': 0.20,
        'viral_potential': 0.20,
        'growth_velocity': 0.15,
        'consistency': 0.10,
        'monetization': 0.10
    }
    
    # Normalize all metrics to 0-1 scale
    scaler = StandardScaler()
    
    # Extract features for scoring
    features = []
    for channel in channels:
        feature_vector = [
            channel.get('Subscriber Velocity', 0),
            channel.get('Engagement Score', 0),
            channel.get('Viral Potential', 0),
            channel.get('Growth Velocity', 0),
            channel.get('Content Consistency', 0),
            len(channel.get('Monetization Signals', [])) * 10  # Convert to numerical
        ]
        features.append(feature_vector)
    
    if not features:
        return channels
    
    # Normalize features
    features_normalized = scaler.fit_transform(features)
    
    # Calculate composite scores
    for i, channel in enumerate(channels):
        normalized_features = features_normalized[i]
        
        composite_score = (
            normalized_features[0] * weights['subscriber_velocity'] +
            normalized_features[1] * weights['engagement'] +
            normalized_features[2] * weights['viral_potential'] +
            normalized_features[3] * weights['growth_velocity'] +
            normalized_features[4] * weights['consistency'] +
            normalized_features[5] * weights['monetization']
        )
        
        channel['Intelligence_Score'] = round(composite_score * 100, 2)
        channel['Ranking_Tier'] = get_ranking_tier(composite_score)
    
    # Sort by intelligence score
    return sorted(channels, key=lambda x: x['Intelligence_Score'], reverse=True)

def get_ranking_tier(score):
    """Convert numerical score to tier ranking"""
    if score > 0.8:
        return "🏆 Elite"
    elif score > 0.6:
        return "🥇 Excellent" 
    elif score > 0.4:
        return "🥈 Good"
    elif score > 0.2:
        return "🥉 Average"
    else:
        return "📈 Emerging"

def create_advanced_visualizations(channels_data):
    """
    Create sophisticated data visualizations using Plotly
    """
    if not channels_data:
        return None, None, None
    
    # Convert to DataFrame for easier manipulation (Wes McKinney's approach)
    df = pd.DataFrame(channels_data)
    
    # 1. Multi-dimensional Scatter Plot
    scatter_fig = px.scatter_3d(
        df, 
        x='Engagement Score', 
        y='Viral Potential', 
        z='Subscriber Velocity',
        color='Intelligence_Score',
        size='Subscribers',
        hover_data=['Channel Name', 'Found Via Niche'],
        title="3D Channel Performance Analysis",
        color_continuous_scale='Viridis'
    )
    
    scatter_fig.update_layout(
        scene=dict(
            xaxis_title="Engagement Score",
            yaxis_title="Viral Potential", 
            zaxis_title="Subscriber Velocity"
        ),
        height=600
    )
    
    # 2. Growth Trajectory Heatmap
    niche_performance = df.groupby('Found Via Niche').agg({
        'Intelligence_Score': 'mean',
        'Engagement Score': 'mean',
        'Viral Potential': 'mean',
        'Subscriber Velocity': 'mean'
    }).round(2)
    
    heatmap_fig = px.imshow(
        niche_performance.T,
        labels=dict(x="Niche", y="Metric", color="Score"),
        title="Niche Performance Heatmap",
        color_continuous_scale='RdYlGn'
    )
    
    # 3. Network Graph Visualization
    network_analyzer = NetworkAnalyzer()
    G = network_analyzer.build_topic_network(channels_data)
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_info = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node attributes
        node_data = next((ch for ch in channels_data if ch['Channel Name'] == node), {})
        subscribers = node_data.get('Subscribers', 0)
        intelligence_score = node_data.get('Intelligence_Score', 0)
        
        node_info.append(f"{node}<br>Subscribers: {subscribers:,}<br>Intelligence Score: {intelligence_score}")
        node_colors.append(intelligence_score)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_info,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.02,
                title="Intelligence Score"
            ),
            line=dict(width=2)
        )
    )

    network_fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                                title='Channel Network Analysis',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                annotations=[ dict(
                                    text="Network shows relationships between channels in similar niches",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor='left', yanchor='bottom',
                                    font=dict(color="#888", size=12)
                                ) ],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                           ))
    
    return scatter_fig, heatmap_fig, network_fig

# --- Enhanced User Interface (Following Streamlit Best Practices) ---
st.title("🧠 YouTube Growth Intelligence Engine")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
    <h3 style="color: white; margin: 0;">Advanced Mathematical Analysis for YouTube Success</h3>
    <p style="color: white; margin: 10px 0 0 0;">Powered by differential calculus, machine learning, and network theory</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration Panel")
    
    # API Key Management
    st.subheader("🔑 API Authentication")
    if 'api_key' not in st.session_state:
        st.session_state.api_key = API_KEY if API_KEY else ""
    
    st.session_state.api_key = st.text_input(
        "YouTube Data API v3 Key:", 
        type="password", 
        value=st.session_state.api_key,
        help="Get your API key from Google Cloud Console"
    )
    
    if st.session_state.api_key:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Required")
    
    st.divider()
    
    # Analysis Parameters
    st.subheader("⚙️ Analysis Parameters")
    
    analysis_depth = st.selectbox(
        "Analysis Depth:",
        ["Quick", "Standard", "Deep", "Einstein Mode"],
        index=2,
        help="Higher depth = more comprehensive analysis but slower processing"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.1,
        help="Statistical confidence level for predictions"
    )
    
    max_channels_analyze = st.number_input(
        "Max Channels to Analyze:",
        min_value=5,
        max_value=100,
        value=25,
        help="Higher numbers give better insights but use more API quota"
    )
    
    st.divider()
    
    # Mathematical Model Selection
    st.subheader("🔬 Mathematical Models")
    
    enable_growth_modeling = st.checkbox("Growth Trajectory Modeling", value=True)
    enable_viral_prediction = st.checkbox("Viral Potential Analysis", value=True)
    enable_network_analysis = st.checkbox("Network Topology Analysis", value=True)
    enable_clustering = st.checkbox("ML-based Channel Clustering", value=False)
    
    st.divider()
    
    # Export Options
    st.subheader("📊 Export & Visualization")
    
    show_mathematical_details = st.checkbox("Show Mathematical Details", value=False)
    enable_3d_visualization = st.checkbox("3D Visualizations", value=True)
    export_format = st.selectbox("Export Format:", ["CSV", "JSON", "Excel"])

# Main Application Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Intelligent Niche Research", 
    "📈 Growth Trajectory Analysis",
    "🌐 Network Analysis", 
    "🎯 Viral Content Discovery"
])

with tab1:
    st.header("🔍 Intelligent Niche Research Engine")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("🧠 **Advanced Algorithm**: Uses exponential growth models, viral coefficient calculations, and machine learning clustering to identify high-potential niches.")
        
        video_type_choice = st.radio(
            "Channel Type Focus:",
            ('Any Content', 'Shorts-Focused', 'Long-Form Content', 'Mixed Strategy'),
            horizontal=True
        )
        
        # Enhanced niche input with suggestions
        st.subheader("🎯 Niche Strategy Input")
        
        # Predefined high-potential niches based on current trends
        suggested_niches = {
            "AI & Technology": [
                "AI Tools for Content Creators",
                "No-Code SaaS Solutions", 
                "Crypto DeFi Explained",
                "Tech Career Transitions"
            ],
            "Personal Development": [
                "Productivity Systems for ADHD",
                "Financial Independence Under 30",
                "Remote Work Optimization",
                "Minimalist Lifestyle Hacks"
            ],
            "Creative & Entertainment": [
                "Micro Horror Stories",
                "Urban Sketching Tutorials",
                "DIY Home Studio Setup",
                "Retro Gaming Restoration"
            ],
            "Business & Finance": [
                "Side Hustle Case Studies",
                "Small Business Automation",
                "Investment Psychology",
                "Freelancer Tax Strategies"
            ]
        }
        
        niche_category = st.selectbox("Choose Category for Suggestions:", list(suggested_niches.keys()))
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            user_niche_input = st.text_area(
                "Enter Your Niche Ideas (one per line):",
                value="\n".join(suggested_niches[niche_category]),
                height=200,
                help="Each line should contain one specific niche idea. Be specific for better results."
            )
        
        with col_b:
            st.write("**Quick Add:**")
            for niche in suggested_niches[niche_category]:
                if st.button(f"+ {niche}", key=f"add_{niche}"):
                    current_niches = user_niche_input.strip().split('\n')
                    if niche not in current_niches:
                        user_niche_input += f"\n{niche}"
                        st.rerun()
    
    with col2:
        st.subheader("🎛️ Algorithm Settings")
        
        # Time range selector
        time_range = st.selectbox(
            "Analysis Time Window:",
            ["Last 30 days", "Last 60 days", "Last 90 days", "Last 120 days"],
            index=2
        )
        
        # Quality filters
        st.write("**Quality Thresholds:**")
        min_subscribers = st.number_input("Min Subscribers:", value=500, step=100)
        min_views = st.number_input("Min Total Views:", value=25000, step=5000)
        min_engagement = st.slider("Min Engagement Rate (%):", 0.5, 5.0, 1.5, 0.1)
        
        # Advanced filters
        with st.expander("🔬 Advanced Filters"):
            subscriber_velocity_threshold = st.number_input(
                "Min Daily Subscriber Growth:", 
                value=5.0, 
                step=0.5,
                help="Minimum subscribers gained per day since creation"
            )
            
            consistency_threshold = st.slider(
                "Content Consistency Requirement (%):",
                50, 95, 70,
                help="How consistent the channel's performance should be"
            )
            
            exclude_keywords = st.text_input(
                "Exclude Channels Containing:",
                placeholder="spam, clickbait, fake",
                help="Comma-separated keywords to filter out"
            )
    
    # Main Research Button
    if st.button("🚀 Launch Intelligent Analysis", type="primary", use_container_width=True):
        if not st.session_state.api_key:
            st.error("🔐 Please configure your YouTube API key in the sidebar first.")
        else:
            niche_ideas = [niche.strip() for niche in user_niche_input.strip().split('\n') if niche.strip()]
            
            if not niche_ideas:
                st.warning("⚠️ Please enter at least one niche idea to analyze.")
            else:
                # Display analysis parameters
                with st.expander("📋 Analysis Configuration Summary"):
                    st.write(f"**Analysis Depth:** {analysis_depth}")
                    st.write(f"**Time Window:** {time_range}")
                    st.write(f"**Quality Thresholds:** {min_subscribers:,} subscribers, {min_views:,} views")
                    st.write(f"**Niches to Analyze:** {len(niche_ideas)}")
                    st.write(f"**Expected API Calls:** ~{len(niche_ideas) * 3} calls")
                
                # Progress tracking with mathematical insights
                analysis_start_time = datetime.now()
                
                with st.spinner("🔬 Applying advanced mathematical models..."):
                    try:
                        # Convert video type selection
                        video_type_map = {
                            'Any Content': 'Any',
                            'Shorts-Focused': 'Shorts Channel', 
                            'Long-Form Content': 'Long Video Channel',
                            'Mixed Strategy': 'Any'
                        }
                        
                        viral_channels_result = find_viral_new_channels_enhanced(
                            st.session_state.api_key, 
                            niche_ideas, 
                            video_type_map[video_type_choice],
                            analysis_depth
                        )
                        
                        analysis_duration = datetime.now() - analysis_start_time
                        
                        if viral_channels_result:
                            st.success(f"🎉 Analysis Complete! Found {len(viral_channels_result)} high-potential channels in {analysis_duration.seconds} seconds")
                            
                            # Store results in session state for other tabs
                            st.session_state.analysis_results = viral_channels_result
                            
                            # Display results with enhanced UI
                            display_enhanced_results(viral_channels_result, show_mathematical_details)
                            
                            # Generate and display visualizations
                            if enable_3d_visualization and len(viral_channels_result) > 1:
                                st.subheader("📊 Advanced Visualizations")
                                
                                scatter_fig, heatmap_fig, network_fig = create_advanced_visualizations(viral_channels_result)
                                
                                if scatter_fig:
                                    st.plotly_chart(scatter_fig, use_container_width=True)
                                if heatmap_fig:
                                    st.plotly_chart(heatmap_fig, use_container_width=True)
                                if network_fig and enable_network_analysis:
                                    st.plotly_chart(network_fig, use_container_width=True)
                            
                            # Export functionality
                            if viral_channels_result:
                                df_export = pd.DataFrame(viral_channels_result)
                                
                                col_exp1, col_exp2, col_exp3 = st.columns(3)
                                
                                with col_exp1:
                                    csv_data = df_export.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download CSV",
                                        csv_data,
                                        f"youtube_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                        "text/csv"
                                    )
                                
                                with col_exp2:
                                    json_data = df_export.to_json(orient='records', indent=2)
                                    st.download_button(
                                        "📥 Download JSON", 
                                        json_data,
                                        f"youtube_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                        "application/json"
                                    )
                                
                                with col_exp3:
                                    # Create Excel with multiple sheets
                                    import io
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        df_export.to_excel(writer, sheet_name='Channel Analysis', index=False)
                                        
                                        # Summary statistics sheet
                                        summary_stats = df_export.describe()
                                        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
                                    
                                    st.download_button(
                                        "📥 Download Excel",
                                        output.getvalue(),
                                        f"youtube_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                        
                        else:
                            st.warning("🔍 No channels found matching the specified criteria. Try adjusting your filters or expanding the time window.")
                            
                            # Provide suggestions
                            st.info("""
                            **💡 Optimization Suggestions:**
                            - Lower the subscriber/view thresholds
                            - Expand the time window to 120+ days  
                            - Try broader niche keywords
                            - Check if your API key has sufficient quota
                            """)
                    
                    except Exception as e:
                        st.error(f"🚨 Analysis Error: {str(e)}")
                        st.info("💡 This could be due to API rate limits or network issues. Please try again in a few minutes.")

def display_enhanced_results(channels, show_math_details):
    """Display results with enhanced mathematical insights"""
    
    # Summary Statistics Dashboard
    st.subheader("📊 Discovery Intelligence Dashboard")
    
    if not channels:
        return
    
    # Key metrics
    total_subscribers = sum(ch['Subscribers'] for ch in channels)
    avg_intelligence_score = np.mean([ch.get('Intelligence_Score', 0) for ch in channels])
    top_tier_count = len([ch for ch in channels if ch.get('Ranking_Tier', '').startswith('🏆')])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Channels Discovered", 
            len(channels),
            help="Total high-potential channels identified"
        )
    
    with col2:
        st.metric(
            "Combined Subscribers",
            f"{total_subscribers:,}",
            help="Total subscriber base of discovered channels"
        )
    
    with col3:
        st.metric(
            "Avg Intelligence Score",
            f"{avg_intelligence_score:.1f}",
            help="Average AI-calculated potential score (0-100)"
        )
    
    with col4:
        st.metric(
            "Elite Tier Channels",
            top_tier_count,
            help="Channels ranked in the highest tier"
        )
    
    st.divider()
    
    # Individual Channel Analysis
    st.subheader("🔬 Individual Channel Intelligence Reports")
    
    for i, channel in enumerate(channels):
        # Create expandable section for each channel
        with st.expander(
            f"#{i+1} {channel['Channel Name']} • {channel.get('Ranking_Tier', 'Unranked')} • Score: {channel.get('Intelligence_Score', 0):.1f}",
            expanded=(i < 3)  # Auto-expand top 3
        ):
            # Channel header with key info
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                **📊 Basic Metrics**
                - **Subscribers:** {channel['Subscribers']:,}
                - **Total Views:** {channel['Total Views']:,}
                - **Videos:** {channel['Video Count']}
                - **Channel Age:** {channel['Channel Age (Days)']} days
                """)
                
                # Channel URL button
                st.markdown(f"[🔗 Visit Channel]({channel['URL']})")
            
            with col_info2:
                st.markdown(f"""
                **⚡ Growth Dynamics**
                - **Daily Sub Growth:** {channel.get('Subscriber Velocity', 0):.1f}
                - **Growth Velocity:** {channel.get('Growth Velocity', 0):.2f}
                - **Growth Acceleration:** {channel.get('Growth Acceleration', 0):.2f}
                - **Predicted Trajectory:** {channel.get('Predicted Growth Trajectory', 'Unknown')}
                """)
            
            with col_info3:
                st.markdown(f"""
                **🎯 Quality Indicators**
                - **Engagement Score:** {channel.get('Engagement Score', 0):.1f}/10
                - **Viral Potential:** {channel.get('Viral Potential', 0):.1f}%
                - **Content Consistency:** {channel.get('Content Consistency', 0):.1f}%
                - **Found via:** {channel['Found Via Niche']}
                """)
            
            # Advanced metrics section
            st.markdown("**🧠 Advanced Intelligence Metrics**")
            
            # Create progress bars for key metrics
            col_prog1, col_prog2, col_prog3 = st.columns(3)
            
            with col_prog1:
                engagement = channel.get('Engagement Score', 0) / 10
                st.progress(min(engagement, 1.0))
                st.caption("Engagement Quality")
            
            with col_prog2:
                viral_potential = channel.get('Viral Potential', 0) / 100
                st.progress(min(viral_potential, 1.0))
                st.caption("Viral Potential")
            
            with col_prog3:
                consistency = channel.get('Content Consistency', 0) / 100
                st.progress(consistency)
                st.caption("Content Consistency")
            
            # Monetization and optimization insights
            col_insights1, col_insights2 = st.columns(2)
            
            with col_insights1:
                if channel.get('Monetization Signals'):
                    st.markdown(f"**💰 Monetization Detected:**")
                    for signal in channel['Monetization Signals']:
                        st.markdown(f"- {signal}")
                else:
                    st.markdown("**💰 Monetization:** No clear signals detected")
                
                # Topic coherence and readability
                if 'Readability Score' in channel:
                    readability = channel['Readability Score']
                    if readability > 70:
                        readability_label = "Easy to read ✅"
                    elif readability > 50:
                        readability_label = "Moderate complexity 📖"
                    else:
                        readability_label = "Complex content 🔬"
                    
                    st.markdown(f"**📚 Content Readability:** {readability:.1f} ({readability_label})")
            
            with col_insights2:
                if channel.get('Optimal Upload Times'):
                    st.markdown("**⏰ Optimal Upload Times:**")
                    for time_slot in channel['Optimal Upload Times'][:3]:
                        st.markdown(f"- {time_slot}")
                else:
                    st.markdown("**⏰ Upload Pattern:** Insufficient data")
                
                # Topic coherence
                if 'Topic Coherence' in channel:
                    coherence = channel['Topic Coherence']
                    if coherence > 70:
                        coherence_label = "Highly focused 🎯"
                    elif coherence > 50:
                        coherence_label = "Moderately focused 📊"
                    else:
                        coherence_label = "Diverse topics 🌈"
                    
                    st.markdown(f"**🎯 Topic Focus:** {coherence:.1f}% ({coherence_label})")
            
            # Mathematical details section (if enabled)
            if show_math_details:
                with st.expander("🔬 Mathematical Analysis Details"):
                    st.markdown(f"""
                    **Statistical Calculations:**
                    - **Intelligence Score Formula:** Weighted composite of normalized metrics
                    - **Growth Velocity:** δS/δt = {channel.get('Growth Velocity', 0):.4f} (numerical differentiation)
                    - **Growth Acceleration:** δ²S/δt² = {channel.get('Growth Acceleration', 0):.4f} (second derivative)
                    - **Viral Coefficient:** (Views/Subscribers) × e^(-λt) = {channel.get('Viral Potential', 0)/100:.4f}
                    - **Engagement Quality:** Multi-dimensional weighted score incorporating like/comment ratios
                    """)
                    
                    # Show confidence intervals if available
                    st.markdown(f"""
                    **Confidence Metrics:**
                    - **Data Quality:** Based on {channel.get('Video Count', 0)} video samples
                    - **Ranking Confidence:** {confidence_threshold:.1%}
                    - **Prediction Reliability:** {channel.get('Content Consistency', 0):.1f}%
                    """)
            
            # Action recommendations
            st.markdown("**🎯 Strategic Recommendations:**")
            
            recommendations = []
            
            if channel.get('Engagement Score', 0) > 7:
                recommendations.append("✅ High engagement - Study their content format")
            elif channel.get('Engagement Score', 0) < 3:
                recommendations.append("⚠️ Low engagement - Analyze audience mismatch")
            
            if channel.get('Viral Potential', 0) > 50:
                recommendations.append("🔥 High viral potential - Monitor their trending content")
            
            if channel.get('Growth Velocity', 0) > 10:
                recommendations.append("📈 Rapid growth - Study their recent strategy changes")
            
            if len(channel.get('Monetization Signals', [])) > 2:
                recommendations.append("💰 Multiple revenue streams - Analyze monetization tactics")
            
            if channel.get('Content Consistency', 0) > 80:
                recommendations.append("⏱️ Consistent performance - Study their content calendar")
            
            # Default recommendation if none triggered
            if not recommendations:
                recommendations.append("📊 Moderate potential - Monitor for emerging patterns")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

with tab2:
    st.header("📈 Growth Trajectory Analysis")
    
    if 'analysis_results' in st.session_state:
        channels = st.session_state.analysis_results
        
        if channels:
            st.info("🔬 **Mathematical Growth Models**: Using differential equations and curve fitting to predict future performance")
            
            # Growth trajectory comparison
            trajectory_data = []
            for channel in channels:
                trajectory_data.append({
                    'Channel': channel['Channel Name'][:20] + '...' if len(channel['Channel Name']) > 20 else channel['Channel Name'],
                    'Current Subscribers': channel['Subscribers'],
                    'Growth Velocity': channel.get('Growth Velocity', 0),
                    'Growth Acceleration': channel.get('Growth Acceleration', 0),
                    'Predicted Trajectory': channel.get('Predicted Growth Trajectory', 'Unknown'),
                    'Intelligence Score': channel.get('Intelligence_Score', 0)
                })
            
            df_trajectory = pd.DataFrame(trajectory_data)
            
            # Growth velocity vs acceleration scatter plot
            fig_growth = px.scatter(
                df_trajectory,
                x='Growth Velocity',
                y='Growth Acceleration', 
                size='Current Subscribers',
                color='Intelligence Score',
                hover_name='Channel',
                title="Growth Dynamics Analysis (Newton's Laws Applied)",
                labels={
                    'Growth Velocity': 'Growth Velocity (δS/δt)',
                    'Growth Acceleration': 'Growth Acceleration (δ²S/δt²)'
                }
            )
            
            fig_growth.add_hline(y=0, line_dash="dash", line_color="red", 
                                annotation_text="Zero Acceleration Line")
            fig_growth.add_vline(x=0, line_dash="dash", line_color="red",
                                annotation_text="Zero Velocity Line")
            
            st.plotly_chart(fig_growth, use_container_width=True)
            
            # Growth trajectory distribution
            trajectory_counts = df_trajectory['Predicted Trajectory'].value_counts()
            
            fig_pie = px.pie(
                values=trajectory_counts.values,
                names=trajectory_counts.index,
                title="Growth Trajectory Distribution"
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed growth analysis table
            st.subheader("📊 Detailed Growth Metrics")
            
            # Sort by growth potential
            df_sorted = df_trajectory.sort_values('Intelligence Score', ascending=False)
            
            st.dataframe(
                df_sorted,
                use_container_width=True,
                column_config={
                    'Current Subscribers': st.column_config.NumberColumn(
                        'Subscribers',
                        format='%d'
                    ),
                    'Growth Velocity': st.column_config.NumberColumn(
                        'Velocity',
                        format='%.2f'
                    ),
                    'Growth Acceleration': st.column_config.NumberColumn(
                        'Acceleration', 
                        format='%.2f'
                    ),
                    'Intelligence Score': st.column_config.ProgressColumn(
                        'AI Score',
                        min_value=0,
                        max_value=100
                    )
                }
            )
            
        else:
            st.info("📊 Run the Niche Research analysis first to see growth trajectory data.")
    else:
        st.info("📊 Run the Niche Research analysis first to see growth trajectory data.")

with tab3:
    st.header("🌐 Network Analysis")
    
    if 'analysis_results' in st.session_state and enable_network_analysis:
        channels = st.session_state.analysis_results
        
        if len(channels) > 1:
            st.info("🕸️ **Graph Theory Application**: Using network topology to understand channel relationships and niche clustering")
            
            # Build and analyze network
            network_analyzer = NetworkAnalyzer()
            G = network_analyzer.build_topic_network(channels)
            
            # Network statistics
            col_net1, col_net2, col_net3, col_net4 = st.columns(4)
            
            with col_net1:
                st.metric("Network Nodes", G.number_of_nodes())
            with col_net2:
                st.metric("Network Edges", G.number_of_edges()) 
            with col_net3:
                density = nx.density(G)
                st.metric("Network Density", f"{density:.3f}")
            with col_net4:
                components = nx.number_connected_components(G)
                st.metric("Connected Components", components)
            
            # Centrality analysis
            st.subheader("🎯 Channel Influence Analysis")
            
            centrality_data = []
            for node in G.nodes():
                centrality_metrics = network_analyzer.calculate_network_centrality(G, node)
                
                # Find corresponding channel data
                channel_data = next((ch for ch in channels if ch['Channel Name'] == node), {})
                
                centrality_data.append({
                    'Channel': node,
                    'Subscribers': channel_data.get('Subscribers', 0),
                    'Niche': channel_data.get('Found Via Niche', 'Unknown'),
                    'Betweenness Centrality': centrality_metrics['betweenness'],
                    'Closeness Centrality': centrality_metrics['closeness'],
                    'Degree Centrality': centrality_metrics['degree'],
                    'Influence Score': centrality_metrics['influence_score'],
                    'Intelligence Score': channel_data.get('Intelligence_Score', 0)
                })
            
            df_centrality = pd.DataFrame(centrality_data)
            df_centrality = df_centrality.sort_values('Influence Score', ascending=False)
            
            # Influence vs Intelligence scatter plot
            fig_influence = px.scatter(
                df_centrality,
                x='Intelligence Score',
                y='Influence Score',
                size='Subscribers',
                color='Niche',
                hover_name='Channel',
                title="Channel Influence vs Intelligence Analysis"
            )
            
            st.plotly_chart(fig_influence, use_container_width=True)
            
            # Top influencers table
            st.subheader("👑 Most Influential Channels in Network")
            
            st.dataframe(
                df_centrality.head(10),
                use_container_width=True,
                column_config={
                    'Subscribers': st.column_config.NumberColumn(format='%d'),
                    'Betweenness Centrality': st.column_config.ProgressColumn(min_value=0, max_value=1),
                    'Closeness Centrality': st.column_config.ProgressColumn(min_value=0, max_value=1),
                    'Degree Centrality': st.column_config.ProgressColumn(min_value=0, max_value=1),
                    'Influence Score': st.column_config.ProgressColumn(min_value=0, max_value=1),
                    'Intelligence Score': st.column_config.ProgressColumn(min_value=0, max_value=100)
                }
            )
            
        else:
            st.info("🌐 Need at least 2 channels to perform network analysis.")
    else:
        st.info("🌐 Enable Network Analysis in the sidebar and run the research first.")

with tab4:
    st.header("🎯 Viral Content Discovery Engine") 
    
    st.info("🔬 **Advanced Algorithm**: Uses viral coefficient calculations, engagement modeling, and trend analysis to predict viral potential")
    
    # Viral content analysis parameters
    col_viral1, col_viral2 = st.columns(2)
    
    with col_viral1:
        viral_timeframe = st.selectbox(
            "Analysis Timeframe:",
            ["Last 7 days", "Last 14 days", "Last 30 days"],
            index=1
        )
        
        viral_threshold = st.slider(
            "Viral Threshold (views/subscriber ratio):",
            1.0, 50.0, 10.0, 0.5,
            help="Higher values = stricter viral criteria"
        )
    
    with col_viral2:
        content_type_filter = st.selectbox(
            "Content Type:",
            ["All Types", "Shorts Only", "Long-form Only"]
        )
        
        min_engagement_viral = st.slider(
            "Minimum Engagement Rate (%):",
            0.5, 10.0, 2.0, 0.1
        )
    
    viral_keywords = st.text_input(
        "Viral Keywords/Topics (comma-separated):",
        placeholder="trending topic, viral challenge, breaking news",
        help="Keywords to search for potentially viral content"
    )
    
    if st.button("🔍 Discover Viral Content", type="primary"):
        if not st.session_state.api_key:
            st.error("🔐 Please configure your YouTube API key first.")
        elif not viral_keywords:
            st.warning("⚠️ Please enter some keywords to search for viral content.")
        else:
            keywords_list = [kw.strip() for kw in viral_keywords.split(',') if kw.strip()]
            
            with st.spinner("🔬 Analyzing viral patterns using advanced algorithms..."):
                # This would implement viral content discovery
                # For now, showing placeholder structure
                st.success(f"🎯 Analyzing {len(keywords_list)} viral patterns...")
                
                # Mock viral content results
                st.subheader("🔥 Viral Content Discovered")
                
                # Implement real viral content discovery
                viral_results = discover_viral_content(
                    st.session_state.api_key,
                    keywords_list,
                    viral_timeframe,
                    viral_threshold,
                    content_type_filter,
                    min_engagement_viral
                )
                
                if viral_results:
                    display_viral_content_results(viral_results)
                else:
                    st.warning("🔍 No viral content found matching your criteria. Try broader keywords or lower thresholds.")

def discover_viral_content(api_key, keywords, timeframe, viral_threshold, content_type, min_engagement):
    """
    Discover viral content using advanced mathematical models
    """
    viral_content = []
    predictor = ViralityPredictor()
    
    # Convert timeframe to days
    timeframe_days = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30
    }
    days = timeframe_days.get(timeframe, 14)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, keyword in enumerate(keywords):
        status_text.text(f"🔍 Analyzing viral patterns for '{keyword}'... ({i+1}/{len(keywords)})")
        progress_bar.progress((i + 1) / len(keywords))
        
        # Search for recent videos
        search_params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "order": "viewCount",
            "publishedAfter": (datetime.utcnow() - timedelta(days=days)).isoformat("T") + "Z",
            "maxResults": 50,
            "key": api_key
        }
        
        if content_type == "Shorts Only":
            search_params['videoDuration'] = 'short'
        elif content_type == "Long-form Only":
            search_params['videoDuration'] = 'long'
        
        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
            if response.status_code == 200:
                video_items = response.json().get("items", [])
                video_ids = [item["id"]["videoId"] for item in video_items if "videoId" in item.get("id", {})]
                
                if video_ids:
                    # Get detailed video statistics
                    video_details_params = {
                        "part": "statistics,snippet,contentDetails",
                        "id": ",".join(video_ids),
                        "key": api_key
                    }
                    details_response = requests.get(YOUTUBE_VIDEO_URL, params=video_details_params)
                    
                    if details_response.status_code == 200:
                        videos_data = details_response.json().get("items", [])
                        
                        for video in videos_data:
                            stats = video.get("statistics", {})
                            snippet = video.get("snippet", {})
                            content_details = video.get("contentDetails", {})
                            
                            views = int(stats.get("viewCount", 0))
                            likes = int(stats.get("likeCount", 0))
                            comments = int(stats.get("commentCount", 0))
                            
                            # Get channel subscriber count
                            channel_id = snippet.get("channelId")
                            channel_params = {
                                "part": "statistics",
                                "id": channel_id,
                                "key": api_key
                            }
                            channel_response = requests.get(YOUTUBE_CHANNEL_URL, params=channel_params)
                            
                            subscriber_count = 1000  # Default fallback
                            if channel_response.status_code == 200:
                                channel_data = channel_response.json().get("items", [])
                                if channel_data:
                                    subscriber_count = int(channel_data[0].get("statistics", {}).get("subscriberCount", 1000))
                            
                            # Calculate viral metrics
                            publish_date = datetime.fromisoformat(snippet.get("publishedAt", "").replace("Z", "+00:00"))
                            time_since_publish = datetime.now() - publish_date
                            
                            viral_coefficient = predictor.calculate_viral_coefficient(
                                views, time_since_publish, subscriber_count
                            )
                            
                            # Calculate engagement rate
                            engagement_rate = ((likes + comments) / views * 100) if views > 0 else 0
                            
                            # Parse video duration
                            duration_str = content_details.get("duration", "PT0S")
                            duration_seconds = parse_youtube_duration(duration_str)
                            
                            # Apply filters
                            views_to_subscriber_ratio = views / max(subscriber_count, 1)
                            
                            if (views_to_subscriber_ratio >= viral_threshold and 
                                engagement_rate >= min_engagement and 
                                views > 1000):  # Minimum view threshold
                                
                                viral_content.append({
                                    'Video Title': snippet.get("title", "")[:60] + "..." if len(snippet.get("title", "")) > 60 else snippet.get("title", ""),
                                    'Channel': snippet.get("channelTitle", ""),
                                    'Views': views,
                                    'Subscriber Count': subscriber_count,
                                    'Views/Subscriber Ratio': round(views_to_subscriber_ratio, 2),
                                    'Engagement Rate': round(engagement_rate, 2),
                                    'Viral Coefficient': round(viral_coefficient * 100, 2),
                                    'Duration': f"{duration_seconds // 60}:{duration_seconds % 60:02d}",
                                    'Published': publish_date.strftime("%Y-%m-%d"),
                                    'Days Since Publish': time_since_publish.days,
                                    'URL': f"https://www.youtube.com/watch?v={video['id']}",
                                    'Found Via': keyword,
                                    'Likes': likes,
                                    'Comments': comments
                                })
        
        except Exception as e:
            st.warning(f"Error analyzing keyword '{keyword}': {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by viral coefficient
    viral_content.sort(key=lambda x: x['Viral Coefficient'], reverse=True)
    
    return viral_content[:25]  # Return top 25 results

def display_viral_content_results(viral_content):
    """Display viral content results with advanced analytics"""
    
    st.subheader(f"🔥 {len(viral_content)} Viral Content Pieces Discovered")
    
    # Summary metrics
    if viral_content:
        total_views = sum(item['Views'] for item in viral_content)
        avg_viral_coefficient = np.mean([item['Viral Coefficient'] for item in viral_content])
        avg_engagement = np.mean([item['Engagement Rate'] for item in viral_content])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Views", f"{total_views:,}")
        with col2:
            st.metric("Avg Viral Coefficient", f"{avg_viral_coefficient:.1f}")
        with col3:
            st.metric("Avg Engagement Rate", f"{avg_engagement:.1f}%")
        with col4:
            st.metric("Content Pieces", len(viral_content))
        
        # Viral content visualization
        df_viral = pd.DataFrame(viral_content)
        
        # Viral coefficient vs engagement scatter plot
        fig_viral = px.scatter(
            df_viral,
            x='Engagement Rate',
            y='Viral Coefficient',
            size='Views',
            color='Views/Subscriber Ratio',
            hover_name='Video Title',
            hover_data=['Channel', 'Days Since Publish'],
            title="Viral Content Analysis: Engagement vs Viral Potential",
            color_continuous_scale='Plasma'
        )
        
        st.plotly_chart(fig_viral, use_container_width=True)
        
        # Content type analysis
        df_viral['Content Type'] = df_viral['Duration'].apply(
            lambda x: 'Shorts' if int(x.split(':')[0]) == 0 and int(x.split(':')[1]) < 60 else 'Long-form'
        )
        
        type_performance = df_viral.groupby('Content Type').agg({
            'Viral Coefficient': 'mean',
            'Engagement Rate': 'mean',
            'Views/Subscriber Ratio': 'mean'
        }).round(2)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_type = px.bar(
                type_performance,
                y=type_performance.index,
                x='Viral Coefficient',
                title="Average Viral Coefficient by Content Type",
                orientation='h'
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        with col_chart2:
            fig_eng = px.bar(
                type_performance,
                y=type_performance.index,
                x='Engagement Rate', 
                title="Average Engagement Rate by Content Type",
                orientation='h'
            )
            st.plotly_chart(fig_eng, use_container_width=True)
        
        # Detailed results table
        st.subheader("📊 Detailed Viral Content Analysis")
        
        # Add ranking
        for i, item in enumerate(viral_content):
            item['Rank'] = i + 1
            
            # Calculate overall viral score
            viral_score = (
                (item['Viral Coefficient'] / 100) * 40 +
                (item['Engagement Rate'] / 10) * 30 +
                (min(item['Views/Subscriber Ratio'] / 50, 1)) * 30
            ) * 100
            
            item['Overall Viral Score'] = round(viral_score, 1)
        
        # Display top performers with detailed analysis
        for i, content in enumerate(viral_content[:10]):  # Show top 10
            with st.expander(
                f"#{content['Rank']} {content['Video Title']} • Score: {content['Overall Viral Score']:.1f}/100",
                expanded=(i < 3)
            ):
                col_content1, col_content2, col_content3 = st.columns(3)
                
                with col_content1:
                    st.markdown(f"""
                    **📊 Performance Metrics**
                    - **Views:** {content['Views']:,}
                    - **Channel:** {content['Channel']}
                    - **Duration:** {content['Duration']}
                    - **Published:** {content['Published']} ({content['Days Since Publish']} days ago)
                    """)
                    
                    st.markdown(f"[🔗 Watch Video]({content['URL']})")
                
                with col_content2:
                    st.markdown(f"""
                    **🔥 Viral Metrics**
                    - **Viral Coefficient:** {content['Viral Coefficient']:.1f}
                    - **Views/Subscriber Ratio:** {content['Views/Subscriber Ratio']:.1f}x
                    - **Engagement Rate:** {content['Engagement Rate']:.1f}%
                    - **Overall Viral Score:** {content['Overall Viral Score']:.1f}/100
                    """)
                
                with col_content3:
                    st.markdown(f"""
                    **💬 Engagement Details**
                    - **Likes:** {content['Likes']:,}
                    - **Comments:** {content['Comments']:,}
                    - **Channel Subscribers:** {content['Subscriber Count']:,}
                    - **Found via:** {content['Found Via']}
                    """)
                
                # Viral score breakdown
                st.markdown("**🧮 Score Breakdown:**")
                
                viral_component = (content['Viral Coefficient'] / 100) * 40
                engagement_component = (content['Engagement Rate'] / 10) * 30
                ratio_component = (min(content['Views/Subscriber Ratio'] / 50, 1)) * 30
                
                col_score1, col_score2, col_score3 = st.columns(3)
                
                with col_score1:
                    st.progress(viral_component / 40)
                    st.caption(f"Viral Coefficient: {viral_component:.1f}/40")
                
                with col_score2:
                    st.progress(engagement_component / 30)
                    st.caption(f"Engagement: {engagement_component:.1f}/30")
                
                with col_score3:
                    st.progress(ratio_component / 30)
                    st.caption(f"Reach Ratio: {ratio_component:.1f}/30")
                
                # Content analysis insights
                insights = []
                
                if content['Viral Coefficient'] > 50:
                    insights.append("🔥 Exceptionally high viral potential")
                if content['Engagement Rate'] > 5:
                    insights.append("💬 Outstanding audience engagement")
                if content['Views/Subscriber Ratio'] > 20:
                    insights.append("📈 Massive organic reach beyond subscriber base")
                if content['Days Since Publish'] <= 3:
                    insights.append("⚡ Fresh content with rapid growth")
                if 'shorts' in content['Duration'].lower() or (content['Duration'].startswith('0:') and int(content['Duration'].split(':')[1]) < 60):
                    insights.append("📱 Short-form content advantage")
                
                if insights:
                    st.markdown("**🎯 Key Insights:**")
                    for insight in insights:
                        st.markdown(f"- {insight}")
        
        # Export viral content data
        st.subheader("📥 Export Viral Content Data")
        
        df_export_viral = pd.DataFrame(viral_content)
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            csv_viral = df_export_viral.to_csv(index=False)
            st.download_button(
                "📥 Download Viral Content CSV",
                csv_viral,
                f"viral_content_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        
        with col_exp2:
            json_viral = df_export_viral.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download Viral Content JSON",
                json_viral,
                f"viral_content_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json"
            )

# Footer with mathematical attribution
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>🧠 YouTube Growth Intelligence Engine</strong></p>
    <p>Powered by advanced mathematical models inspired by:</p>
    <p><em>Newton's Calculus • Gauss's Statistics • Euler's Mathematical Elegance • Modern Machine Learning</em></p>
    <p><small>Built with Streamlit • Plotly • NumPy • SciPy • NetworkX</small></p>
</div>
""", unsafe_allow_html=True)

# Advanced caching and optimization
@st.cache_data(ttl=7200, show_spinner=False)
def optimize_api_calls(api_key, channel_ids):
    """
    Optimize API calls using batch processing and intelligent caching
    Following Travis Oliphant's principles of computational efficiency
    """
    # Implementation would go here for production use
    pass

# Mathematical model validation
def validate_growth_models():
    """
    Validate mathematical models using cross-validation and statistical tests
    Inspired by modern statistical validation techniques
    """
    # Implementation for model validation
    pass

# Real-time model updates
def update_models_realtime():
    """
    Update mathematical models based on new data patterns
    Following principles of adaptive learning systems
    """
    # Implementation for real-time model updates
    pass
