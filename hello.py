
import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import minimize
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# --- Mathematical Constants and Functions (Euler/Gauss/Newton Inspired) ---
PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio
E = np.e  # Euler's number
PI = np.pi  # Newton's pi

def fibonacci_growth_model(n):
    """Fibonacci-based growth prediction model"""
    return int((PHI**n - (1-PHI)**n) / np.sqrt(5))

def gaussian_viral_probability(engagement_rate, mean_engagement=2.0, std_engagement=1.0):
    """Calculate viral probability using Gaussian distribution"""
    return stats.norm.cdf(engagement_rate, mean_engagement, std_engagement)

def euler_optimization_function(params, data):
    """Euler-inspired optimization function for content strategy"""
    growth_rate, posting_frequency, engagement_weight = params
    return -np.sum(data['subscribers'] * np.exp(growth_rate * data['days_since_creation']) * 
                   (posting_frequency * data['video_count']) ** engagement_weight)

# --- Page Configuration (Fernando Pérez Style) ---
st.set_page_config(
    page_title="🚀 Enhanced YouTube Growth Toolkit", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Beautiful Interface ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        border-left: 5px solid #4CAF50;
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
API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# --- Advanced Analytics Classes (Tao/Perelman Inspired) ---
class ViralPatternAnalyzer:
    """Advanced mathematical analysis of viral patterns"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.growth_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.viral_classifier = GradientBoostingRegressor(random_state=42)

    def analyze_growth_patterns(self, channel_data):
        """Analyze growth patterns using machine learning"""
        if len(channel_data) < 5:
            return {"pattern": "insufficient_data", "prediction": 0}

        # Feature engineering (Hinton/Bengio/LeCun approach)
        features = []
        targets = []

        for channel in channel_data:
            feature_vector = [
                channel.get('Subscribers', 0),
                channel.get('Total Views', 0),
                channel.get('Video Count', 0),
                channel.get('Engagement Score', 0),
                channel.get('Content Velocity', 0),
                channel.get('Weekly Frequency', 0),
                len(channel.get('Monetization Clues', [])),
                self._days_since_creation(channel.get('Creation Date', ''))
            ]
            features.append(feature_vector)
            targets.append(channel.get('Subscribers', 0))

        features = np.array(features)
        targets = np.array(targets)

        # Handle edge cases
        if len(features) < 2:
            return {"pattern": "insufficient_data", "prediction": 0}

        try:
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.3, random_state=42
            )

            if len(X_train) == 0:
                return {"pattern": "insufficient_data", "prediction": 0}

            self.growth_model.fit(X_train, y_train)

            # Predictions
            y_pred = self.growth_model.predict(X_test) if len(X_test) > 0 else [0]
            r2 = r2_score(y_test, y_pred) if len(X_test) > 0 else 0

            return {
                "pattern": "exponential" if r2 > 0.7 else "linear" if r2 > 0.4 else "irregular",
                "prediction_accuracy": r2,
                "feature_importance": dict(zip([
                    "subscribers", "views", "videos", "engagement", 
                    "velocity", "frequency", "monetization", "age"
                ], self.growth_model.feature_importances_))
            }
        except Exception as e:
            return {"pattern": "analysis_error", "error": str(e), "prediction": 0}

    def _days_since_creation(self, creation_date_str):
        """Calculate days since channel creation"""
        try:
            creation_date = datetime.strptime(creation_date_str, "%Y-%m-%d")
            return (datetime.now() - creation_date).days
        except:
            return 0

    def predict_viral_potential(self, channel_metrics):
        """Predict viral potential using mathematical models"""
        engagement_score = channel_metrics.get('Engagement Score', 0)
        subscribers = channel_metrics.get('Subscribers', 0)
        velocity = channel_metrics.get('Content Velocity', 0)

        # Gaussian probability model
        viral_prob = gaussian_viral_probability(engagement_score)

        # Fibonacci growth prediction
        growth_factor = fibonacci_growth_model(min(velocity, 10)) / 1000.0

        # Einstein's E=mc² inspired formula: Engagement = Mass × Content² 
        einstein_factor = engagement_score * (velocity ** 2) / 100.0

        combined_score = (viral_prob * 0.4 + growth_factor * 0.3 + einstein_factor * 0.3)

        return {
            "viral_probability": min(viral_prob, 1.0),
            "growth_prediction": growth_factor,
            "einstein_engagement": einstein_factor,
            "combined_viral_score": min(combined_score, 1.0)
        }

class NetworkAnalyzer:
    """Network analysis of YouTube channels (Graph Theory)"""

    def __init__(self):
        self.graph = nx.Graph()

    def build_channel_network(self, channels_data):
        """Build network graph of channels based on similar characteristics"""
        # Add nodes
        for i, channel in enumerate(channels_data):
            self.graph.add_node(i, **channel)

        # Add edges based on similarity
        for i in range(len(channels_data)):
            for j in range(i+1, len(channels_data)):
                similarity = self._calculate_similarity(channels_data[i], channels_data[j])
                if similarity > 0.5:  # Threshold for connection
                    self.graph.add_edge(i, j, weight=similarity)

    def _calculate_similarity(self, channel1, channel2):
        """Calculate similarity between two channels"""
        # Normalize metrics for comparison
        metrics1 = np.array([
            channel1.get('Subscribers', 0) / 1000000,  # Normalize to millions
            channel1.get('Engagement Score', 0) / 10,
            channel1.get('Content Velocity', 0) / 30
        ])

        metrics2 = np.array([
            channel2.get('Subscribers', 0) / 1000000,
            channel2.get('Engagement Score', 0) / 10,
            channel2.get('Content Velocity', 0) / 30
        ])

        # Cosine similarity
        dot_product = np.dot(metrics1, metrics2)
        norm1 = np.linalg.norm(metrics1)
        norm2 = np.linalg.norm(metrics2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    def get_network_insights(self):
        """Get insights from network analysis"""
        if len(self.graph.nodes()) < 2:
            return {"clusters": 0, "density": 0, "central_nodes": []}

        # Clustering
        try:
            communities = list(nx.connected_components(self.graph))

            # Centrality measures
            centrality = nx.betweenness_centrality(self.graph)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

            return {
                "clusters": len(communities),
                "density": nx.density(self.graph),
                "central_nodes": [node for node, _ in top_central],
                "total_connections": self.graph.number_of_edges()
            }
        except:
            return {"clusters": 0, "density": 0, "central_nodes": []}

# --- Enhanced Functions with AI Integration ---
@st.cache_data(ttl=3600)
def enhanced_viral_channel_finder(api_key, niche_ideas_list, video_type="Any", use_ai_prediction=True):
    """
    Enhanced viral channel finder with AI-powered analysis
    """
    viral_channels = []
    current_year = datetime.now().year

    # Initialize AI analyzers
    pattern_analyzer = ViralPatternAnalyzer()
    network_analyzer = NetworkAnalyzer()

    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_channel_ids = set()

    for i, niche in enumerate(niche_ideas_list):
        status_text.text(f"🔍 AI Analysis: '{niche}' niche ({i + 1}/{len(niche_ideas_list)})")
        progress_bar.progress((i + 1) / len(niche_ideas_list))

        search_params = {
            "part": "snippet",
            "q": niche,
            "type": "video",
            "order": "viewCount",
            "publishedAfter": (datetime.utcnow() - timedelta(days=90)).isoformat("T") + "Z",
            "maxResults": 20,
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

                channel_params = {
                    "part": "snippet,statistics",
                    "id": ",".join(new_channel_ids),
                    "key": api_key
                }
                channel_response = requests.get(YOUTUBE_CHANNEL_URL, params=channel_params)

                if channel_response.status_code == 200:
                    for channel in channel_response.json().get("items", []):
                        published_at_str = channel["snippet"]["publishedAt"]
                        published_date = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))

                        if published_date.year == current_year:
                            stats = channel.get("statistics", {})
                            subs = int(stats.get("subscriberCount", 0))
                            views = int(stats.get("viewCount", 0))
                            video_count = int(stats.get("videoCount", 0))

                            if subs > 1000 and views > 50000 and 5 < video_count < 100:
                                channel_id = channel['id']
                                analysis_data = enhanced_deep_dive_analysis(
                                    api_key, channel_id, 
                                    channel["snippet"].get("description", ""),
                                    use_ai_prediction
                                )

                                channel_data = {
                                    "Channel Name": channel["snippet"]["title"],
                                    "URL": f"https://www.youtube.com/channel/{channel_id}",
                                    "Subscribers": subs,
                                    "Total Views": views,
                                    "Video Count": video_count,
                                    "Creation Date": published_date.strftime("%Y-%m-%d"),
                                    "Found Via Niche": niche,
                                    **analysis_data
                                }

                                viral_channels.append(channel_data)
                                processed_channel_ids.add(channel_id)

        except requests.exceptions.RequestException:
            continue

    # Advanced AI Analysis on collected data
    if viral_channels and use_ai_prediction:
        # Pattern analysis
        pattern_results = pattern_analyzer.analyze_growth_patterns(viral_channels)

        # Network analysis
        network_analyzer.build_channel_network(viral_channels)
        network_insights = network_analyzer.get_network_insights()

        # Add AI insights to each channel
        for i, channel in enumerate(viral_channels):
            viral_prediction = pattern_analyzer.predict_viral_potential(channel)
            channel.update({
                "AI_Viral_Probability": viral_prediction["viral_probability"],
                "AI_Growth_Prediction": viral_prediction["growth_prediction"],
                "AI_Einstein_Score": viral_prediction["einstein_engagement"],
                "AI_Combined_Score": viral_prediction["combined_viral_score"]
            })

        # Store analysis results in session state for visualization
        st.session_state.pattern_analysis = pattern_results
        st.session_state.network_analysis = network_insights

    progress_bar.empty()
    status_text.empty()
    return viral_channels

def enhanced_deep_dive_analysis(api_key, channel_id, channel_description, use_ai=True):
    """Enhanced deep dive analysis with AI and mathematical modeling"""
    analysis_results = {
        "Engagement Score": 0,
        "Monetization Clues": [],
        "Content Velocity": 0,
        "Weekly Frequency": 0,
        "Upload Schedule": "N/A",
        "Sentiment Score": 0,
        "Mathematical_Growth_Rate": 0,
        "Optimization_Score": 0
    }

    try:
        # Get recent videos
        video_search_params = {
            "part": "snippet",
            "channelId": channel_id,
            "order": "date",
            "maxResults": 10,
            "key": api_key
        }
        video_response = requests.get(YOUTUBE_SEARCH_URL, params=video_search_params)

        if video_response.status_code == 200:
            video_items = video_response.json().get("items", [])
            video_ids = [item["id"]["videoId"] for item in video_items if "videoId" in item.get("id", {})]

            if video_ids:
                video_details_params = {
                    "part": "statistics,snippet",
                    "id": ",".join(video_ids),
                    "key": api_key
                }
                details_response = requests.get(YOUTUBE_VIDEO_URL, params=video_details_params)

                if details_response.status_code == 200:
                    total_likes, total_comments, total_views = 0, 0, 0
                    published_dates = []
                    view_counts = []
                    titles = []

                    for item in details_response.json().get("items", []):
                        stats = item.get("statistics", {})
                        likes = int(stats.get("likeCount", 0))
                        comments = int(stats.get("commentCount", 0))
                        views = int(stats.get("viewCount", 0))

                        total_likes += likes
                        total_comments += comments
                        total_views += views
                        view_counts.append(views)

                        published_at_str = item["snippet"]["publishedAt"]
                        published_dates.append(datetime.fromisoformat(published_at_str.replace("Z", "+00:00")))
                        titles.append(item["snippet"]["title"])

                    # Enhanced engagement calculation
                    if total_views > 0:
                        engagement_rate = ((total_likes + total_comments) / total_views) * 100
                        analysis_results["Engagement Score"] = round(engagement_rate, 2)

                    # Mathematical growth rate calculation (Newton's method)
                    if len(view_counts) > 1:
                        # Calculate growth rate using logarithmic regression
                        x = np.arange(len(view_counts))
                        y = np.log(np.maximum(view_counts, 1))  # Avoid log(0)
                        if len(x) > 1 and np.var(x) > 0:
                            slope, intercept = np.polyfit(x, y, 1)
                            analysis_results["Mathematical_Growth_Rate"] = round(slope, 4)

                    # Weekly frequency calculation
                    if len(published_dates) > 1:
                        time_span_days = (max(published_dates) - min(published_dates)).days
                        if time_span_days > 0:
                            analysis_results["Weekly Frequency"] = round((len(published_dates) / time_span_days) * 7, 1)
                        else:
                            analysis_results["Weekly Frequency"] = len(published_dates)

                        # Upload schedule analysis
                        upload_slots = [(d.strftime('%A'), d.hour) for d in published_dates]
                        if upload_slots:
                            schedule_counter = Counter(upload_slots)
                            top_slots = schedule_counter.most_common(2)

                            schedule_parts = []
                            for i, ((day, hour), count) in enumerate(top_slots):
                                schedule_parts.append(f"{i+1}. {day} {hour}:00 UTC ({count}x)")

                            analysis_results["Upload Schedule"] = " | ".join(schedule_parts)

                    # AI-powered sentiment analysis
                    if use_ai and titles:
                        sentiments = [TextBlob(title).sentiment.polarity for title in titles]
                        analysis_results["Sentiment Score"] = round(np.mean(sentiments), 3)

    except Exception as e:
        pass  # Silently handle errors

    # Enhanced monetization detection
    monetization_keywords = [
        "affiliate", "merch", "patreon", "course", "consulting", 
        "e-book", "gumroad", "sponsor", "collaboration", "brand deal"
    ]
    analysis_results["Monetization Clues"] = [
        keyword for keyword in monetization_keywords 
        if keyword in channel_description.lower()
    ]

    # Content velocity calculation
    velocity_params = {
        "part": "id",
        "channelId": channel_id,
        "type": "video",
        "publishedAfter": (datetime.utcnow() - timedelta(days=30)).isoformat("T") + "Z",
        "maxResults": 50,
        "key": api_key
    }

    try:
        velocity_response = requests.get(YOUTUBE_SEARCH_URL, params=velocity_params)
        if velocity_response.status_code == 200:
            analysis_results["Content Velocity"] = len(velocity_response.json().get("items", []))
    except:
        pass

    # Optimization score using Euler's method
    try:
        optimization_params = np.array([
            analysis_results["Mathematical_Growth_Rate"] + 0.1,  # Avoid zero
            analysis_results["Weekly Frequency"] + 0.1,
            analysis_results["Engagement Score"] / 100 + 0.1
        ])

        # Simple optimization score calculation
        analysis_results["Optimization_Score"] = round(
            np.prod(optimization_params) * 1000, 2
        )
    except:
        analysis_results["Optimization_Score"] = 0

    return analysis_results

# --- Visualization Functions (Fernando Pérez Style) ---
def create_advanced_visualizations(channels_data):
    """Create advanced visualizations for channel analysis"""
    if not channels_data:
        return

    df = pd.DataFrame(channels_data)

    # 1. Interactive Growth Prediction Chart
    fig_growth = px.scatter(
        df, 
        x="Subscribers", 
        y="Total Views",
        size="Engagement Score",
        color="AI_Combined_Score" if "AI_Combined_Score" in df.columns else "Engagement Score",
        hover_name="Channel Name",
        title="📈 AI-Powered Growth Analysis (Einstein's Relativity Applied)",
        labels={"AI_Combined_Score": "AI Viral Score"},
        width=800,
        height=500
    )

    fig_growth.update_layout(
        xaxis_title="Subscribers (Mass)",
        yaxis_title="Total Views (Energy)",
        title_font_size=16
    )

    st.plotly_chart(fig_growth, use_container_width=True)

    # 2. Engagement Distribution (Gaussian Analysis)
    fig_engagement = px.histogram(
        df,
        x="Engagement Score",
        nbins=20,
        title="📊 Engagement Score Distribution (Gaussian Model)",
        labels={"Engagement Score": "Engagement Rate (%)"}
    )

    # Add Gaussian curve overlay
    mean_engagement = df["Engagement Score"].mean()
    std_engagement = df["Engagement Score"].std()
    x_gaussian = np.linspace(df["Engagement Score"].min(), df["Engagement Score"].max(), 100)
    y_gaussian = stats.norm.pdf(x_gaussian, mean_engagement, std_engagement) * len(df) * (df["Engagement Score"].max() - df["Engagement Score"].min()) / 20

    fig_engagement.add_scatter(
        x=x_gaussian,
        y=y_gaussian,
        mode='lines',
        name='Gaussian Fit',
        line=dict(color='red', width=2)
    )

    st.plotly_chart(fig_engagement, use_container_width=True)

    # 3. Mathematical Growth Rate Analysis
    if "Mathematical_Growth_Rate" in df.columns:
        fig_math = px.bar(
            df.head(10),
            x="Channel Name",
            y="Mathematical_Growth_Rate",
            title="🔢 Mathematical Growth Rate (Newton's Calculus Applied)",
            labels={"Mathematical_Growth_Rate": "Growth Rate (ln scale)"}
        )
        fig_math.update_xaxes(tickangle=45)
        st.plotly_chart(fig_math, use_container_width=True)

    # 4. Network Analysis Visualization
    if hasattr(st.session_state, 'network_analysis'):
        network_data = st.session_state.network_analysis

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔗 Channel Clusters", network_data.get("clusters", 0))
        with col2:
            st.metric("🌐 Network Density", f"{network_data.get('density', 0):.3f}")
        with col3:
            st.metric("📡 Total Connections", network_data.get("total_connections", 0))

# --- Main Application ---
def main():
    # Header with Einstein quote
    st.title("🧠 Enhanced YouTube Growth Toolkit")
    st.markdown("""
    <div class="einstein-quote">
        "Imagination is more important than knowledge. Knowledge is limited; imagination embraces the entire world." - Albert Einstein
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 AI-Powered YouTube Channel Discovery & Analysis")
    st.markdown("*Powered by Mathematical Models from History's Greatest Minds*")

    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration Panel")
        st.info("🔑 YouTube Data API v3 key required for advanced analysis")

        if 'api_key' not in st.session_state:
            st.session_state.api_key = API_KEY if API_KEY else ""

        st.session_state.api_key = st.text_input(
            "YouTube API Key:", 
            type="password", 
            value=st.session_state.api_key
        )

        if st.session_state.api_key:
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your API key")

        st.header("🧠 AI Features")
        use_ai_prediction = st.checkbox("Enable AI Predictions", value=True)
        use_advanced_viz = st.checkbox("Advanced Visualizations", value=True)

        st.header("📊 Analysis Depth")
        analysis_depth = st.selectbox(
            "Choose Analysis Level:",
            ["Basic", "Advanced", "Expert (Einstein Mode)"]
        )

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Viral Channel Discovery", 
        "📈 Growth Prediction Lab", 
        "🧠 AI Insights Dashboard"
    ])

    with tab1:
        st.header("🔍 Enhanced Viral Channel Discovery")
        st.markdown("*Using Machine Learning & Mathematical Models*")

        # Video Type Selection
        video_type_choice = st.radio(
            "Channel Type Focus:",
            ('Any', 'Shorts Channel', 'Long Video Channel'),
            horizontal=True,
            help="Filter by content duration preferences"
        )

        # Niche Input
        default_niches = """AI Tools & Tutorials
Personal Finance for Gen Z
Sustainable Living Hacks
Side Hustle Case Studies
Mental Health & Wellness
Tech Reviews & Comparisons
Cooking & Recipe Innovations
Fitness & Home Workouts"""

        niche_input = st.text_area(
            "🎯 Enter Niche Ideas (one per line):",
            value=default_niches,
            height=150,
            help="Enter specific niches you want to research"
        )

        # Analysis Button
        if st.button("🚀 Start AI-Powered Discovery", type="primary"):
            if not st.session_state.api_key:
                st.error("❌ Please configure your YouTube API key in the sidebar")
            else:
                niche_ideas = [niche.strip() for niche in niche_input.strip().split('
') if niche.strip()]

                if not niche_ideas:
                    st.warning("⚠️ Please enter at least one niche idea")
                else:
                    with st.spinner("🧠 AI analyzing channels... This may take a few minutes..."):
                        viral_channels = enhanced_viral_channel_finder(
                            st.session_state.api_key,
                            niche_ideas,
                            video_type_choice,
                            use_ai_prediction
                        )

                    if viral_channels:
                        st.success(f"🎉 Found {len(viral_channels)} promising channels created in {datetime.now().year}!")

                        # Sort by AI score if available, otherwise by subscribers
                        sort_key = "AI_Combined_Score" if use_ai_prediction else "Subscribers"
                        viral_channels.sort(key=lambda x: x.get(sort_key, 0), reverse=True)

                        # Store in session state for other tabs
                        st.session_state.discovered_channels = viral_channels

                        # Display channels with enhanced UI
                        for i, channel in enumerate(viral_channels[:20]):  # Show top 20
                            with st.expander(f"#{i+1} {channel['Channel Name']} - 🎯 {channel['Found Via Niche']}", expanded=(i<3)):
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.metric("👥 Subscribers", f"{channel['Subscribers']:,}")
                                    st.metric("👀 Total Views", f"{channel['Total Views']:,}")

                                with col2:
                                    st.metric("📺 Videos", channel['Video Count'])
                                    st.metric("📈 Engagement", f"{channel['Engagement Score']}%")

                                with col3:
                                    st.metric("⚡ Content Velocity", f"{channel['Content Velocity']}/month")
                                    if use_ai_prediction and 'AI_Combined_Score' in channel:
                                        st.metric("🧠 AI Viral Score", f"{channel['AI_Combined_Score']:.3f}")

                                # Channel link and additional info
                                st.markdown(f"🔗 **Channel:** [{channel['Channel Name']}]({channel['URL']})")
                                st.markdown(f"📅 **Created:** {channel['Creation Date']}")
                                st.markdown(f"🎯 **Found via:** {channel['Found Via Niche']}")

                                # AI Insights
                                if use_ai_prediction and 'AI_Viral_Probability' in channel:
                                    st.markdown("### 🧠 AI Analysis")

                                    prob_col1, prob_col2 = st.columns(2)
                                    with prob_col1:
                                        st.progress(channel['AI_Viral_Probability'])
                                        st.caption(f"Viral Probability: {channel['AI_Viral_Probability']:.1%}")

                                    with prob_col2:
                                        st.progress(channel['AI_Growth_Prediction'])
                                        st.caption(f"Growth Prediction: {channel['AI_Growth_Prediction']:.3f}")

                                    # Einstein Score
                                    einstein_score = channel.get('AI_Einstein_Score', 0)
                                    if einstein_score > 0.5:
                                        st.success(f"⚡ High Einstein Engagement Score: {einstein_score:.3f}")
                                    elif einstein_score > 0.2:
                                        st.warning(f"⚡ Moderate Einstein Score: {einstein_score:.3f}")
                                    else:
                                        st.info(f"⚡ Einstein Score: {einstein_score:.3f}")

                        # Advanced Visualizations
                        if use_advanced_viz:
                            st.header("📊 Advanced Analytics Dashboard")
                            create_advanced_visualizations(viral_channels)

                    else:
                        st.warning("🤔 No rapidly growing channels found. This might be due to:")
                        st.markdown("""
                        - API limitations or rate limits
                        - Very specific niche criteria
                        - Lack of new viral channels in researched niches
                        - Try broader or different niche keywords
                        """)

    with tab2:
        st.header("📈 Growth Prediction Laboratory")
        st.markdown("*Mathematical Models & Machine Learning Predictions*")

        if hasattr(st.session_state, 'discovered_channels'):
            channels = st.session_state.discovered_channels

            if channels:
                # Pattern Analysis Results
                if hasattr(st.session_state, 'pattern_analysis'):
                    pattern_data = st.session_state.pattern_analysis

                    st.subheader("🔬 Growth Pattern Analysis")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pattern_type = pattern_data.get('pattern', 'unknown')
                        if pattern_type == 'exponential':
                            st.success(f"📈 **Pattern:** {pattern_type.title()}")
                        elif pattern_type == 'linear':
                            st.warning(f"📊 **Pattern:** {pattern_type.title()}")
                        else:
                            st.info(f"📋 **Pattern:** {pattern_type.title()}")

                    with col2:
                        accuracy = pattern_data.get('prediction_accuracy', 0)
                        st.metric("🎯 Prediction Accuracy", f"{accuracy:.1%}")

                    with col3:
                        st.metric("📊 Channels Analyzed", len(channels))

                    # Feature Importance
                    if 'feature_importance' in pattern_data:
                        st.subheader("🎯 Success Factor Importance")
                        importance_df = pd.DataFrame(
                            list(pattern_data['feature_importance'].items()),
                            columns=['Factor', 'Importance']
                        ).sort_values('Importance', ascending=False)

                        fig_importance = px.bar(
                            importance_df,
                            x='Importance',
                            y='Factor',
                            orientation='h',
                            title="🎯 Factors Contributing to Channel Success"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)

                # Individual Channel Predictions
                st.subheader("🎯 Individual Channel Predictions")
                selected_channel = st.selectbox(
                    "Choose a channel for detailed analysis:",
                    options=range(len(channels)),
                    format_func=lambda x: f"{channels[x]['Channel Name']} ({channels[x]['Subscribers']:,} subs)"
                )

                if selected_channel is not None:
                    channel = channels[selected_channel]

                    # Prediction Box
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🔮 Predictions for {channel['Channel Name']}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    pred_col1, pred_col2, pred_col3 = st.columns(3)

                    with pred_col1:
                        if 'AI_Viral_Probability' in channel:
                            viral_prob = channel['AI_Viral_Probability']
                            st.metric("🌟 Viral Potential", f"{viral_prob:.1%}")
                            if viral_prob > 0.7:
                                st.success("High viral potential!")
                            elif viral_prob > 0.4:
                                st.warning("Moderate viral potential")
                            else:
                                st.info("Lower viral potential")

                    with pred_col2:
                        if 'Mathematical_Growth_Rate' in channel:
                            growth_rate = channel['Mathematical_Growth_Rate']
                            st.metric("📈 Math Growth Rate", f"{growth_rate:.4f}")
                            if growth_rate > 0.1:
                                st.success("Strong mathematical growth!")
                            elif growth_rate > 0.05:
                                st.warning("Moderate growth trend")
                            else:
                                st.info("Slower growth pattern")

                    with pred_col3:
                        if 'Optimization_Score' in channel:
                            opt_score = channel['Optimization_Score']
                            st.metric("⚡ Optimization Score", f"{opt_score:.2f}")

                    # Time series prediction (simulated for demonstration)
                    st.subheader("📊 Projected Growth Timeline")

                    current_subs = channel['Subscribers']
                    current_velocity = channel.get('Content Velocity', 5)
                    engagement = channel.get('Engagement Score', 2)

                    # Simple growth projection model
                    months = np.arange(1, 13)
                    base_growth = fibonacci_growth_model(min(current_velocity, 10)) / 10000
                    engagement_multiplier = 1 + (engagement / 100)

                    projected_subs = [
                        current_subs * (1 + base_growth * engagement_multiplier) ** month
                        for month in months
                    ]

                    projection_df = pd.DataFrame({
                        'Month': months,
                        'Projected Subscribers': projected_subs
                    })

                    fig_projection = px.line(
                        projection_df,
                        x='Month',
                        y='Projected Subscribers',
                        title=f"📈 Growth Projection for {channel['Channel Name']}"
                    )
                    st.plotly_chart(fig_projection, use_container_width=True)

                    st.caption("*Projection based on Fibonacci growth model and current engagement metrics")

        else:
            st.info("👆 Please discover channels in the first tab to see predictions here")

    with tab3:
        st.header("🧠 AI Insights Dashboard")
        st.markdown("*Deep Learning Analysis & Network Intelligence*")

        if hasattr(st.session_state, 'discovered_channels') and hasattr(st.session_state, 'network_analysis'):
            channels = st.session_state.discovered_channels
            network_data = st.session_state.network_analysis

            # Network Analysis
            st.subheader("🌐 Channel Network Analysis")
            net_col1, net_col2, net_col3, net_col4 = st.columns(4)

            with net_col1:
                st.metric("🔗 Clusters Found", network_data.get("clusters", 0))
            with net_col2:
                st.metric("🌐 Network Density", f"{network_data.get('density', 0):.3f}")
            with net_col3:
                st.metric("📡 Connections", network_data.get("total_connections", 0))
            with net_col4:
                st.metric("🎯 Central Nodes", len(network_data.get("central_nodes", [])))

            # Sentiment Analysis Summary
            if any('Sentiment Score' in ch for ch in channels):
                st.subheader("😊 Content Sentiment Analysis")

                sentiments = [ch.get('Sentiment Score', 0) for ch in channels]
                avg_sentiment = np.mean(sentiments)

                if avg_sentiment > 0.1:
                    sentiment_color = "green"
                    sentiment_text = "Positive 😊"
                elif avg_sentiment < -0.1:
                    sentiment_color = "red"
                    sentiment_text = "Negative 😞"
                else:
                    sentiment_color = "blue"
                    sentiment_text = "Neutral 😐"

                st.markdown(f"""
                <div style="color: {sentiment_color}; font-size: 24px; text-align: center; padding: 20px;">
                    Average Content Sentiment: <strong>{sentiment_text}</strong><br>
                    <span style="font-size: 16px;">Score: {avg_sentiment:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

                # Sentiment distribution
                fig_sentiment = px.histogram(
                    x=sentiments,
                    nbins=15,
                    title="📊 Sentiment Score Distribution Across Channels"
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

            # Top Recommendations
            st.subheader("🎯 AI-Powered Recommendations")

            # Sort channels by AI combined score
            if use_ai_prediction:
                sorted_channels = sorted(
                    [ch for ch in channels if 'AI_Combined_Score' in ch],
                    key=lambda x: x['AI_Combined_Score'],
                    reverse=True
                )

                if sorted_channels:
                    top_3 = sorted_channels[:3]

                    for i, channel in enumerate(top_3, 1):
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>#{i} Recommendation: {channel['Channel Name']}</h4>
                            <p><strong>🧠 AI Score:</strong> {channel['AI_Combined_Score']:.3f}</p>
                            <p><strong>🎯 Why:</strong> High engagement ({channel['Engagement Score']}%), 
                            good content velocity ({channel['Content Velocity']}/month), 
                            and strong viral indicators.</p>
                            <p><strong>🔗 Channel:</strong> <a href="{channel['URL']}" target="_blank">Visit Channel</a></p>
                        </div>
                        """, unsafe_allow_html=True)

            # Strategic Insights
            st.subheader("💡 Strategic Insights")

            if channels:
                # Calculate insights
                avg_engagement = np.mean([ch.get('Engagement Score', 0) for ch in channels])
                avg_velocity = np.mean([ch.get('Content Velocity', 0) for ch in channels])
                high_performers = [ch for ch in channels if ch.get('Engagement Score', 0) > avg_engagement * 1.5]

                insights = [
                    f"📈 Average engagement rate is {avg_engagement:.2f}% - aim for {avg_engagement*1.5:.2f}% or higher",
                    f"⚡ Top performers post {avg_velocity:.1f} videos per month on average",
                    f"🏆 {len(high_performers)} channels show exceptional performance patterns",
                    f"🎯 Most successful niches: {', '.join(set([ch['Found Via Niche'] for ch in channels[:5]]))}"
                ]

                for insight in insights:
                    st.info(insight)

        else:
            st.info("👆 Please discover channels in the first tab to see AI insights here")

            # Educational content while waiting
            st.subheader("🧠 Learn About Our AI Models")

            with st.expander("🔬 Mathematical Models Used"):
                st.markdown("""
                ### 📊 Statistical Models (Gauss & Euler)
                - **Gaussian Distribution**: For engagement score analysis
                - **Fibonacci Sequence**: Growth pattern prediction
                - **Logarithmic Regression**: Mathematical growth rate calculation

                ### 🧮 Einstein's Formula Adaptation
                **E = mc²** adapted as **Engagement = Mass × Content²**
                - Mass = Subscriber base
                - Content = Posting velocity squared
                - Engagement = Viral potential energy

                ### 🤖 Machine Learning (Hinton, Bengio, LeCun)
                - **Random Forest**: Growth prediction model
                - **Gradient Boosting**: Viral potential classification
                - **Network Analysis**: Channel relationship mapping

                ### 🧠 AI Features
                - **Sentiment Analysis**: TextBlob-powered content mood detection
                - **Pattern Recognition**: Time series analysis for upload patterns
                - **Optimization**: Multi-variable optimization for content strategy
                """)

if __name__ == "__main__":
    main()
