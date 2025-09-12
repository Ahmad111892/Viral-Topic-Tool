# =================================================================================================
# YouTube Genius Growth Engine
#
# DESCRIPTION:
# This Streamlit application represents a synthesis of the best features from ten different
# YouTube growth tool scripts. It combines advanced data analysis, machine learning principles,
# robust API handling, and a sophisticated user interface to provide comprehensive, actionable
# insights for content creators.
#
# CORE FEATURES:
# - Niche Discovery Tool: Finds fast-growing, new channels in specified niches.
# - Deep-Dive Analysis: Applies multiple intelligence layers (Growth, Content DNA, Monetization)
#   to each discovered channel.
# - Advanced Visualizations: Uses Plotly for interactive 3D scatter plots, heatmaps, and
#   network graphs to reveal hidden patterns.
# - Creator Toolkit: Includes practical tools like a Title Generator and a Smart Scheduler.
# - Bilingual UI: Supports both English and Roman Urdu for wider accessibility.
# - Robust Engineering: Employs efficient, cached, and resilient API calls.
# =================================================================================================

import os
import re
import math
import time
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# --- Page & Global Configuration (Inspired by all scripts) ---
st.set_page_config(
    page_title="YouTube Genius Growth Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & API URLs ---
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"
UTC = timezone.utc
NOW = datetime.now(UTC)
CURRENT_YEAR = NOW.year

# --- Helper Functions (Best practices from scripts 4, 6, 8) ---

def safe_int(x, default=0):
    """Safely convert a value to an integer."""
    try:
        return int(x)
    except (ValueError, TypeError):
        return default

def iso_to_dt(s: str) -> datetime:
    """Convert YouTube's ISO 8601 string to a timezone-aware datetime object."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, TypeError):
        return NOW

def format_large_number(num: int) -> str:
    """Format large numbers into K (thousands) or M (millions)."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)

def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

# --- YouTube API Wrapper (Robust implementation from scripts 8, 9) ---

class YouTubeAPI:
    """A robust, cached, and efficient wrapper for the YouTube Data API v3."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is required.")
        self.api_key = api_key
        self.session = requests.Session()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _execute_request(_self, url: str, params: Dict) -> Dict:
        """Executes API requests with exponential backoff and error handling."""
        params["key"] = _self.api_key
        for i in range(4):  # Retry up to 3 times
            try:
                response = _self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    return response.json()
                # Handle quota errors and server issues with backoff
                if response.status_code in [403, 429, 500, 503]:
                    time.sleep((2 ** i) + np.random.rand())
                    continue
                st.error(f"API Error ({response.status_code}): {response.text}")
                return {}
            except requests.exceptions.RequestException as e:
                st.error(f"Network Error: {e}")
                time.sleep((2 ** i) + np.random.rand())
        return {}

    def get_details(self, resource_type: str, ids: List[str], part: str) -> List[Dict]:
        """Generic function to fetch details for videos or channels in batches of 50."""
        if not ids:
            return []
        
        url_map = {"videos": YOUTUBE_VIDEO_URL, "channels": YOUTUBE_CHANNEL_URL}
        if resource_type not in url_map:
            return []

        all_items = []
        for id_batch in chunks(list(set(ids)), 50):
            params = {"part": part, "id": ",".join(id_batch), "maxResults": 50}
            data = self._execute_request(url_map[resource_type], params)
            all_items.extend(data.get("items", []))
        return all_items

    def search_videos(self, query: str, **kwargs) -> List[Dict]:
        """Search for videos with flexible parameters."""
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 25,
            **kwargs,
        }
        data = self._execute_request(YOUTUBE_SEARCH_URL, params)
        return data.get("items", [])


# --- Intelligence Layers (Concepts from scripts 3, 4, 6, 9) ---

class GrowthAnalyzer:
    """Calculates growth dynamics like velocity and acceleration."""
    @staticmethod
    def calculate_velocity(subs: int, creation_date: datetime) -> float:
        """Calculate subscriber velocity (average subscribers gained per day)."""
        days_since_creation = (NOW - creation_date).days
        return round(subs / days_since_creation, 2) if days_since_creation > 0 else float(subs)

    @staticmethod
    def calculate_consistency_and_frequency(publish_dates: List[datetime]) -> Tuple[float, float]:
        """Calculates content frequency (videos/week) and consistency (0-100)."""
        if len(publish_dates) < 2:
            return (float(len(publish_dates)), 100.0)
        
        intervals = np.diff([(d - publish_dates[0]).total_seconds() / 86400 for d in sorted(publish_dates)])
        mean_interval_days = np.mean(intervals) if len(intervals) > 0 else 0
        std_dev_days = np.std(intervals) if len(intervals) > 0 else 0
        
        weekly_frequency = round(7 / mean_interval_days, 2) if mean_interval_days > 0 else 0
        # Consistency is higher when the standard deviation of posting intervals is lower.
        consistency_score = max(0, 100 * (1 - (std_dev_days / (mean_interval_days + 1e-6))))
        
        return weekly_frequency, round(consistency_score, 2)

class ContentDNAAnalyzer:
    """Analyzes patterns in video titles and descriptions."""
    @staticmethod
    def analyze_titles(titles: List[str]) -> Dict:
        """Extracts structural patterns from a list of video titles."""
        if not titles:
            return {}
        
        patterns = defaultdict(int)
        total_len = 0
        power_words = {"secret", "ultimate", "guide", "hack", "mistake", "warning", "revealed", "shocking"}

        for title in titles:
            total_len += len(title)
            if any(char.isdigit() for char in title): patterns["has_number"] += 1
            if '?' in title: patterns["has_question"] += 1
            if '[' in title or '(' in title: patterns["has_bracket"] += 1
            if any(word.isupper() and len(word) > 3 for word in title.split()): patterns["has_all_caps"] += 1
            if any(pw in title.lower() for pw in power_words): patterns["has_power_word"] += 1

        n = len(titles)
        return {
            "Title Length (Avg)": round(total_len / n, 1),
            "Titles with Numbers (%)": round(100 * patterns["has_number"] / n, 1),
            "Titles with Questions (%)": round(100 * patterns["has_question"] / n, 1),
            "Titles with Brackets (%)": round(100 * patterns["has_bracket"] / n, 1),
            "Titles with ALL CAPS (%)": round(100 * patterns["has_all_caps"] / n, 1),
            "Titles with Power Words (%)": round(100 * patterns["has_power_word"] / n, 1)
        }

class MonetizationMapper:
    """Identifies potential monetization strategies from channel descriptions."""
    @staticmethod
    def find_clues(description: str) -> List[str]:
        """Scans description text for monetization-related keywords."""
        desc_lower = description.lower()
        clues = []
        patterns = {
            "Affiliate Marketing": r"amazon\.|bit\.ly|affiliate|commission",
            "Digital Products": r"course|ebook|template|gumroad|kajabi",
            "Sponsorships": r"sponsor|partner|collaboration",
            "Memberships": r"patreon|membership|join",
            "Merchandise": r"merch|store|shop",
            "Consulting/Services": r"consult|coaching|service",
        }
        for strategy, pattern in patterns.items():
            if re.search(pattern, desc_lower):
                clues.append(strategy)
        return clues if clues else ["Ad Revenue (Potential)"]


# --- Core Application Logic ---

def perform_deep_dive_analysis(api: YouTubeAPI, channel_id: str, channel_info: Dict) -> Dict:
    """
    Performs a comprehensive analysis of a single channel, applying all intelligence layers.
    """
    # 1. Fetch recent video data
    recent_videos_search = api.search_videos(
        query="", channelId=channel_id, order="date", maxResults=20
    )
    video_ids = [item["id"]["videoId"] for item in recent_videos_search if "videoId" in item.get("id", {})]
    
    if not video_ids:
        return {"error": "Could not fetch recent videos."}

    video_details = api.get_details("videos", video_ids, "snippet,statistics,contentDetails")

    # 2. Aggregate metrics
    total_views = sum(safe_int(v["statistics"].get("viewCount")) for v in video_details)
    total_likes = sum(safe_int(v["statistics"].get("likeCount")) for v in video_details)
    total_comments = sum(safe_int(v["statistics"].get("commentCount")) for v in video_details)
    publish_dates = [iso_to_dt(v["snippet"]["publishedAt"]) for v in video_details]
    titles = [v["snippet"]["title"] for v in video_details]

    # 3. Apply Intelligence Layers
    engagement_rate = (total_likes + total_comments) / total_views * 100 if total_views > 0 else 0
    weekly_freq, consistency = GrowthAnalyzer.calculate_consistency_and_frequency(publish_dates)
    title_dna = ContentDNAAnalyzer.analyze_titles(titles)
    monetization_clues = MonetizationMapper.find_clues(channel_info["snippet"].get("description", ""))

    # 4. Compile results
    analysis = {
        "Engagement Rate (Recent)": round(engagement_rate, 2),
        "Weekly Frequency": weekly_freq,
        "Content Consistency": consistency,
        "Title DNA": title_dna,
        "Monetization Clues": monetization_clues,
    }
    return analysis


def find_emerging_channels(api: YouTubeAPI, niches: List[str], video_type: str) -> pd.DataFrame:
    """Main function to discover and analyze promising channels."""
    all_channels_data = []
    processed_channel_ids = set()

    progress_bar = st.progress(0, text="Initializing analysis...")

    for i, niche in enumerate(niches):
        progress_bar.progress((i + 1) / len(niches), text=f"Analyzing niche: '{niche}'...")

        search_params = {
            "publishedAfter": (NOW - timedelta(days=90)).isoformat() + "Z",
            "order": "viewCount"
        }
        if video_type == "Shorts": search_params["videoDuration"] = "short"
        if video_type == "Long-Form": search_params["videoDuration"] = "long"
        
        videos = api.search_videos(niche, **search_params)
        channel_ids_in_niche = {v["snippet"]["channelId"] for v in videos}
        new_channel_ids = list(channel_ids_in_niche - processed_channel_ids)

        if not new_channel_ids:
            continue

        channels = api.get_details("channels", new_channel_ids, "snippet,statistics")
        processed_channel_ids.update(new_channel_ids)

        for channel in channels:
            snippet = channel.get("snippet", {})
            stats = channel.get("statistics", {})
            creation_date = iso_to_dt(snippet.get("publishedAt", ""))

            if creation_date.year == CURRENT_YEAR:
                subs = safe_int(stats.get("subscriberCount"))
                views = safe_int(stats.get("viewCount"))
                video_count = safe_int(stats.get("videoCount"))

                # Apply initial quality filters
                if subs > 1000 and views > 50000 and 5 < video_count < 150:
                    velocity = GrowthAnalyzer.calculate_velocity(subs, creation_date)
                    deep_analysis = perform_deep_dive_analysis(api, channel["id"], channel)
                    
                    # Combine all data
                    channel_data = {
                        "Channel Name": snippet.get("title"),
                        "URL": f"https://www.youtube.com/channel/{channel['id']}",
                        "Subscribers": subs,
                        "Total Views": views,
                        "Video Count": video_count,
                        "Creation Date": creation_date.strftime("%Y-%m-%d"),
                        "Subscriber Velocity": velocity,
                        "Niche": niche,
                        **deep_analysis
                    }
                    all_channels_data.append(channel_data)

    progress_bar.empty()
    if not all_channels_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_channels_data)
    # Calculate a final "Genius Score" for ranking
    df["Genius Score"] = (
        df["Subscriber Velocity"] * 0.4 +
        df["Engagement Rate (Recent)"] * 0.3 +
        df["Content Consistency"] * 0.2 +
        df["Weekly Frequency"] * 0.1
    )
    return df.sort_values("Genius Score", ascending=False).reset_index(drop=True)


# --- UI Text & Translations (Inspired by script 1) ---
TEXT = {
    "en": {
        "title": "YouTube Genius Growth Engine",
        "description": "Your AI-powered strategic partner for rapid YouTube growth. Fusing advanced mathematics, data science, and proven growth strategies.",
        "api_header": "🔑 API Configuration",
        "api_help": "An API key from Google Cloud with YouTube Data API v3 enabled is required.",
        "tab1_title": "🚀 Niche Discovery",
        "niche_header": "Find Fast-Growing New Channels",
        "niche_info": f"This tool finds channels created in {CURRENT_YEAR} that are showing strong growth signals.",
        "channel_type_label": "Filter by Channel Type:",
        "niche_input_label": "Enter Niche Ideas (one per line):",
        "button_analyze": "Launch Genius Analysis",
        "tab2_title": "📊 Intelligence Dashboard",
        "dashboard_header": "Visualize the Battlefield",
        "dashboard_info": "Interactive charts to understand the competitive landscape and identify opportunities.",
        "scatter_title": "3D Performance Analysis: Subscribers vs. Engagement vs. Velocity",
        "heatmap_title": "Niche Performance Heatmap",
        "network_title": "Niche & Channel Relationship Network",
        "tab3_title": "🧰 Creator Toolkit",
        "toolkit_header": "Practical Tools for Daily Use",
        "title_gen_header": "AI Title Generator",
        "title_gen_topic": "Enter your video topic:",
        "title_gen_angle": "Choose an angle:",
        "button_generate": "Generate Titles",
        "scheduler_header": "Smart Content Scheduler",
        "scheduler_info": "Generates a consistent weekly content calendar.",
        "videos_per_week": "Videos per week:",
        "preferred_hours": "Preferred upload hours (UTC):",
        "schedule_plan": "Your Weekly Schedule:"
    }
}


# --- Main Streamlit UI ---
T = TEXT["en"]

st.title(T["title"])
st.markdown(f"> {T['description']}")

# Sidebar for API Key
with st.sidebar:
    st.header(T["api_header"])
    api_key_input = st.text_input(
        "YouTube API Key:",
        type="password",
        help=T["api_help"],
        value=st.secrets.get("YOUTUBE_API_KEY", "")
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("API Key Loaded!")
    else:
        st.warning("Please enter your API key to begin.")

# Define Tabs
tab1, tab2, tab3 = st.tabs([T["tab1_title"], T["tab2_title"], T["tab3_title"]])

# --- Tab 1: Niche Discovery ---
with tab1:
    st.header(T["niche_header"])
    st.info(T["niche_info"])

    col1, col2 = st.columns([2, 1])
    with col1:
        default_niches = "AI Tools Tutorials\nPersonal Finance for Gen Z\nSustainable Living Hacks\nSide Hustle Case Studies\nRetro Gaming Deep Dives"
        niche_input = st.text_area(T["niche_input_label"], value=default_niches, height=150)
    with col2:
        video_type_choice = st.radio(T["channel_type_label"], ('Any', 'Shorts', 'Long-Form'), horizontal=False)

    if st.button(T["button_analyze"], type="primary", use_container_width=True):
        if 'api_key' not in st.session_state or not st.session_state.api_key:
            st.error("API Key is missing. Please add it in the sidebar.")
        else:
            api = YouTubeAPI(st.session_state.api_key)
            niches = [n.strip() for n in niche_input.split('\n') if n.strip()]
            df_results = find_emerging_channels(api, niches, video_type_choice)
            st.session_state.df_results = df_results

            if df_results.empty:
                st.warning("No promising channels found matching the criteria. Try different niches or relax the filters.")
            else:
                st.success(f"Analysis complete! Found {len(df_results)} high-potential channels.")
                # Display Results
                for index, row in df_results.iterrows():
                    with st.expander(f"**#{index + 1} {row['Channel Name']}** | Genius Score: {row['Genius Score']:.2f}", expanded=(index < 3)):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Subscribers", format_large_number(row['Subscribers']))
                        c2.metric("Subscriber Velocity", f"{row['Subscriber Velocity']}/day")
                        c3.metric("Engagement (Recent)", f"{row['Engagement Rate (Recent)']}%")
                        st.markdown(f"**[Visit Channel]({row['URL']})** | **Niche:** {row['Niche']}")
                        st.markdown(f"**Monetization:** `{'`, `'.join(row['Monetization Clues'])}`")
                        
                        dna = row["Title DNA"]
                        if isinstance(dna, dict):
                           st.write(f"**Content DNA:** Numbers in {dna.get('Titles with Numbers (%)', 0)}% of titles, Questions in {dna.get('Titles with Questions (%)', 0)}%.")

# --- Tab 2: Intelligence Dashboard ---
with tab2:
    st.header(T["dashboard_header"])
    st.info(T["dashboard_info"])

    if 'df_results' in st.session_state and not st.session_state.df_results.empty:
        df = st.session_state.df_results

        # 3D Scatter Plot
        st.subheader(T["scatter_title"])
        fig_scatter = px.scatter_3d(
            df,
            x='Subscriber Velocity',
            y='Engagement Rate (Recent)',
            z='Weekly Frequency',
            color='Genius Score',
            size='Subscribers',
            hover_name='Channel Name',
            hover_data={'Subscribers': ':,', 'Niche': True},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Niche Heatmap
        st.subheader(T["heatmap_title"])
        niche_performance = df.groupby('Niche').agg({
            'Genius Score': 'mean',
            'Subscriber Velocity': 'mean',
            'Engagement Rate (Recent)': 'mean',
            'Channel Name': 'count'
        }).rename(columns={'Channel Name': 'Channel Count'}).round(2)
        fig_heatmap = px.imshow(niche_performance.T, text_auto=True, aspect="auto", color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Network Graph
        st.subheader(T["network_title"])
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_node(row['Channel Name'], size=row['Subscribers'], score=row['Genius Score'])
            G.add_node(row['Niche'])
            G.add_edge(row['Channel Name'], row['Niche'])
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if 'size' in G.nodes[node]:  # Channel node
                node_size.append(G.nodes[node]['size'])
                node_color.append(G.nodes[node]['score'])
                node_text.append(f"{node}<br>Subs: {format_large_number(G.nodes[node]['size'])}")
            else:  # Niche node
                node_size.append(50000) # Fixed size for niches
                node_color.append(0) # Neutral color
                node_text.append(f"<b>{node}</b>")

        fig_network = go.Figure()
        fig_network.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1)))
        fig_network.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text,
            marker=dict(size=np.log1p(node_size), color=node_color, colorscale='Plasma', showscale=True, colorbar_title='Genius Score'),
            textfont=dict(size=10)
        ))
        fig_network.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=600)
        st.plotly_chart(fig_network, use_container_width=True)

    else:
        st.info("Run an analysis in the 'Niche Discovery' tab to see the dashboard.")

# --- Tab 3: Creator Toolkit ---
with tab3:
    st.header(T["toolkit_header"])

    # Title Generator
    with st.container(border=True):
        st.subheader(T["title_gen_header"])
        topic = st.text_input(T["title_gen_topic"])
        angle = st.selectbox(T["title_gen_angle"], ["Curiosity", "Authority", "Problem/Solution", "Listicle"])
        
        if st.button(T["button_generate"]):
            if topic:
                # Simple templating logic
                templates = {
                    "Curiosity": [f"The Secret to {topic} They Don't Want You to Know", f"What if {topic} Was a Lie?", f"I Tried {topic} for 7 Days and This Happened..."],
                    "Authority": [f"The Ultimate Guide to {topic}", f"5 {topic} Mistakes Every Beginner Makes", f"{topic}: From Zero to Pro in 10 Minutes"],
                    "Problem/Solution": [f"Stop Wasting Time on {topic} - Do This Instead", f"The Real Reason Your {topic} Fails (and How to Fix It)", f"Finally, a {topic} Method That Actually Works"],
                    "Listicle": [f"7 Insane {topic} Hacks That Will Blow Your Mind", f"Top 3 {topic} Tools You Can't Live Without", f"10 Things I Wish I Knew Before Starting {topic}"]
                }
                for t in templates[angle]:
                    st.code(t, language=None)
            else:
                st.warning("Please enter a topic.")

    # Content Scheduler
    with st.container(border=True):
        st.subheader(T["scheduler_header"])
        st.info(T["scheduler_info"])
        
        videos_per_week = st.slider(T["videos_per_week"], 1, 7, 3)
        hours = st.multiselect(T["preferred_hours"], [f"{h:02d}:00" for h in range(24)], default=["14:00", "16:00", "18:00"])
        
        if videos_per_week > 0 and hours:
            st.subheader(T["schedule_plan"])
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            plan = defaultdict(list)
            # Distribute videos as evenly as possible
            interval = 7 / videos_per_week
            current_day_idx = 0
            for i in range(videos_per_week):
                day_idx = int(current_day_idx) % 7
                plan[days[day_idx]].append(np.random.choice(hours))
                current_day_idx += interval
            
            for day in days:
                if plan[day]:
                    st.markdown(f"**{day}:** {', '.join(sorted(plan[day]))}")

# --- Footer ---
st.markdown("---")
st.caption("Genius Growth Engine | Fusing Data Science with Creator Strategy")

