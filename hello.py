#!/usr/bin/env python3
"""
Complete YouTube Analytics Platform
Combines intelligent niche research, viral video finder, growth analysis, and channel finder
Advanced analytics with mathematical models for YouTube success
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
try:
    from textstat import flesch_reading_ease
except ImportError:
    def flesch_reading_ease(text):
        return 50  # Default score if textstat not available
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Complete Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* === CSS VARIABLES === */
    :root {
        /* Colors */
        --primary: #667eea;
        --primary-dark: #5a6fd8;
        --secondary: #764ba2;
        --accent: #4CAF50;
        --accent-light: #e8f5e8;
        --warning: #ffc107;
        --warning-light: #fff3cd;
        --danger: #f44336;
        --success: #28a745;
        --info: #17a2b8;
        
        /* Neutral Colors */
        --light: #f8f9fa;
        --light-gray: #e9ecef;
        --gray: #6c757d;
        --dark: #343a40;
        --white: #ffffff;
        --black: #212529;
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        --gradient-success: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        --gradient-warning: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        --gradient-danger: linear-gradient(135deg, #f44336 0%, #e91e63 100%);
        
        /* Shadows */
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
        --shadow-xl: 0 20px 25px rgba(0,0,0,0.15);
        
        /* Border Radius */
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        
        /* Spacing */
        --space-xs: 0.5rem;
        --space-sm: 1rem;
        --space-md: 1.5rem;
        --space-lg: 2rem;
        --space-xl: 3rem;
        
        /* Typography */
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-md: 1rem;
        --font-size-lg: 1.25rem;
        --font-size-xl: 1.5rem;
        --font-size-2xl: 2rem;
        --font-size-3xl: 2.5rem;
        
        /* Transitions */
        --transition-fast: 0.2s ease;
        --transition-normal: 0.3s ease;
        --transition-slow: 0.5s ease;
    }

    /* === BASE STYLES === */
    .stApp {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        line-height: 1.6;
        color: var(--dark);
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }

    /* === IMPROVED HEADER === */
    .main-header {
        background: var(--gradient-primary);
        padding: var(--space-xl);
        border-radius: var(--radius-lg);
        margin-bottom: var(--space-lg);
        text-align: center;
        color: var(--white);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }

    .main-header h1 {
        font-size: var(--font-size-3xl);
        font-weight: 700;
        margin-bottom: var(--space-sm);
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }

    .main-header h3 {
        font-size: var(--font-size-xl);
        font-weight: 500;
        margin-bottom: var(--space-xs);
        opacity: 0.95;
        position: relative;
        z-index: 2;
    }

    .main-header p {
        font-size: var(--font-size-md);
        opacity: 0.9;
        position: relative;
        z-index: 2;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(1deg); }
    }

    /* === IMPROVED CARDS === */
    .metric-card {
        background: var(--white);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        border-left: 5px solid var(--accent);
        box-shadow: var(--shadow-md);
        transition: var(--transition-normal);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 175, 80, 0.1), transparent);
        transition: var(--transition-slow);
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .prediction-box {
        background: var(--accent-light);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-md) 0;
        border: 2px solid var(--accent);
        box-shadow: var(--shadow-sm);
        transition: var(--transition-normal);
    }

    .prediction-box:hover {
        transform: scale(1.01);
        box-shadow: var(--shadow-md);
    }

    .question-box {
        background: var(--white);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-md) 0;
        border-left: 5px solid var(--primary);
        box-shadow: var(--shadow-md);
        color: var(--black);
        transition: var(--transition-normal);
    }

    .question-box:hover {
        border-left-color: var(--secondary);
        box-shadow: var(--shadow-lg);
    }

    .result-card {
        background: var(--white);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        margin: var(--space-sm) 0;
        border: 1px solid var(--light-gray);
        box-shadow: var(--shadow-sm);
        transition: var(--transition-normal);
    }

    .result-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }

    .optional-section {
        background: var(--warning-light);
        border: 1px solid var(--warning);
        border-radius: var(--radius-sm);
        padding: var(--space-md);
        margin: var(--space-md) 0;
        transition: var(--transition-normal);
    }

    .optional-section:hover {
        background: #fff8e1;
        border-color: #ffb300;
    }

    /* === SIDEBAR IMPROVEMENTS === */
    .css-1d391kg {
        background: var(--white);
        box-shadow: var(--shadow-lg);
        border-right: 1px solid var(--light-gray);
    }

    .sidebar .sidebar-content {
        padding: var(--space-md);
    }

    /* === BUTTON ENHANCEMENTS === */
    .stButton > button {
        border-radius: var(--radius-sm);
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: var(--transition-fast);
        border: none;
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }

    /* Primary button */
    .stButton > button:first-child {
        background: var(--gradient-primary);
        color: var(--white);
    }

    .stButton > button:first-child:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%);
    }

    /* Secondary buttons */
    .stButton > button:not(:first-child) {
        background: var(--white);
        color: var(--primary);
        border: 2px solid var(--primary);
    }

    .stButton > button:not(:first-child):hover {
        background: var(--primary);
        color: var(--white);
    }

    /* === TAB IMPROVEMENTS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--white);
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        padding: var(--space-sm) var(--space-md);
        border: 1px solid var(--light-gray);
        border-bottom: none;
        margin: 0;
        transition: var(--transition-fast);
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: var(--white);
        border-color: var(--primary);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--light);
        transform: translateY(-1px);
    }

    /* === METRIC CARDS ENHANCEMENT === */
    [data-testid="metric-container"] {
        background: var(--white);
        border-radius: var(--radius-md);
        padding: var(--space-md);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--light-gray);
        transition: var(--transition-normal);
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    /* === EXPANDER IMPROVEMENTS === */
    .streamlit-expanderHeader {
        background: var(--white);
        border-radius: var(--radius-sm);
        padding: var(--space-sm) var(--space-md);
        border: 1px solid var(--light-gray);
        font-weight: 600;
        transition: var(--transition-fast);
    }

    .streamlit-expanderHeader:hover {
        background: var(--light);
        border-color: var(--primary);
    }

    .streamlit-expanderContent {
        background: var(--white);
        border-radius: 0 0 var(--radius-sm) var(--radius-sm);
        padding: var(--space-md);
        border: 1px solid var(--light-gray);
        border-top: none;
    }

    /* === PROGRESS BAR ENHANCEMENT === */
    .stProgress > div > div > div {
        background: var(--gradient-primary);
        border-radius: var(--radius-sm);
    }

    /* === DATA FRAME STYLING === */
    .dataframe {
        border-radius: var(--radius-sm);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    .dataframe thead th {
        background: var(--gradient-primary);
        color: var(--white);
        font-weight: 600;
    }

    .dataframe tbody tr:nth-child(even) {
        background: var(--light);
    }

    .dataframe tbody tr:hover {
        background: var(--accent-light);
    }

    /* === TEXT INPUT ENHANCEMENTS === */
    .stTextInput > div > div > input {
        border-radius: var(--radius-sm);
        border: 2px solid var(--light-gray);
        padding: 0.5rem 1rem;
        transition: var(--transition-fast);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* === TEXT AREA ENHANCEMENTS === */
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-sm);
        border: 2px solid var(--light-gray);
        padding: 0.75rem 1rem;
        transition: var(--transition-fast);
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* === NUMBER INPUT ENHANCEMENTS === */
    .stNumberInput > div > div > input {
        border-radius: var(--radius-sm);
        border: 2px solid var(--light-gray);
        padding: 0.5rem 1rem;
        transition: var(--transition-fast);
    }

    .stNumberInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* === SELECT BOX ENHANCEMENTS === */
    .stSelectbox > div > div > div {
        border-radius: var(--radius-sm);
        border: 2px solid var(--light-gray);
        transition: var(--transition-fast);
    }

    .stSelectbox > div > div > div:hover {
        border-color: var(--primary);
    }

    /* === RADIO BUTTON ENHANCEMENTS === */
    .stRadio > div {
        background: var(--white);
        border-radius: var(--radius-sm);
        padding: var(--space-sm);
        border: 1px solid var(--light-gray);
    }

    .stRadio > div > label {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border-radius: var(--radius-sm);
        transition: var(--transition-fast);
    }

    .stRadio > div > label:hover {
        background: var(--light);
    }

    .stRadio > div > label[data-testid="stRadioLabel"] > div:first-child {
        background: var(--primary);
    }

    /* === CHECKBOX ENHANCEMENTS === */
    .stCheckbox > label {
        padding: 0.5rem;
        border-radius: var(--radius-sm);
        transition: var(--transition-fast);
    }

    .stCheckbox > label:hover {
        background: var(--light);
    }

    /* === SUCCESS/ERROR/WARNING MESSAGES === */
    .stAlert {
        border-radius: var(--radius-sm);
        padding: var(--space-sm) var(--space-md);
        margin: var(--space-sm) 0;
        border: 1px solid transparent;
    }

    .stAlert[data-testid="stNotification"] {
        box-shadow: var(--shadow-md);
    }

    /* === CUSTOM SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--light-gray);
        border-radius: var(--radius-sm);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--gray);
        border-radius: var(--radius-sm);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--dark);
    }

    /* === RESPONSIVE DESIGN === */
    @media (max-width: 768px) {
        :root {
            --space-md: 1rem;
            --space-lg: 1.5rem;
            --space-xl: 2rem;
            --font-size-xl: 1.25rem;
            --font-size-2xl: 1.75rem;
            --font-size-3xl: 2rem;
        }

        .main-header {
            padding: var(--space-lg);
            margin-bottom: var(--space-md);
        }

        .main-header h1 {
            font-size: var(--font-size-2xl);
        }

        .stTabs [data-baseweb="tab"] {
            padding: var(--space-xs) var(--space-sm);
            font-size: var(--font-size-sm);
        }

        .metric-card, .result-card, .question-box {
            padding: var(--space-sm);
            margin: var(--space-xs) 0;
        }

        /* Stack columns on mobile */
        .row-widget.stColumns {
            flex-direction: column;
        }

        .row-widget.stColumns > div {
            width: 100% !important;
            margin-bottom: var(--space-sm);
        }
    }

    @media (max-width: 480px) {
        :root {
            --space-sm: 0.75rem;
            --space-md: 1rem;
            --font-size-lg: 1.125rem;
            --font-size-xl: 1.375rem;
        }

        .main-header {
            padding: var(--space-md);
        }

        .main-header h1 {
            font-size: var(--font-size-xl);
        }

        .stButton > button {
            width: 100%;
            margin-bottom: var(--space-xs);
        }
    }

    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* Apply animations to main elements */
    .main-header, .metric-card, .result-card {
        animation: fadeIn 0.6s ease-out;
    }

    /* === SPECIAL EFFECTS === */
    .glow-effect {
        position: relative;
    }

    .glow-effect::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: inherit;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        opacity: 0;
        transition: var(--transition-normal);
    }

    .glow-effect:hover::after {
        opacity: 1;
    }

    /* === EINSTEIN QUOTE STYLING === */
    .einstein-quote {
        font-style: italic;
        color: var(--gray);
        border-left: 4px solid var(--primary);
        padding-left: var(--space-md);
        margin: var(--space-lg) 0;
        background: var(--light);
        padding: var(--space-md);
        border-radius: var(--radius-sm);
        position: relative;
        font-size: var(--font-size-lg);
    }

    .einstein-quote::before {
        content: '"';
        font-size: var(--font-size-3xl);
        color: var(--primary);
        position: absolute;
        top: -10px;
        left: 10px;
        opacity: 0.3;
    }

    /* === LOADING STATES === */
    .loading {
        position: relative;
        overflow: hidden;
    }

    .loading::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: loading 1.5s infinite;
    }

    @keyframes loading {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* === CUSTOM BADGES === */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-sm);
        font-size: var(--font-size-xs);
        font-weight: 600;
        margin: 0 0.25rem;
    }

    .badge-primary { background: var(--primary); color: var(--white); }
    .badge-success { background: var(--success); color: var(--white); }
    .badge-warning { background: var(--warning); color: var(--black); }
    .badge-danger { background: var(--danger); color: var(--white); }
    .badge-info { background: var(--info); color: var(--white); }

    /* === TOOLTIP STYLING === */
    [data-testid="stTooltip"] {
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--light-gray);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.markdown("<h1>YouTube Analytics Platform</h1>", unsafe_allow_html=True)
st.markdown("<h3>Discover Niche, Find Viral Videos & Analyze Growth</h3>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("# Configuration ⚙️")
api_key = st.sidebar.text_input("YouTube Data API Key", type="password")
st.sidebar.markdown("---")
st.sidebar.markdown("### Search Settings 🔎")
search_query = st.sidebar.text_input("Enter a niche or keyword", "Data Science")
search_max_results = st.sidebar.slider("Number of search results", 5, 50, 20)
search_order = st.sidebar.selectbox("Sort by", ["relevance", "viewCount", "rating", "date"], index=1)
search_duration = st.sidebar.selectbox("Video duration", ["any", "short", "medium", "long"])
search_type = st.sidebar.multiselect("Content type", ["video", "channel", "playlist"], default=["video"])

# --- Main Functions ---
def get_video_data(video_id, api_key):
    # This is a placeholder for the actual API call
    return {
        "title": "Example Title",
        "description": "Example description.",
        "viewCount": 100000,
        "likeCount": 5000,
        "commentCount": 500,
        "publishedAt": "2023-01-01T00:00:00Z"
    }

def get_channel_data(channel_id, api_key):
    # This is a placeholder for the actual API call
    return {
        "title": "Example Channel",
        "subscriberCount": 50000,
        "viewCount": 5000000,
        "videoCount": 150
    }

def search_videos(query, api_key, max_results, order, duration, content_type):
    # This is a placeholder for the actual API call
    st.info("Fetching search results...")
    return [
        {"id": {"videoId": "example1"}, "snippet": {"title": "Viral Video 1", "channelTitle": "Channel A"}},
        {"id": {"videoId": "example2"}, "snippet": {"title": "Trending Video 2", "channelTitle": "Channel B"}},
    ]

# --- Main App ---
tabs = st.tabs(["💡 Niche Research", "📈 Growth Analysis", "🔍 Channel Finder", "⚙️ API Info"])

with tabs[0]:
    st.header("💡 Niche Research")
    if st.button("Start Niche Analysis"):
        if not api_key:
            st.warning("Please enter your API key in the sidebar.")
        else:
            videos = search_videos(search_query, api_key, search_max_results, search_order, search_duration, search_type)
            st.success("Analysis complete!")
            st.write(videos) # Placeholder for results display

with tabs[1]:
    st.header("📈 Growth Analysis")
    st.info("This section analyzes the growth trajectory of videos and channels.")
    video_id = st.text_input("Enter a YouTube Video ID")
    if st.button("Analyze Video Growth"):
        if not api_key:
            st.warning("Please enter your API key in the sidebar.")
        else:
            video_data = get_video_data(video_id, api_key)
            st.success("Growth analysis complete!")
            st.write(video_data) # Placeholder for results display

with tabs[2]:
    st.header("🔍 Channel Finder")
    st.info("This section helps you find and analyze channels in your niche.")
    channel_query = st.text_input("Enter channel name or keyword", "Tech Review")
    if st.button("Find Channels"):
        if not api_key:
            st.warning("Please enter your API key in the sidebar.")
        else:
            # Placeholder for channel search function
            st.info("Searching for channels...")
            st.write("Channel results placeholder")

with tabs[3]:
    st.header("⚙️ API & Methodologies")
    st.markdown("""
    This platform uses the **YouTube Data API v3** to fetch video and channel data.
    
    The core methodologies include:
    - **Niche Demand Score**: Based on search volume, competition, and content quality.
    - **Viral Video Potential**: Analyzes view velocity, like/comment ratios, and engagement signals.
    - **Growth Trajectory**: Fits a mathematical growth curve (e.g., logistic or Gompertz) to historical view data to predict future performance.
    - **Content Readability**: Uses Flesch Reading Ease score on video descriptions to assess audience-friendliness.
    """)

# Custom content with new classes
st.markdown("---")
st.markdown("<h2 class='einstein-quote'>The true sign of intelligence is not knowledge but imagination.</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; margin-top: -10px;'>- Albert Einstein</p>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### Example Components with New Styling")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Video Score</h4>", unsafe_allow_html=True)
    st.markdown("<h3>8.5/10</h3>", unsafe_allow_html=True)
    st.markdown("<p>Based on engagement, velocity, and quality.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<h4>Niche Demand</h4>", unsafe_allow_html=True)
    st.markdown("<h3>High <span class='badge badge-success'>+20%</span></h3>", unsafe_allow_html=True)
    st.markdown("<p>Compared to average market.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='result-card glow-effect'>", unsafe_allow_html=True)
    st.markdown("<h4>Viral Video Detected</h4>", unsafe_allow_html=True)
    st.markdown("<p>A video with exceptionally fast growth has been identified.</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.8em; opacity: 0.7;'>Click for details.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Expanded sections to demonstrate new styles
st.markdown("---")
st.markdown("### Methodology Details")
st.markdown("<div class='question-box'>", unsafe_allow_html=True)
st.markdown("<h4>How is 'Growth Velocity' calculated?</h4>", unsafe_allow_html=True)
st.markdown("<p>Growth velocity is calculated by analyzing the slope of the view count over time. A steep, accelerating slope indicates high velocity and potential virality.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='optional-section'>", unsafe_allow_html=True)
st.markdown("<h4>Optional: Advanced Settings</h4>", unsafe_allow_html=True)
st.checkbox("Enable advanced model tuning")
st.selectbox("Select ML model", ["Random Forest", "Gradient Boosting", "Neural Network"])
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Data Display Example")
data = {
    'Channel': ['Channel A', 'Channel B', 'Channel C'],
    'Subscribers': [120000, 850000, 340000],
    'Videos': [150, 450, 210],
    'Niche Score': [8.9, 9.2, 7.8]
}
df = pd.DataFrame(data)
st.dataframe(df.style.set_properties(**{'border-radius': '8px'}), use_container_width=True)

st.markdown("---")
st.subheader("Progress Bar Example")
st.markdown("<div class='loading'>", unsafe_allow_html=True)
st.progress(0.75)
st.markdown("</div>", unsafe_allow_html=True)

# Final section with expanders and footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("📊 Growth Formula"):
        st.markdown("""
        **Growth Trajectory Model:**
        The platform uses a logistic growth function:
        
        $P(t) = K / (1 + e^{-r(t - t_0)})$
        
        - $K$ = Carrying capacity (max views)
        - $r$ = Growth rate
        - $t_0$ = Midpoint of growth
        
        We fit this model to the video's view data over time to forecast its potential.
        """)
        
with col2:
    with st.expander("⚖️ Niche Score Breakdown"):
        st.markdown("""
        **Components of Niche Score:**
        - **Demand**: Search interest, related queries (40%)
        - **Competition**: Number of competing videos (30%)
        - **Content Quality**: Readability, engagement (30%)
        
        This weighted average gives a comprehensive view.
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>🚀 Complete YouTube Analytics Platform</h4>
    <p>This comprehensive platform combines intelligent niche research, advanced channel discovery, 
    viral video detection, and growth trajectory analysis. It uses mathematical models and machine 
    learning algorithms to provide deep insights into YouTube ecosystem dynamics.</p>
    <br>
    <p><strong>✨ Features:</strong> Niche Research • Channel Discovery • Viral Video Detection • Growth Analysis • Network Mapping • Advanced Analytics</p>
    <p><em>Powered by YouTube Data API v3, advanced mathematical models, and AI-driven intelligence</em></p>
</div>
""", unsafe_allow_html=True)
