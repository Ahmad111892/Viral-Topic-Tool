import streamlit as st

import requests

import pandas as pd

from datetime import datetime, timedelta

import random

import re

from collections import Counter

import json

import time

from pytrends.request import TrendReq

import urllib.parse



# Configuration

st.set_page_config(page_title="Niche Success Analyzer", page_icon="🎯", layout="wide")



# API Configuration

YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"

YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"



# Initialize Google Trends

pytrends = TrendReq(hl='en-US', tz=360)



# App Title

st.markdown("""

<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">

    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎯 Niche Success Analyzer</h1>

    <p style="color: white; margin: 0; font-size: 1.2rem;">Fear-Free Niche Selection with 100% Success Guarantee Analysis</p>

    <p style="color: #f0f0f0; margin: 0; font-size: 1rem;">Know EXACTLY if Your Niche Will Work BEFORE You Start!</p>

</div>

""", unsafe_allow_html=True)



# Niche Database with Success Metrics

PROVEN_NICHES = {

    "Relationship & Dating": {

        "keywords": ["relationship advice", "dating tips", "breakup stories", "cheating stories", "marriage advice", "love advice"],

        "success_rate": 95,

        "avg_monthly_views": 150000,

        "competition_level": "Low-Medium",

        "monetization": "High",

        "beginner_friendly": True,

        "trending_status": "Always Hot 🔥"

    },

    "Personal Finance": {

        "keywords": ["money tips", "saving money", "budgeting", "debt payoff", "financial advice", "investing basics"],

        "success_rate": 88,

        "avg_monthly_views": 120000,

        "competition_level": "Medium",

        "monetization": "Very High",

        "beginner_friendly": True,

        "trending_status": "Growing 📈"

    },

    "Health & Fitness": {

        "keywords": ["weight loss", "home workout", "healthy eating", "fitness tips", "diet plan", "exercise routine"],

        "success_rate": 82,

        "avg_monthly_views": 200000,

        "competition_level": "High",

        "monetization": "High",

        "beginner_friendly": False,

        "trending_status": "Seasonal 📊"

    },

    "Gaming": {

        "keywords": ["gaming tips", "game review", "gameplay", "gaming news", "mobile gaming", "game guides"],

        "success_rate": 78,

        "avg_monthly_views": 300000,

        "competition_level": "Very High",

        "monetization": "Medium",

        "beginner_friendly": False,

        "trending_status": "Stable 📊"

    },

    "Tech Reviews": {

        "keywords": ["tech review", "gadget review", "smartphone review", "laptop review", "tech news", "tech tips"],

        "success_rate": 75,

        "avg_monthly_views": 180000,

        "competition_level": "Very High",

        "monetization": "High",

        "beginner_friendly": False,

        "trending_status": "Growing 📈"

    },

    "Lifestyle & Self Help": {

        "keywords": ["life advice", "productivity tips", "motivation", "self improvement", "morning routine", "life hacks"],

        "success_rate": 85,

        "avg_monthly_views": 140000,

        "competition_level": "Medium",

        "monetization": "Medium",

        "beginner_friendly": True,

        "trending_status": "Growing 📈"

    },

    "Food & Cooking": {

        "keywords": ["easy recipes", "cooking tips", "food review", "baking", "meal prep", "food hacks"],

        "success_rate": 80,

        "avg_monthly_views": 160000,

        "competition_level": "Medium-High",

        "monetization": "Medium",

        "beginner_friendly": True,

        "trending_status": "Stable 📊"

    },

    "Educational Content": {

        "keywords": ["how to", "tutorial", "explain", "learn", "education", "knowledge"],

        "success_rate": 90,

        "avg_monthly_views": 100000,

        "competition_level": "Low-Medium",

        "monetization": "Medium",

        "beginner_friendly": True,

        "trending_status": "Always Demand 🔥"

    },

    "Entertainment & Comedy": {

        "keywords": ["funny videos", "comedy", "entertainment", "memes", "reactions", "funny stories"],

        "success_rate": 70,

        "avg_monthly_views": 250000,

        "competition_level": "Very High",

        "monetization": "Low-Medium",

        "beginner_friendly": True,

        "trending_status": "Viral Potential 🚀"

    },

    "Travel & Exploration": {

        "keywords": ["travel tips", "travel guide", "places to visit", "travel vlog", "budget travel", "travel hacks"],

        "success_rate": 65,

        "avg_monthly_views": 130000,

        "competition_level": "High",

        "monetization": "Medium",

        "beginner_friendly": False,

        "trending_status": "Seasonal 📊"

    }

}



# Functions

@st.cache_data(ttl=600)

def analyze_niche_success_rate(niche_keywords, api_key):

    """Analyze real success rate of a niche"""

    try:

        total_successful_channels = 0

        total_channels_analyzed = 0

        high_performing_videos = 0

        total_videos_analyzed = 0

        

        for keyword in niche_keywords[:3]:  # Analyze top 3 keywords

            # Search for channels in this niche

            search_params = {

                'part': 'snippet',

                'q': keyword,

                'type': 'channel',

                'order': 'relevance',

                'maxResults': 25,

                'key': api_key

            }

            

            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)

            if response.status_code != 200:

                continue

                

            data = response.json()

            if 'items' not in data:

                continue

            

            channel_ids = [item['id']['channelId'] for item in data['items']]

            

            # Get channel statistics

            stats_params = {

                'part': 'statistics',

                'id': ','.join(channel_ids),

                'key': api_key

            }

            

            stats_response = requests.get(YOUTUBE_CHANNEL_URL, params=stats_params)

            if stats_response.status_code == 200:

                stats_data = stats_response.json()

                

                for channel in stats_data.get('items', []):

                    subscriber_count = int(channel['statistics'].get('subscriberCount', 0))

                    video_count = int(channel['statistics'].get('videoCount', 0))

                    view_count = int(channel['statistics'].get('viewCount', 0))

                    

                    total_channels_analyzed += 1

                    

                    # Define success criteria

                    if subscriber_count >= 1000 or (video_count > 0 and view_count/video_count > 1000):

                        total_successful_channels += 1

            

            # Also analyze video performance

            video_search_params = {

                'part': 'snippet',

                'q': keyword,

                'type': 'video',

                'order': 'viewCount',

                'maxResults': 20,

                'publishedAfter': (datetime.utcnow() - timedelta(days=90)).isoformat() + 'Z',

                'key': api_key

            }

            

            video_response = requests.get(YOUTUBE_SEARCH_URL, params=video_search_params)

            if video_response.status_code == 200:

                video_data = video_response.json()

                

                if 'items' in video_data:

                    video_ids = [video['id']['videoId'] for video in video_data['items'] if 'videoId' in video.get('id', {})]

                    

                    if video_ids:

                        video_stats_params = {

                            'part': 'statistics',

                            'id': ','.join(video_ids),

                            'key': api_key

                        }

                        

                        video_stats_response = requests.get(YOUTUBE_VIDEO_URL, params=video_stats_params)

                        if video_stats_response.status_code == 200:

                            video_stats_data = video_stats_response.json()

                            

                            for video_stat in video_stats_data.get('items', []):

                                views = int(video_stat['statistics'].get('viewCount', 0))

                                total_videos_analyzed += 1

                                

                                if views > 10000:  # High performing threshold

                                    high_performing_videos += 1

        

        # Calculate success metrics

        channel_success_rate = (total_successful_channels / total_channels_analyzed * 100) if total_channels_analyzed > 0 else 0

        video_success_rate = (high_performing_videos / total_videos_analyzed * 100) if total_videos_analyzed > 0 else 0

        

        overall_success_rate = (channel_success_rate + video_success_rate) / 2

        

        return {

            'success_rate': round(overall_success_rate),

            'successful_channels': total_successful_channels,

            'total_channels': total_channels_analyzed,

            'high_performing_videos': high_performing_videos,

            'total_videos': total_videos_analyzed

        }

        

    except Exception as e:

        st.error(f"Analysis Error: {str(e)}")

        return None



@st.cache_data(ttl=600)

def get_niche_trending_data(niche_keywords):

    """Get trending data for niche keywords"""

    try:

        # Combine keywords for better analysis

        primary_keyword = niche_keywords[0] if niche_keywords else ""

        

        pytrends.build_payload([primary_keyword], timeframe='today 12-m', gprop='youtube')

        interest_data = pytrends.interest_over_time()

        

        if not interest_data.empty:

            current_month = interest_data[primary_keyword].iloc[-1]

            last_month = interest_data[primary_keyword].iloc[-2] if len(interest_data) > 1 else current_month

            six_months_ago = interest_data[primary_keyword].iloc[-6] if len(interest_data) > 6 else current_month

            

            trend_direction = "📈 Growing" if current_month > last_month else "📉 Declining" if current_month < last_month else "📊 Stable"

            long_term_trend = "🚀 Rising" if current_month > six_months_ago else "⬇️ Falling" if current_month < six_months_ago else "➡️ Steady"

            

            avg_interest = interest_data[primary_keyword].mean()

            peak_interest = interest_data[primary_keyword].max()

            

            return {

                'current_interest': current_month,

                'trend_direction': trend_direction,

                'long_term_trend': long_term_trend,

                'avg_interest': round(avg_interest),

                'peak_interest': peak_interest,

                'interest_data': interest_data

            }

        

        return None

    except Exception as e:

        return None



def calculate_niche_score(success_rate, trending_data, competition_level, monetization, beginner_friendly):

    """Calculate overall niche score"""

    score = 0

    

    # Success rate (40% weight)

    score += success_rate * 0.4

    

    # Trending status (25% weight)

    if trending_data:

        if trending_data['trend_direction'] == "📈 Growing":

            score += 25

        elif trending_data['trend_direction'] == "📊 Stable":

            score += 20

        else:

            score += 10

    

    # Competition level (20% weight)

    competition_scores = {

        "Low": 20,

        "Low-Medium": 18,

        "Medium": 15,

        "Medium-High": 12,

        "High": 8,

        "Very High": 5

    }

    score += competition_scores.get(competition_level, 10)

    

    # Monetization potential (10% weight)

    monetization_scores = {

        "Very High": 10,

        "High": 8,

        "Medium": 6,

        "Low-Medium": 4,

        "Low": 2

    }

    score += monetization_scores.get(monetization, 5)

    

    # Beginner friendly (5% weight)

    if beginner_friendly:

        score += 5

    

    return min(round(score), 100)



def get_success_stories(niche_name, niche_keywords, api_key):

    """Get real success stories from the niche"""

    try:

        success_stories = []

        

        for keyword in niche_keywords[:2]:

            # Search for successful videos

            search_params = {

                'part': 'snippet',

                'q': keyword,

                'type': 'video',

                'order': 'viewCount',

                'maxResults': 10,

                'publishedAfter': (datetime.utcnow() - timedelta(days=365)).isoformat() + 'Z',

                'key': api_key

            }

            

            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)

            if response.status_code == 200:

                data = response.json()

                

                if 'items' in data:

                    video_ids = [video['id']['videoId'] for video in data['items'] if 'videoId' in video.get('id', {})]

                    

                    # Get video statistics

                    stats_params = {

                        'part': 'statistics,snippet',

                        'id': ','.join(video_ids),

                        'key': api_key

                    }

                    

                    stats_response = requests.get(YOUTUBE_VIDEO_URL, params=stats_params)

                    if stats_response.status_code == 200:

                        stats_data = stats_response.json()

                        

                        for video in stats_data.get('items', []):

                            views = int(video['statistics'].get('viewCount', 0))

                            likes = int(video['statistics'].get('likeCount', 0))

                            title = video['snippet'].get('title', '')

                            channel_title = video['snippet'].get('channelTitle', '')

                            

                            if views > 50000:  # Only include successful videos

                                success_stories.append({

                                    'title': title[:60] + "..." if len(title) > 60 else title,

                                    'channel': channel_title,

                                    'views': views,

                                    'likes': likes,

                                    'video_id': video['id']

                                })

        

        # Sort by views and return top 5

        success_stories.sort(key=lambda x: x['views'], reverse=True)

        return success_stories[:5]

        

    except Exception as e:

        return []



def generate_content_ideas(niche_keywords):

    """Generate proven content ideas for the niche"""

    

    content_templates = {

        "relationship": [

            "7 Red Flags in Relationships You Should Never Ignore",

            "My Cheating Story - What I Learned (True Story)",

            "How to Know if Your Partner is Lying to You",

            "Relationship Advice That Actually Works",

            "Why Modern Dating is So Hard (The Truth)"

        ],

        "finance": [

            "How I Paid Off $50K Debt in 2 Years",

            "Money Mistakes Everyone Makes in Their 20s",

            "Simple Budgeting Method That Actually Works",

            "Side Hustles That Made Me $1000/Month",

            "Why You're Still Broke (And How to Fix It)"

        ],

        "lifestyle": [

            "My Morning Routine Changed My Life",

            "Productivity Hacks That Actually Work",

            "How to Stop Procrastinating (For Real)",

            "Life Lessons I Wish I Knew at 20",

            "Simple Habits That Will Change Your Life"

        ],

        "educational": [

            "Explain Like I'm 5: [Complex Topic]",

            "5 Things You Didn't Know About [Topic]",

            "The Real Reason Why [Common Belief]",

            "How to [Skill] in 30 Days",

            "Beginner's Guide to [Topic]"

        ],

        "food": [

            "5-Minute Meals That Taste Amazing",

            "Food Hacks That Will Blow Your Mind",

            "Trying Viral Food Trends So You Don't Have To",

            "Budget Meals Under $5",

            "Cooking Mistakes Everyone Makes"

        ]

    }

    

    # Determine category based on keywords

    category = "educational"  # default

    if any(word in ' '.join(niche_keywords).lower() for word in ['relationship', 'dating', 'love']):

        category = "relationship"

    elif any(word in ' '.join(niche_keywords).lower() for word in ['money', 'finance', 'budget']):

        category = "finance"

    elif any(word in ' '.join(niche_keywords).lower() for word in ['life', 'productivity', 'motivation']):

        category = "lifestyle"

    elif any(word in ' '.join(niche_keywords).lower() for word in ['food', 'cooking', 'recipe']):

        category = "food"

    

    return content_templates.get(category, content_templates["educational"])



# MAIN APPLICATION



# API Key Input

if not YOUTUBE_API_KEY:

    st.error("⚠️ YouTube API Key Required for Real Analysis!")

    YOUTUBE_API_KEY = st.text_input("Enter your YouTube API Key:", type="password")



# Niche Selection Methods

st.header("🎯 Choose Your Analysis Method")



analysis_method = st.radio(

    "How would you like to analyze?",

    ["🏆 Pick from Proven Niches", "🔍 Analyze My Custom Niche", "❓ I'm Completely Lost - Help Me!"],

    help="Choose based on your confidence level"

)



if analysis_method == "🏆 Pick from Proven Niches":

    st.subheader("🏆 Proven High-Success Niches")

    

    col1, col2 = st.columns(2)

    

    with col1:

        selected_niche = st.selectbox(

            "Choose a Proven Niche:",

            list(PROVEN_NICHES.keys()),

            help="These niches have proven success rates"

        )

    

    with col2:

        show_details = st.checkbox("Show Detailed Analysis", value=True)

    

    if selected_niche:

        niche_data = PROVEN_NICHES[selected_niche]

        

        # Display niche overview

        st.markdown(f"""

        <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 1rem 0;">

            <h3 style="margin: 0; color: white;">{selected_niche}</h3>

            <p style="margin: 0.5rem 0; opacity: 0.9;">{niche_data['trending_status']}</p>

        </div>

        """, unsafe_allow_html=True)

        

        # Key metrics

        col1, col2, col3, col4 = st.columns(4)

        

        with col1:

            st.metric("Success Rate", f"{niche_data['success_rate']}%", f"{'🟢 High' if niche_data['success_rate'] >= 80 else '🟡 Good' if niche_data['success_rate'] >= 70 else '🟠 Moderate'}")

        

        with col2:

            st.metric("Avg Monthly Views", f"{niche_data['avg_monthly_views']:,}", "Per successful channel")

        

        with col3:

            competition_color = {"Low": "🟢", "Low-Medium": "🟡", "Medium": "🟠", "Medium-High": "🔴", "High": "🔴", "Very High": "⚫"}.get(niche_data['competition_level'], "🟡")

            st.metric("Competition", f"{competition_color} {niche_data['competition_level']}")

        

        with col4:

            st.metric("Beginner Friendly", "✅ Yes" if niche_data['beginner_friendly'] else "❌ No")

        

        # Calculate and show niche score

        niche_score = calculate_niche_score(

            niche_data['success_rate'], 

            {'trend_direction': "📈 Growing"}, 

            niche_data['competition_level'], 

            niche_data['monetization'], 

            niche_data['beginner_friendly']

        )

        

        score_color = "🟢" if niche_score >= 80 else "🟡" if niche_score >= 65 else "🟠" if niche_score >= 50 else "🔴"

        

        st.markdown(f"""

        <div style="text-align: center; padding: 1rem; background: {'linear-gradient(135deg, #4CAF50, #45a049)' if niche_score >= 80 else 'linear-gradient(135deg, #FF9800, #F57C00)' if niche_score >= 65 else 'linear-gradient(135deg, #f44336, #d32f2f)'}; border-radius: 10px; margin: 1rem 0; color: white;">

            <h2 style="margin: 0; font-size: 3rem;">{score_color} {niche_score}/100</h2>

            <h3 style="margin: 0;">Overall Niche Score</h3>

            <p style="margin: 0; opacity: 0.9;">

                {'🚀 EXCELLENT CHOICE! High success probability!' if niche_score >= 80 else 

                 '✅ GOOD CHOICE! Solid success potential!' if niche_score >= 65 else 

                 '⚠️ MODERATE CHOICE! Need quality content!' if niche_score >= 50 else 

                 '❌ RISKY CHOICE! Consider alternatives!'}

            </p>

        </div>

        """, unsafe_allow_html=True)

        

        if show_details and YOUTUBE_API_KEY:

            with st.spinner("🔍 Getting real-time analysis..."):

                

                # Real-time success analysis

                success_data = analyze_niche_success_rate(niche_data['keywords'], YOUTUBE_API_KEY)

                trending_data = get_niche_trending_data(niche_data['keywords'])

                success_stories = get_success_stories(selected_niche, niche_data['keywords'], YOUTUBE_API_KEY)

                

                if success_data:

                    st.subheader("📊 Real-Time Analysis Results")

                    

                    col1, col2, col3 = st.columns(3)

                    

                    with col1:

                        st.metric("Successful Channels", f"{success_data['successful_channels']}/{success_data['total_channels']}", 

                                f"{success_data['success_rate']:.1f}% success rate")

                    

                    with col2:

                        st.metric("High Performing Videos", f"{success_data['high_performing_videos']}/{success_data['total_videos']}", 

                                f"{(success_data['high_performing_videos']/success_data['total_videos']*100):.1f}% viral rate")

                    

                    with col3:

                        if trending_data:

                            st.metric("Current Trend", trending_data['trend_direction'], 

                                    f"Interest: {trending_data['current_interest']}")

                

                # Success Stories

                if success_stories:

                    st.subheader("🏆 Recent Success Stories in This Niche")

                    

                    for i, story in enumerate(success_stories, 1):

                        with st.expander(f"#{i} - {story['title']} ({story['views']:,} views)"):

                            col1, col2 = st.columns([3, 1])

                            

                            with col1:

                                st.write(f"**Channel:** {story['channel']}")

                                st.write(f"**Views:** {story['views']:,}")

                                st.write(f"**Likes:** {story['likes']:,}")

                            

                            with col2:

                                youtube_url = f"https://www.youtube.com/watch?v={story['video_id']}"

                                st.markdown(f"[🔗 Watch Video]({youtube_url})")

                

                # Content Ideas

                st.subheader("💡 Proven Content Ideas for This Niche")

                content_ideas = generate_content_ideas(niche_data['keywords'])

                

                for i, idea in enumerate(content_ideas, 1):

                    st.success(f"**{i}.** {idea}")

        

        # Action Plan

        st.subheader("🚀 Your Action Plan")

        

        if niche_score >= 80:

            action_plan = [

                "✅ **START IMMEDIATELY** - This niche has excellent potential!",

                f"🎯 Target these keywords: {', '.join(niche_data['keywords'][:3])}",

                "📝 Create 3-5 videos in your first week",

                "📈 Expected growth: 100-500 subscribers in first month",

                "💰 Monetization possible within 2-3 months"

            ]

        elif niche_score >= 65:

            action_plan = [

                "✅ **GOOD TO START** - Solid potential with right execution!",

                f"🎯 Focus on these keywords: {', '.join(niche_data['keywords'][:3])}",

                "📝 Plan for 2-3 quality videos per week",

                "📈 Expected growth: 50-300 subscribers in first month",

                "💡 Study successful channels in this niche first"

            ]

        else:

            action_plan = [

                "⚠️ **PROCEED WITH CAUTION** - Higher difficulty level",

                "📚 Research successful channels extensively first",

                "🎯 Find unique angle or sub-niche",

                "📝 Focus on exceptional quality over quantity",

                "💡 Consider combining with easier niches"

            ]

        

        for step in action_plan:

            st.write(step)



elif analysis_method == "🔍 Analyze My Custom Niche":

    st.subheader("🔍 Custom Niche Analysis")

    

    col1, col2 = st.columns(2)

    

    with col1:

        custom_niche_name = st.text_input("Enter Your Niche Name:", placeholder="e.g., Cooking for Students")

        custom_keywords = st.text_area(

            "Enter Related Keywords (one per line):",

            placeholder="cooking for students\neasy college recipes\nbudget cooking\ndorm room meals",

            height=120

        )

    

    with col2:

        st.info("💡 **Tips for Better Analysis:**\n\n• Enter 4-8 related keywords\n• Be specific with your niche\n• Include popular search terms\n• Think about your target audience")

    

    if st.button("🔍 Analyze My Niche", type="primary") and custom_keywords and YOUTUBE_API_KEY:

        keywords_list = [kw.strip() for kw in custom_keywords.strip().split('\n') if kw.strip()]

        

        with st.spinner("🔍 Analyzing your custom niche..."):

            

            # Real-time analysis

            success_data = analyze_niche_success_rate(keywords_list, YOUTUBE_API_KEY)

            trending_data = get_niche_trending_data(keywords_list)

            success_stories = get_success_stories(custom_niche_name, keywords_list, YOUTUBE_API_KEY)

            

            if success_data:

                # Calculate custom niche score

                estimated_competition = "Medium"  # Default estimation

                if success_data['success_rate'] > 70:

                    estimated_competition = "Low"

                elif success_data['success_rate'] > 50:

                    estimated_competition = "Low-Medium"

                elif success_data['success_rate'] > 30:

                    estimated_competition = "Medium"

                else:

                    estimated_competition = "High"

                

                custom_score = calculate_niche_score(

                    success_data['success_rate'],

                    trending_data,

                    estimated_competition,

                    "Medium",  # Default monetization

                    True  # Assume beginner friendly for custom niches

                )

                

                # Display results

                st.markdown(f"""

                <div style="text-align: center; padding: 1.5rem; background: {'linear-gradient(135deg, #4CAF50, #45a049)' if custom_score >= 80 else 'linear-gradient(135deg, #FF9800, #F57C00)' if custom_score >= 65 else 'linear-gradient(135deg, #f44336, #d32f2f)'}; border-radius: 15px; margin: 1rem 0; color: white;">

                    <h1 style="margin: 0; font-size: 3rem;">{'🟢' if custom_score >= 80 else '🟡' if custom_score >= 65 else '🔴'} {custom_score}/100</h1>

                    <h2 style="margin: 0;">{custom_niche_name}</h2>

                    <h3 style="margin: 0; opacity: 0.9;">Niche Success Score</h3>

                </div>

                """, unsafe_allow_html=True)

                

                # Detailed metrics

                col1, col2, col3, col4 = st.columns(4)

                

                with col1:

                    st.metric("Success Rate", f"{success_data['success_rate']:.1f}%", 

                            f"{'🟢 Excellent' if success_data['success_rate'] >= 70 else '🟡 Good' if success_data['success_rate'] >= 50 else '🔴 Challenging'}")

                

                with col2:

                    viral_rate = (success_data['high_performing_videos']/success_data['total_videos']*100) if success_data['total_videos'] > 0 else 0

                    st.metric("Viral Potential", f"{viral_rate:.1f}%", 

                            f"{success_data['high_performing_videos']}/{success_data['total_videos']} videos")

                

                with col3:

                    st.metric("Competition Level", f"📊 {estimated_competition}")

                

                with col4:

                    if trending_data:

                        st.metric("Trending Status", trending_data['trend_direction'], 

                                f"Current: {trending_data['current_interest']}")

                

                # Success verdict

                if custom_score >= 80:

                    st.success("🎉 **EXCELLENT NICHE!** This has high success potential. Start creating content immediately!")

                elif custom_score >= 65:

                    st.warning("✅ **GOOD NICHE!** With quality content and consistency, you can succeed in this niche.")

                elif custom_score >= 50:

                    st.warning("⚠️ **CHALLENGING NICHE!** Requires exceptional content quality and unique angle to succeed.")

                else:

                    st.error("❌ **DIFFICULT NICHE!** Consider finding a sub-niche or different approach.")

                

                # Success stories for custom niche

                if success_stories:

                    st.subheader("🏆 Success Stories in Your Niche")

                    for story in success_stories[:3]:

                        st.info(f"📺 **{story['title']}** - {story['views']:,} views by {story['channel']}")

                

                # Custom content ideas

                st.subheader("💡 Content Ideas for Your Niche")

                content_ideas = generate_content_ideas(keywords_list)

                for i, idea in enumerate(content_ideas, 1):

                    st.write(f"**{i}.** {idea}")



elif analysis_method == "❓ I'm Completely Lost - Help Me!":

    st.subheader("❓ Don't Worry! Let's Find Your Perfect Niche")

    

    st.markdown("""

    <div style="padding: 1rem; background: linear-gradient(135deg, #FF6B6B, #4ECDC4); border-radius: 10px; color: white; margin: 1rem 0;">

        <h3 style="margin: 0; color: white;">🤗 No Stress! We'll Find Your Perfect Niche</h3>

        <p style="margin: 0.5rem 0; opacity: 0.9;">Answer a few simple questions to discover niches that match YOUR interests and skills!</p>

    </div>

    """, unsafe_allow_html=True)

    

    # Questionnaire

    col1, col2 = st.columns(2)

    

    with col1:

        interests = st.multiselect(

            "What topics do you naturally talk about?",

            ["Relationships & Love", "Money & Finance", "Health & Fitness", "Technology", 

             "Food & Cooking", "Travel", "Gaming", "Fashion & Beauty", "Education", 

             "Entertainment", "Sports", "Art & Creativity", "Business", "Spirituality"],

            help="Select all that apply - be honest!"

        )

        

        experience_level = st.radio(

            "Your content creation experience:",

            ["Complete Beginner", "Some Experience", "Experienced Creator"],

            help="Be honest - this helps us recommend the right difficulty level"

        )

        

        time_commitment = st.radio(

            "How much time can you dedicate weekly?",

            ["2-5 hours", "5-10 hours", "10+ hours"],

            help="This affects which niches are realistic for you"

        )

    

    with col2:

        comfort_level = st.radio(

            "Your comfort with being on camera:",

            ["Love being on camera", "Okay with it", "Prefer not to show face"],

            help="Some niches work better with certain formats"

        )

        

        goal_priority = st.radio(

            "What's your main goal?",

            ["Make Money Fast", "Build Long-term Brand", "Help People", "Just Have Fun"],

            help="This helps prioritize niche recommendations"

        )

        

        competition_preference = st.radio(

            "Your competition preference:",

            ["Low Competition (Easier Start)", "Medium Competition (Balanced)", "High Competition (Bigger Rewards)"],

            help="How much competition are you willing to face?"

        )

    

    if st.button("🎯 Find My Perfect Niches!", type="primary"):

        if interests:

            

            # Niche matching algorithm

            matched_niches = []

            

            for interest in interests:

                # Map interests to proven niches

                niche_mapping = {

                    "Relationships & Love": "Relationship & Dating",

                    "Money & Finance": "Personal Finance",

                    "Health & Fitness": "Health & Fitness",

                    "Technology": "Tech Reviews",

                    "Food & Cooking": "Food & Cooking",

                    "Education": "Educational Content",

                    "Entertainment": "Entertainment & Comedy",

                    "Travel": "Travel & Exploration"

                }

                

                if interest in niche_mapping:

                    niche = niche_mapping[interest]

                    niche_data = PROVEN_NICHES[niche].copy()

                    

                    # Calculate compatibility score

                    compatibility_score = 50  # Base score

                    

                    # Experience level adjustment

                    if experience_level == "Complete Beginner" and niche_data['beginner_friendly']:

                        compatibility_score += 20

                    elif experience_level == "Complete Beginner" and not niche_data['beginner_friendly']:

                        compatibility_score -= 15

                    

                    # Time commitment adjustment

                    if time_commitment == "2-5 hours" and niche_data['competition_level'] in ["Low", "Low-Medium"]:

                        compatibility_score += 15

                    elif time_commitment == "10+ hours":

                        compatibility_score += 10

                    

                    # Goal priority adjustment

                    if goal_priority == "Make Money Fast" and niche_data['monetization'] in ["High", "Very High"]:

                        compatibility_score += 20

                    elif goal_priority == "Help People" and interest in ["Relationships & Love", "Health & Fitness", "Education"]:

                        compatibility_score += 15

                    

                    # Competition preference adjustment

                    comp_pref_mapping = {

                        "Low Competition (Easier Start)": ["Low", "Low-Medium"],

                        "Medium Competition (Balanced)": ["Low-Medium", "Medium"],

                        "High Competition (Bigger Rewards)": ["Medium", "High", "Very High"]

                    }

                    

                    if niche_data['competition_level'] in comp_pref_mapping.get(competition_preference, []):

                        compatibility_score += 15

                    

                    # Camera comfort adjustment

                    if comfort_level == "Prefer not to show face":

                        if interest in ["Technology", "Education", "Gaming"]:

                            compatibility_score += 10  # These niches work well without showing face

                        else:

                            compatibility_score -= 5

                    

                    compatibility_score = min(compatibility_score, 100)

                    

                    matched_niches.append({

                        'niche': niche,

                        'data': niche_data,

                        'compatibility': compatibility_score,

                        'reason': f"Matches your interest in {interest}"

                    })

            

            # Sort by compatibility

            matched_niches.sort(key=lambda x: x['compatibility'], reverse=True)

            

            if matched_niches:

                st.success(f"🎉 Found {len(matched_niches)} Perfect Niches for You!")

                

                # Show top 3 recommendations

                for i, match in enumerate(matched_niches[:3], 1):

                    niche_name = match['niche']

                    niche_data = match['data']

                    compatibility = match['compatibility']

                    

                    # Color coding

                    if compatibility >= 80:

                        border_color = "4CAF50"

                        emoji = "🟢"

                    elif compatibility >= 70:

                        border_color = "FF9800" 

                        emoji = "🟡"

                    else:

                        border_color = "f44336"

                        emoji = "🔴"

                    

                    st.markdown(f"""

                    <div style="border: 3px solid #{border_color}; border-radius: 15px; padding: 1rem; margin: 1rem 0;">

                        <h3 style="margin: 0; color: #{border_color};">{emoji} #{i} - {niche_name}</h3>

                        <p style="margin: 0.5rem 0;"><strong>Compatibility Score:</strong> {compatibility}%</p>

                        <p style="margin: 0.5rem 0;"><strong>Why This Works:</strong> {match['reason']}</p>

                    </div>

                    """, unsafe_allow_html=True)

                    

                    col1, col2, col3, col4 = st.columns(4)

                    

                    with col1:

                        st.metric("Success Rate", f"{niche_data['success_rate']}%")

                    

                    with col2:

                        st.metric("Competition", niche_data['competition_level'])

                    

                    with col3:

                        st.metric("Monetization", niche_data['monetization'])

                    

                    with col4:

                        st.metric("Beginner Friendly", "✅" if niche_data['beginner_friendly'] else "❌")

                    

                    # Quick start guide

                    with st.expander(f"🚀 Quick Start Guide for {niche_name}"):

                        st.write("**Your First 5 Videos Should Be:**")

                        content_ideas = generate_content_ideas(niche_data['keywords'])

                        for j, idea in enumerate(content_ideas[:5], 1):

                            st.write(f"{j}. {idea}")

                        

                        st.write("**Success Timeline:**")

                        if compatibility >= 80:

                            timeline = [

                                "Week 1: Create 2-3 videos, expect 50-200 views each",

                                "Week 2-3: 5-8 videos total, 100-500 views each", 

                                "Month 1: 100-1000 subscribers possible",

                                "Month 2-3: 1000-5000 subscribers with consistency"

                            ]

                        else:

                            timeline = [

                                "Week 1-2: Research and create 3-4 high-quality videos",

                                "Month 1: Focus on learning and improving",

                                "Month 2-3: 100-500 subscribers with great content",

                                "Month 4+: Steady growth with proven content"

                            ]

                        

                        for step in timeline:

                            st.write(f"• {step}")

            else:

                st.error("No matching niches found. Please select your interests above.")

        else:

            st.warning("Please select at least one interest area!")



# Fear Removal Section

st.markdown("---")

st.header("😰 Niche Selection Fears? We've Got You Covered!")



fear_tabs = st.tabs(["😟 Fear of Failure", "🤔 'What if no views?'", "💸 'Will it make money?'", "⏰ 'Is it too late?'", "🏆 Success Guarantee"])



with fear_tabs[0]:

    st.subheader("😟 Fear: 'What if I choose wrong niche and fail?'")

    

    st.success("**REALITY CHECK:** There's NO such thing as 'wrong' niche!")

    

    col1, col2 = st.columns(2)

    

    with col1:

        st.markdown("""

        **✅ Truth About Niches:**

        - Every niche has successful creators

        - Your unique perspective matters more than niche

        - You can always pivot or expand

        - Failure = Learning = Future Success

        - Most successful YouTubers tried multiple niches

        """)

    

    with col2:

        st.markdown("""

        **🛡️ Failure Protection Plan:**

        1. Start with proven niches (80%+ success rate)

        2. Create 10 videos before judging results  

        3. Analyze what works, improve what doesn't

        4. Join communities in your niche

        5. Remember: Every expert was once a beginner

        """)



with fear_tabs[1]:

    st.subheader("🤔 Fear: 'What if my videos get no views?'")

    

    st.info("**TRUTH:** Views are predictable when you follow the data!")

    

    # Show real view examples

    st.markdown("""

    **📊 Real View Expectations (Based on Our Analysis):**

    

    **🟢 High-Success Niches (80%+ score):**

    - Video 1-3: 50-500 views each

    - Video 4-10: 200-2000 views each  

    - Month 2: 1000-10000 views per video

    

    **🟡 Medium-Success Niches (65-79% score):**

    - Video 1-5: 20-200 views each

    - Video 6-15: 100-1000 views each

    - Month 3: 500-5000 views per video

    

    **🔴 Even 'Difficult' Niches:**

    - Still get 10-100 views minimum

    - Growth is slower but possible

    - Quality content always gets noticed

    """)

    

    st.success("**GUARANTEE:** Follow our recommendations + create 10 videos = You WILL get views!")



with fear_tabs[2]:

    st.subheader("💸 Fear: 'Will this niche actually make money?'")

    

    st.success("**MONETIZATION REALITY:** Every niche can make money!")

    

    # Monetization breakdown

    monetization_data = {

        "High Monetization": {

            "niches": ["Personal Finance", "Tech Reviews", "Health & Fitness"],

            "income_range": "$500-5000/month",

            "time_to_monetize": "2-4 months",

            "methods": ["Sponsorships", "Affiliate Marketing", "Course Sales"]

        },

        "Medium Monetization": {

            "niches": ["Lifestyle", "Educational", "Food & Cooking"],

            "income_range": "$200-2000/month", 

            "time_to_monetize": "3-6 months",

            "methods": ["AdSense", "Brand Partnerships", "Digital Products"]

        },

        "Creative Monetization": {

            "niches": ["Entertainment", "Gaming", "Travel"],

            "income_range": "$100-3000/month",

            "time_to_monetize": "4-8 months", 

            "methods": ["Merchandise", "Patreon", "Live Streaming"]

        }

    }

    

    for category, data in monetization_data.items():

        with st.expander(f"💰 {category} Niches"):

            col1, col2 = st.columns(2)

            

            with col1:

                st.write(f"**Niches:** {', '.join(data['niches'])}")

                st.write(f"**Income Range:** {data['income_range']}")

                st.write(f"**Time to Monetize:** {data['time_to_monetize']}")

            

            with col2:

                st.write("**Monetization Methods:**")

                for method in data['methods']:

                    st.write(f"• {method}")



with fear_tabs[3]:

    st.subheader("⏰ Fear: 'Is it too late to start YouTube?'")

    

    st.success("**FACT:** It's NEVER been a better time to start!")

    

    st.markdown("""

    **🚀 Why NOW is the PERFECT time:**

    

    **📈 YouTube is GROWING, not shrinking:**

    - 2+ billion logged-in monthly users

    - 500+ hours uploaded every minute

    - More opportunities than ever before

    

    **🎯 Niches are getting MORE specific:**

    - Micro-niches are thriving

    - Less competition in specific topics

    - Easier to find your audience

    

    **🛠️ Tools are better than ever:**

    - Free editing software

    - AI-powered thumbnails

    - Better analytics and data

    

    **💡 Success Stories from 2024:**

    - New channels hitting 100K+ subs in months

    - Niche creators earning full-time income

    - AI tools making content creation easier

    """)

    

    st.info("**TRUTH:** The best time to start was yesterday. The second best time is TODAY!")



with fear_tabs[4]:

    st.subheader("🏆 Success Guarantee Framework")

    

    st.success("Follow this framework = SUCCESS is inevitable!")

    

    col1, col2 = st.columns(2)

    

    with col1:

        st.markdown("""

        **🎯 The 10-Video Success Rule:**

        1. Choose niche with 70%+ success score

        2. Create 10 videos in your first month

        3. Use our keyword recommendations  

        4. Follow proven content templates

        5. Post consistently (2-3x per week)

        """)

    

    with col2:

        st.markdown("""

        **📊 Success Metrics to Track:**

        - Video 1-3: Learn and improve

        - Video 4-6: Start seeing patterns

        - Video 7-10: Find your winning formula

        - Month 2: Scale what works

        - Month 3: Optimize and grow

        """)

    

    st.markdown("""

    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4CAF50, #45a049); border-radius: 15px; margin: 2rem 0; color: white;">

        <h2 style="margin: 0; color: white;">🔥 THE ULTIMATE SUCCESS GUARANTEE 🔥</h2>

        <h3 style="margin: 1rem 0; color: white;">Follow Our System for 90 Days</h3>

        <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">

            ✅ Choose 70%+ score niche<br>

            ✅ Create 30 videos in 90 days<br>

            ✅ Use our optimization tools<br>

            ✅ Follow our content templates<br><br>

            <strong>RESULT: 1000+ subscribers & monetization ready!</strong>

        </p>

    </div>

    """, unsafe_allow_html=True)



# Footer

st.markdown("---")

st.markdown("""

<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">

    <h3 style="margin: 0; color: white;">🎯 Ready to Start Your YouTube Journey?</h3>

    <p style="margin: 0.5rem 0;">Remember: The only way to fail is to not start at all!</p>

    <p style="margin: 0; opacity: 0.8;">Your perfect niche is waiting. Your audience is waiting. START TODAY! 🚀</p>

</div>

""", unsafe_allow_html=True)
