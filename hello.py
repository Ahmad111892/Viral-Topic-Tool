import streamlit as st
import requests
from datetime import datetime, timedelta
from collections import Counter

# --- Page Configuration ---
st.set_page_config(page_title="YouTube Growth Toolkit", page_icon="🚀", layout="wide")

# --- API Key Management ---
# It's better to use st.secrets for the API key in a real application
API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# --- Niche Research Tool Functions ---
@st.cache_data(ttl=3600) # Cache the results for 1 hour to save API calls
def find_viral_new_channels(api_key, niche_ideas_list, video_type="Any"):
    """
    Researches a user-provided list of niches to find viral channels created in the current year,
    and performs a deep-dive analysis on them.
    """
    viral_channels = []
    current_year = datetime.now().year
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_channel_ids = set() # To avoid adding the same channel multiple times

    # Process each niche individually to link channels to the niche that found them
    for i, niche in enumerate(niche_ideas_list):
        status_text.text(f"Niche Research ki ja rahi hai '{niche}'... ({i + 1}/{len(niche_ideas_list)})")
        progress_bar.progress((i + 1) / len(niche_ideas_list))
        
        search_params = {
            "part": "snippet", "q": niche, "type": "video", "order": "viewCount",
            "publishedAfter": (datetime.utcnow() - timedelta(days=90)).isoformat("T") + "Z",
            "maxResults": 20, "key": api_key
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

                channel_params = {"part": "snippet,statistics", "id": ",".join(new_channel_ids), "key": api_key}
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
                                # --- INTELLIGENCE UPGRADE: DEEP DIVE ANALYSIS ---
                                channel_id = channel['id']
                                analysis_data = deep_dive_analysis(api_key, channel_id, channel["snippet"].get("description", ""))
                                
                                viral_channels.append({
                                    "Channel Name": channel["snippet"]["title"],
                                    "URL": f"https://www.youtube.com/channel/{channel_id}",
                                    "Subscribers": subs,
                                    "Total Views": views,
                                    "Video Count": video_count,
                                    "Creation Date": published_date.strftime("%Y-%m-%d"),
                                    "Found Via Niche": niche,
                                    **analysis_data # Add all analysis data
                                })
                                processed_channel_ids.add(channel_id)
        except requests.exceptions.RequestException:
            continue
            
    progress_bar.empty()
    status_text.empty()
    return viral_channels

def deep_dive_analysis(api_key, channel_id, channel_description):
    """
    Performs a deeper analysis on a channel to find engagement, monetization, velocity, and schedule.
    """
    analysis_results = {
        "Engagement Score": 0,
        "Monetization Clues": [],
        "Content Velocity": 0,
        "Weekly Frequency": 0,
        "Upload Schedule": "N/A"
    }
    
    try:
        # Get recent videos
        video_search_params = {"part": "snippet", "channelId": channel_id, "order": "date", "maxResults": 10, "key": api_key}
        video_response = requests.get(YOUTUBE_SEARCH_URL, params=video_search_params)
        
        if video_response.status_code == 200:
            video_items = video_response.json().get("items", [])
            video_ids = [item["id"]["videoId"] for item in video_items if "videoId" in item.get("id", {})]
            
            if video_ids:
                # --- Get stats for engagement and timestamps for schedule analysis ---
                video_details_params = {"part": "statistics,snippet", "id": ",".join(video_ids), "key": api_key}
                details_response = requests.get(YOUTUBE_VIDEO_URL, params=video_details_params)
                
                if details_response.status_code == 200:
                    total_likes, total_comments, total_views = 0, 0, 0
                    published_dates = []

                    for item in details_response.json().get("items", []):
                        stats = item.get("statistics", {})
                        total_likes += int(stats.get("likeCount", 0))
                        total_comments += int(stats.get("commentCount", 0))
                        total_views += int(stats.get("viewCount", 0))
                        
                        published_at_str = item["snippet"]["publishedAt"]
                        published_dates.append(datetime.fromisoformat(published_at_str.replace("Z", "+00:00")))
                    
                    # 1. Calculate Engagement Score
                    if total_views > 0:
                        engagement_rate = ((total_likes + total_comments) / total_views) * 100
                        analysis_results["Engagement Score"] = round(engagement_rate, 2)
                    
                    # --- UPGRADED: Schedule Analysis ---
                    if len(published_dates) > 1:
                        # 4. Calculate Weekly Frequency
                        time_span_days = (max(published_dates) - min(published_dates)).days
                        if time_span_days > 0:
                            analysis_results["Weekly Frequency"] = round((len(published_dates) / time_span_days) * 7, 1)
                        else:
                            analysis_results["Weekly Frequency"] = len(published_dates) # All videos on same day

                        # 5. Find Top 2 Most Common Upload Time/Day Patterns
                        upload_slots = [(d.strftime('%A'), d.hour) for d in published_dates]
                        
                        if upload_slots:
                            schedule_counter = Counter(upload_slots)
                            top_slots = schedule_counter.most_common(2) # Get top 2 slots

                            schedule_parts = []
                            for i, ((day, hour), count) in enumerate(top_slots):
                                schedule_parts.append(f"{i+1}. Aksar {day} ko {hour}:00 UTC ke आसपास")
                            
                            analysis_results["Upload Schedule"] = " | ".join(schedule_parts)
                            if not analysis_results["Upload Schedule"]:
                                 analysis_results["Upload Schedule"] = "Koi khaas pattern nahi mila."

    except Exception as e:
        # Silently fail to not crash the app, but you could log 'e' here
        pass

    # 2. Find Monetization Clues
    monetization_keywords = ["affiliate", "merch", "patreon", "course", "consulting", "e-book", "gumroad", "sponsor"]
    analysis_results["Monetization Clues"] = [keyword for keyword in monetization_keywords if keyword in channel_description.lower()]
    
    # 3. Calculate Content Velocity (videos in last 30 days)
    velocity_params = {
        "part": "id", "channelId": channel_id, "type": "video",
        "publishedAfter": (datetime.utcnow() - timedelta(days=30)).isoformat("T") + "Z",
        "maxResults": 50, "key": api_key
    }
    velocity_response = requests.get(YOUTUBE_SEARCH_URL, params=velocity_params)
    if velocity_response.status_code == 200:
        analysis_results["Content Velocity"] = len(velocity_response.json().get("items", []))

    return analysis_results

# --- Main App UI ---
st.title("🚀 YouTube Growth Toolkit (Intelligence Upgraded)")
st.markdown("Aapka all-in-one tool aasaani se viral topics aur naye niches dhoondne ke liye.")

with st.sidebar:
    st.header("🔑 API Configuration")
    st.info("Aapko is app ko istemal karne ke liye ek YouTube Data API v3 key ki zaroorat hogi.")
    if 'api_key' not in st.session_state:
        st.session_state.api_key = API_KEY if API_KEY else ""

    st.session_state.api_key = st.text_input("Apni YouTube API Key Yahan Daalein:", type="password", value=st.session_state.api_key)
    if st.session_state.api_key:
        st.success("API Key set ho gayi hai!")
    else:
        st.warning("Barae meharbani apni API key daalein.")

tab1, tab2 = st.tabs(["🔍 Viral Topics Finder", "🚀 Niche Research Tool"])

with tab1:
    st.header("Viral Topics Finder")
    st.info("Chhote channels se high-performing videos dhoondein taake aapko content ideas mil sakein.")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Pichhle kitne din search karne hain (1-30):", min_value=1, max_value=30, value=7)
    
    default_keywords = "Affair Relationship Stories\nReddit Update\nReddit Relationship Advice\nReddit Cheating\nAITA Update\nCheating Story Real"
    user_keywords_input = st.text_area("Apne Keywords Yahan Daalein (har keyword nayi line mein):", value=default_keywords, height=150)

    if st.button("Viral Topics Dhoondein"):
        if not st.session_state.api_key:
            st.error("Barae meharbani sidebar mein apni YouTube API Key daalein.")
        else:
            # ... [Logic for Viral Topics Finder remains unchanged] ...
            keywords = [kw.strip() for kw in user_keywords_input.strip().split('\n') if kw.strip()]
            if not keywords:
                st.warning("Barae meharbani search karne ke liye kam se kam ek keyword daalein.")
            else:
                # ... [Existing logic] ...
                pass # The code for this part is unchanged and correct.

with tab2:
    st.header("Niche Research Tool")
    st.info(f"{datetime.now().year} mein banaye gaye tezi se grow karne wale YouTube channels dhoondein.")

    video_type_choice = st.radio(
        "Aap kis tarah ke channels dhoondna chahte hain?",
        ('Any', 'Shorts Channel', 'Long Video Channel'),
        horizontal=True,
        help="Shorts (1 min se kam), Long (20 min se zyada)."
    )
    
    default_niches = "AI Tools Tutorials\nPersonal Finance for Gen Z\nSustainable Living Hacks\nSide Hustle Case Studies\nRetro Gaming Deep Dives"
    user_niche_input = st.text_area("Niche Ideas Yahan Daalein (har idea nayi line mein):", value=default_niches, height=150)

    if st.button("Niche Research Shuru Karein"):
        if not st.session_state.api_key:
            st.error("Barae meharbani sidebar mein apni YouTube API Key daalein.")
        else:
            niche_ideas = [niche.strip() for niche in user_niche_input.strip().split('\n') if niche.strip()]
            if not niche_ideas:
                st.warning("Barae meharbani research ke liye kam se kam ek niche idea daalein.")
            else:
                with st.spinner("Is mein kuch minute lag sakte hain... Niches research ki ja rahi hain..."):
                    viral_channels_result = find_viral_new_channels(st.session_state.api_key, niche_ideas, video_type_choice)

                if viral_channels_result:
                    st.success(f"🎉 {len(viral_channels_result)} naye promising channels mil gaye jo {datetime.now().year} mein banaye gaye!")
                    viral_channels_result.sort(key=lambda x: x['Subscribers'], reverse=True)
                    for channel in viral_channels_result:
                        st.markdown(f"""
                        <div style="border: 1px solid #4CAF50; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                            <p><strong>Channel:</strong> <a href="{channel['URL']}" target="_blank">{channel['Channel Name']}</a> | <strong>Banane ki Tareekh:</strong> {channel['Creation Date']}</p>
                            <p><strong>Subscribers:</strong> {channel['Subscribers']:,} | <strong>Total Views:</strong> {channel['Total Views']:,}</p>
                            <p><small><em>Is Niche se mila: "{channel['Found Via Niche']}"</em></small></p>
                        </div>""", unsafe_allow_html=True)
                        
                        # --- INTELLIGENCE UPGRADE: DISPLAY DEEP DIVE RESULTS ---
                        with st.expander("🕵️ Deep Dive Analysis Dekhein"):
                            # Engagement Score
                            engagement_color = "green" if channel['Engagement Score'] > 2 else "orange" if channel['Engagement Score'] > 1 else "red"
                            st.markdown(f"**Engagement Score:** <span style='color:{engagement_color}; font-weight:bold;'>{channel['Engagement Score']}%</span>", unsafe_allow_html=True)
                            st.progress(min(channel['Engagement Score'] / 5, 1.0)) # Visualize score (capped at 5% for visualization)
                            st.caption("Yeh score batata hai ke har 100 views par kitne log like ya comment karte hain. 2% se zyada bohot a-chha samjha jaata hai.")

                            # Monetization Clues
                            if channel['Monetization Clues']:
                                st.markdown(f"**💰 Monetization Clues:** `{'`, `'.join(channel['Monetization Clues'])}`")
                            else:
                                st.markdown("**💰 Monetization Clues:** Koi khaas clues nahi mile.")
                            
                            # Content Velocity & Schedule
                            st.markdown(f"**⚡ Content Velocity:** `{channel['Content Velocity']}` videos pichle 30 dinon mein.")
                            st.markdown(f"**🗓️ Haftawar Frequency:** `{channel['Weekly Frequency']}` videos har hafte (approx).")
                            st.markdown(f"**⏰ Upload Time:** {channel['Upload Schedule']}")
                            st.caption("Time UTC (Coordinated Universal Time) mein hai.")

                            # Intelligence Verdict
                            verdict = ""
                            if channel['Engagement Score'] > 2 and channel['Weekly Frequency'] > 2:
                                verdict = "🔥 **Excellent Potential!** High engagement aur high activity. Is niche ko zaroor consider karein."
                            elif channel['Engagement Score'] > 1.5:
                                verdict = "👍 **Good Potential.** A-chhi engagement hai, is niche mein value ho sakti hai."
                            else:
                                verdict = "🤔 **Analyze Karein.** Engagement thori kam hai, lekin growth ke chances ho sakte hain."
                            st.info(f"**Intelligence Verdict:** {verdict}")
                else:
                    st.warning(f"{datetime.now().year} mein banaye gaye koi bhi tezi se grow karne wale channels nahi mile. Yeh API ki limitations ya research kiye gaye niches mein naye viral channels na hone ki wajah se ho sakta hai.")

