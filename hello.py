import streamlit as st
import requests
from datetime import datetime, timedelta

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
    and tracks which niche found the channel.
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
        
        # Add video duration filter based on user's choice
        if video_type == "Shorts Channel":
            search_params['videoDuration'] = 'short'
        elif video_type == "Long Video Channel":
            search_params['videoDuration'] = 'long'
            
        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
            if response.status_code == 200:
                # Get channel IDs from the video search results for this niche
                niche_channel_ids = {item["snippet"]["channelId"] for item in response.json().get("items", [])}
                
                # Filter out channels that have already been processed
                new_channel_ids = list(niche_channel_ids - processed_channel_ids)

                if not new_channel_ids:
                    continue

                # Fetch details for the new channels found
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

                            # Virality criteria (Average views check is REMOVED)
                            if subs > 1000 and views > 50000 and 5 < video_count < 100:
                                avg_views = views / video_count if video_count > 0 else 0
                                viral_channels.append({
                                    "Channel Name": channel["snippet"]["title"],
                                    "URL": f"https://www.youtube.com/channel/{channel['id']}",
                                    "Subscribers": subs,
                                    "Total Views": views,
                                    "Video Count": video_count,
                                    "Creation Date": published_date.strftime("%Y-%m-%d"),
                                    "Avg Views/Video": int(avg_views),
                                    "Found Via Niche": niche # Link the channel to the niche
                                })
                                processed_channel_ids.add(channel['id']) # Mark as processed
        except requests.exceptions.RequestException:
            continue
            
    progress_bar.empty()
    status_text.empty()
    return viral_channels

# --- Main App UI ---
st.title("🚀 YouTube Growth Toolkit")
st.markdown("Aapka all-in-one tool aasaani se viral topics aur naye niches dhoondne ke liye.")

# API Key Input (centralized and always visible)
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

# Tool selection using tabs for a cleaner look
tab1, tab2 = st.tabs(["🔍 Viral Topics Finder", "🚀 Niche Research Tool"])

# --- Tool 1: Viral Topics Finder ---
with tab1:
    st.header("Viral Topics Finder")
    st.info("Chhote channels se high-performing videos dhoondein taake aapko content ideas mil sakein.")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Pichhle kitne din search karne hain (1-30):", min_value=1, max_value=30, value=7)
    
    # NEW: User can input their own keywords
    default_keywords = "Affair Relationship Stories\nReddit Update\nReddit Relationship Advice\nReddit Cheating\nAITA Update\nCheating Story Real"
    user_keywords_input = st.text_area("Apne Keywords Yahan Daalein (har keyword nayi line mein):", value=default_keywords, height=150)

    if st.button("Viral Topics Dhoondein"):
        if not st.session_state.api_key:
            st.error("Barae meharbani sidebar mein apni YouTube API Key daalein.")
        else:
            keywords = [kw.strip() for kw in user_keywords_input.strip().split('\n') if kw.strip()]
            if not keywords:
                st.warning("Barae meharbani search karne ke liye kam se kam ek keyword daalein.")
            else:
                try:
                    # ... [Existing logic for Viral Topics Finder] ...
                    start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
                    all_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, keyword in enumerate(keywords):
                        status_text.text(f"Keyword search kiya ja raha hai: {keyword} ({i+1}/{len(keywords)})")
                        progress_bar.progress((i + 1) / len(keywords))
                        # ... API calls and processing logic remains the same ...
                        search_params = {"part": "snippet", "q": keyword, "type": "video", "order": "viewCount", "publishedAfter": start_date, "maxResults": 10, "key": st.session_state.api_key}
                        response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
                        data = response.json()
                        if "items" not in data or not data["items"]: continue
                        videos = data["items"]
                        video_ids = [v["id"]["videoId"] for v in videos if "id" in v and "videoId" in v["id"]]
                        channel_ids = [v["snippet"]["channelId"] for v in videos if "snippet" in v and "channelId" in v["snippet"]]
                        if not video_ids or not channel_ids: continue
                        stats_params = {"part": "statistics", "id": ",".join(video_ids), "key": st.session_state.api_key}
                        stats_response = requests.get(YOUTUBE_VIDEO_URL, params=stats_params)
                        stats_data = stats_response.json().get("items", [])
                        channel_params = {"part": "statistics", "id": ",".join(channel_ids), "key": st.session_state.api_key}
                        channel_response = requests.get(YOUTUBE_CHANNEL_URL, params=channel_params)
                        channel_data = channel_response.json().get("items", [])
                        video_stats_map = {item['id']: item['statistics'] for item in stats_data}
                        channel_stats_map = {item['id']: item['statistics'] for item in channel_data}
                        for video in videos:
                            video_id = video['id']['videoId']
                            channel_id = video['snippet']['channelId']
                            video_stat = video_stats_map.get(video_id)
                            channel_stat = channel_stats_map.get(channel_id)
                            if not (video_stat and channel_stat): continue
                            subs = int(channel_stat.get("subscriberCount", 0))
                            if subs < 5000: # Increased subscriber limit slightly
                                all_results.append({"Title": video["snippet"].get("title", "N/A"),"URL": f"https://www.youtube.com/watch?v={video_id}","Views": int(video_stat.get("viewCount", 0)),"Subscribers": subs,"Keyword": keyword})
                    
                    progress_bar.empty()
                    status_text.empty()
                    if all_results:
                        st.success(f"{len(all_results)} results chhote channels se mil gaye!")
                        all_results.sort(key=lambda x: x['Views'], reverse=True)
                        for result in all_results:
                            st.markdown(f"""<div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;"><p><strong>Title:</strong> {result['Title']}</p><p><strong>URL:</strong> <a href="{result['URL']}" target="_blank">Video Dekhein</a></p><p><strong>Views:</strong> {result['Views']:,} | <strong>Subscribers:</strong> {result['Subscribers']:,}</p><p><small><em>Is keyword se mila: "{result['Keyword']}"</em></small></p></div>""", unsafe_allow_html=True)
                    else:
                        st.warning("5,000 se kam subscribers wale channels se koi result nahi mila.")
                except Exception as e:
                    st.error(f"Ek error aayi: {e}")

# --- Tool 2: Niche Research Tool ---
with tab2:
    st.header("Niche Research Tool")
    st.info(f"{datetime.now().year} mein banaye gaye tezi se grow karne wale YouTube channels dhoondein.")

    # ADDED: Radio button for video type selection
    video_type_choice = st.radio(
        "Aap kis tarah ke channels dhoondna chahte hain?",
        ('Any', 'Shorts Channel', 'Long Video Channel'),
        horizontal=True,
        help="Shorts (1 min se kam), Long (20 min se zyada)."
    )
    
    # NEW: User can input their own niche ideas
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
                            <p><strong>Channel:</strong> <a href="{channel['URL']}" target="_blank">{channel['Channel Name']}</a></p>
                            <p><strong>Banane ki Tareekh:</strong> {channel['Creation Date']}</p>
                            <p><strong>Subscribers:</strong> {channel['Subscribers']:,} | <strong>Total Views:</strong> {channel['Total Views']:,} | <strong>Avg Views/Video:</strong> {channel['Avg Views/Video']:,}</p>
                            <p><small><em>Is Niche se mila: "{channel['Found Via Niche']}"</em></small></p>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.warning(f"{datetime.now().year} mein banaye gaye koi bhi tezi se grow karne wale channels nahi mile. Yeh API ki limitations ya research kiye gaye niches mein naye viral channels na hone ki wajah se ho sakta hai.")

