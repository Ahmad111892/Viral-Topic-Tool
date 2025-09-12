import streamlit as st
import requests
from datetime import datetime, timedelta

# YouTube API Key
# It's better to use st.secrets for the API key in a real application
# For example: API_KEY = st.secrets.get("YOUTUBE_API_KEY", "Enter your API Key here")
API_KEY = "Enter your API Key here"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# Streamlit App Title
st.title("YouTube Viral Topics Tool")

# Input Fields
days = st.number_input("Enter Days to Search (1-30):", min_value=1, max_value=30, value=5)

# List of broader keywords
keywords = [
    "Affair Relationship Stories", "Reddit Update", "Reddit Relationship Advice", "Reddit Relationship",
    "Reddit Cheating", "AITA Update", "Open Marriage", "Open Relationship", "X BF Caught",
    "Stories Cheat", "X GF Reddit", "AskReddit Surviving Infidelity", "GurlCan Reddit",
    "Cheating Story Actually Happened", "Cheating Story Real", "True Cheating Story",
    "Reddit Cheating Story", "R/Surviving Infidelity", "Surviving Infidelity",
    "Reddit Marriage", "Wife Cheated I Can't Forgive", "Reddit AP", "Exposed Wife",
    "Cheat Exposed"
]

# Fetch Data Button
if st.button("Fetch Data"):
    if API_KEY == "Enter your API Key here" or not API_KEY:
        st.error("Please enter your YouTube API Key in the code.")
    else:
        try:
            # Calculate date range
            start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
            all_results = []

            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Iterate over the list of keywords
            for i, keyword in enumerate(keywords):
                status_text.text(f"Searching for keyword: {keyword} ({i+1}/{len(keywords)})")
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(keywords))

                # Define search parameters
                search_params = {
                    "part": "snippet",
                    "q": keyword,
                    "type": "video",
                    "order": "viewCount",
                    "publishedAfter": start_date,
                    "maxResults": 5, # Fetching 5 videos per keyword
                    "key": API_KEY,
                }

                # Fetch video data
                response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
                data = response.json()

                if "items" not in data or not data["items"]:
                    # No need to show a warning for every missed keyword, it clutters the UI
                    continue

                videos = data["items"]
                video_ids = [video["id"]["videoId"] for video in videos if "id" in video and "videoId" in video["id"]]
                channel_ids = [video["snippet"]["channelId"] for video in videos if "snippet" in video and "channelId" in video["snippet"]]

                if not video_ids or not channel_ids:
                    continue

                # Fetch video statistics
                stats_params = {"part": "statistics", "id": ",".join(video_ids), "key": API_KEY}
                stats_response = requests.get(YOUTUBE_VIDEO_URL, params=stats_params)
                stats_data = stats_response.json().get("items", [])

                # Fetch channel statistics
                channel_params = {"part": "statistics", "id": ",".join(channel_ids), "key": API_KEY}
                channel_response = requests.get(YOUTUBE_CHANNEL_URL, params=channel_params)
                channel_data = channel_response.json().get("items", [])
                
                # Create dictionaries for quick lookups
                video_stats_map = {item['id']: item['statistics'] for item in stats_data}
                channel_stats_map = {item['id']: item['statistics'] for item in channel_data}

                # Collect results
                for video in videos:
                    video_id = video['id']['videoId']
                    channel_id = video['snippet']['channelId']
                    
                    video_stat = video_stats_map.get(video_id)
                    channel_stat = channel_stats_map.get(channel_id)

                    if not (video_stat and channel_stat):
                        continue

                    subs = int(channel_stat.get("subscriberCount", 0))

                    if subs < 3000:  # Only include channels with fewer than 3,000 subscribers
                        title = video["snippet"].get("title", "N/A")
                        description = video["snippet"].get("description", "")[:200]
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        views = int(video_stat.get("viewCount", 0))

                        all_results.append({
                            "Title": title,
                            "Description": description,
                            "URL": video_url,
                            "Views": views,
                            "Subscribers": subs,
                            "Keyword": keyword
                        })
            
            # Clear progress bar and status text
            progress_bar.empty()
            status_text.empty()

            # Display results
            if all_results:
                st.success(f"Found {len(all_results)} results from small channels!")
                # Sort results by views
                all_results.sort(key=lambda x: x['Views'], reverse=True)
                
                for result in all_results:
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                            <p><strong>Title:</strong> {result['Title']}</p>
                            <p><strong>URL:</strong> <a href="{result['URL']}" target="_blank">Watch Video</a></p>
                            <p><strong>Views:</strong> {result['Views']:,} | <strong>Subscribers:</strong> {result['Subscribers']:,}</p>
                            <p><small><em>Found with keyword: "{result['Keyword']}"</em></small></p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.warning("No results found for channels with fewer than 3,000 subscribers.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
