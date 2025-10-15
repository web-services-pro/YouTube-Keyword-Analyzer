import os
import math
import requests
import csv
import time
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from io import StringIO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="YouTube Keyword Analyzer",
    page_icon="🔍",
    layout="wide"
)

# --- TITLE ---
st.title("🎯 YouTube Keyword Competition Analyzer")
st.markdown("Analyze YouTube competition and commercial intent for your keywords")

# --- SIDEBAR: API KEYS ---
with st.sidebar:
    st.header("⚙️ API Configuration")
    
    # Try to load from secrets first, fallback to user input
    youtube_api_key = None
    ke_api_key = None
    
    try:
        # Try to get keys from secrets
        youtube_api_key = st.secrets.get("YOUTUBE_API_KEY", None)
        ke_api_key = st.secrets.get("KE_API_KEY", None)
    except (FileNotFoundError, KeyError):
        # Secrets file doesn't exist or keys not found - that's okay
        pass
    
    # If not in secrets, ask user to input
    if not youtube_api_key:
        youtube_api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="Get your key from Google Cloud Console"
        )
    else:
        st.success("✅ YouTube API Key loaded from secrets")
    
    if not ke_api_key:
        ke_api_key = st.text_input(
            "Keywords Everywhere API Key",
            type="password",
            help="Get your key from Keywords Everywhere"
        )
    else:
        st.success("✅ Keywords Everywhere API Key loaded from secrets")
    
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. Enter your API keys
    2. Input keywords (one per line)
    3. Click Analyze
    4. Download results as CSV
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Competition Scores")
    st.markdown("""
    - 🟢 **0-30**: Low competition
    - 🟡 **30-60**: Medium competition
    - 🔴 **60+**: High competition
    """)

# --- HELPER FUNCTIONS ---

@st.cache_resource
def get_youtube_client(api_key):
    """Creates and caches YouTube API client."""
    return build('youtube', 'v3', developerKey=api_key)

def get_youtube_search_results(youtube, keyword):
    """Fetches top 50 search results from YouTube for a keyword."""
    try:
        request = youtube.search().list(
            part="snippet",
            q=keyword,
            type="video",
            maxResults=50,
            order="relevance"
        )
        response = request.execute()
        return response.get('items', [])
    except Exception as e:
        st.error(f"YouTube API Error: {e}")
        return []

def get_video_and_channel_stats(youtube, video_items):
    """Fetches detailed stats for videos and channels."""
    try:
        video_ids = [item['id']['videoId'] for item in video_items[:10]]
        channel_ids = list(set([item['snippet']['channelId'] for item in video_items[:10]]))

        # Fetch video stats
        video_stats_request = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids)
        )
        video_stats = video_stats_request.execute().get('items', [])
        
        # Fetch channel stats
        channel_stats_request = youtube.channels().list(
            part="statistics",
            id=",".join(channel_ids)
        )
        channel_stats = channel_stats_request.execute().get('items', [])

        return video_stats, channel_stats
    except Exception as e:
        st.error(f"Stats API Error: {e}")
        return [], []

def get_keywords_everywhere_data(keyword, api_key):
    """Fetches Google Search Volume and CPC from Keywords Everywhere."""
    try:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        # Keywords Everywhere API uses POST with form data
        data = {
            'kw[]': [keyword],  # Array of keywords
            'country': 'us',
            'currency': 'usd',
            'dataSource': 'cli'  # Use 'cli' for combined GKP + Clickstream data
        }
        response = requests.post(
            'https://api.keywordseverywhere.com/v1/get_keyword_data',
            headers=headers,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                return result['data'][0]
        elif response.status_code == 401:
            st.error("❌ Keywords Everywhere API: Invalid API Key")
        elif response.status_code == 402:
            st.error("❌ Keywords Everywhere API: Insufficient credits or invalid subscription")
        else:
            st.warning(f"Keywords Everywhere API returned status code: {response.status_code}")
        
        return None
    except Exception as e:
        st.error(f"Keywords Everywhere API Error: {e}")
        return None

def calculate_youtube_competition_score(keyword, video_items, video_stats, channel_stats):
    """Calculates the composite YouTube competition score."""
    # Metric 1: Direct Competition Score (Weight: 50%)
    exact_matches = 0
    for item in video_items:
        title = item['snippet']['title'].lower()
        if keyword.lower() in title:
            exact_matches += 1
    direct_score = (exact_matches / len(video_items)) if video_items else 0

    # Metric 2: Broad Competition Score (Weight: 20%)
    description_matches = 0
    for item in video_items:
        description = item['snippet']['description'].lower()
        if keyword.lower() in description:
            description_matches += 1
    broad_score = (description_matches / len(video_items)) if video_items else 0

    # Metric 3: Authority Score (Weight: 30%)
    avg_views = sum(int(v['statistics'].get('viewCount', 0)) for v in video_stats) / len(video_stats) if video_stats else 0
    
    views_score = (math.log10(avg_views + 1) / 8) if avg_views > 0 else 0 
    
    # Combine scores
    final_score = (direct_score * 50) + (broad_score * 20) + (min(views_score, 1.0) * 30)
    
    return {
        "direct_competition": round(direct_score, 2),
        "authority_views": int(avg_views),
        "composite_score": round(final_score, 2)
    }

def get_recommendation(score, cpc):
    """Returns recommendation emoji and text."""
    if score < 30:
        competition = "🟢 Low"
        advice = "Good opportunity!"
    elif score < 60:
        competition = "🟡 Medium"
        advice = "High-quality video needed"
    else:
        competition = "🔴 High"
        advice = "Very competitive"
    
    monetization = "💰 Strong" if cpc > 0.50 else "💵 Weak"
    
    return competition, advice, monetization

def analyze_keyword(youtube, keyword, ke_api_key):
    """Analyzes a single keyword and returns results."""
    # Fetch data
    youtube_videos = get_youtube_search_results(youtube, keyword)
    ke_data = get_keywords_everywhere_data(keyword, ke_api_key)

    yt_competition = {}
    if youtube_videos:
        video_stats, channel_stats = get_video_and_channel_stats(youtube, youtube_videos)
        yt_competition = calculate_youtube_competition_score(keyword, youtube_videos, video_stats, channel_stats)

    # Prepare result
    result = {
        "Keyword": keyword,
        "Competition Score": yt_competition.get('composite_score', 'N/A'),
        "Direct Competition %": f"{yt_competition.get('direct_competition', 0) * 100:.0f}%" if yt_competition.get('direct_competition') != 'N/A' else 'N/A',
        "Avg Top 10 Views": yt_competition.get('authority_views', 'N/A'),
        "Search Volume": ke_data.get('vol', 'N/A') if ke_data else 'N/A',
        "CPC": ke_data.get('cpc', {}).get('value', 0.00) if ke_data else 0.00
    }

    return result

# --- MAIN APP ---

# Input area
st.markdown("### 📝 Enter Keywords to Analyze")
keywords_input = st.text_area(
    "Enter keywords (one per line)",
    height=150,
    placeholder="advanced sourdough baking techniques\nhow to build a gaming pc 2025\nbeginner landscape photography tutorial"
)

# Parse keywords
keywords_list = [k.strip() for k in keywords_input.split('\n') if k.strip()]

# Show keyword count
if keywords_list:
    st.info(f"📊 Ready to analyze **{len(keywords_list)}** keyword(s)")

# Analyze button
if st.button("🚀 Analyze Keywords", type="primary", disabled=not keywords_list):
    
    # Validate API keys
    if not youtube_api_key or not ke_api_key:
        st.error("⚠️ Please enter both API keys in the sidebar!")
        st.stop()
    
    # Initialize YouTube client
    try:
        youtube = get_youtube_client(youtube_api_key)
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        st.stop()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    # Analyze each keyword
    for i, keyword in enumerate(keywords_list):
        status_text.text(f"Analyzing: {keyword} ({i+1}/{len(keywords_list)})")
        
        try:
            result = analyze_keyword(youtube, keyword, ke_api_key)
            results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(keywords_list))
            
            # Rate limiting
            if i < len(keywords_list) - 1:
                time.sleep(1)
                
        except Exception as e:
            st.warning(f"Error analyzing '{keyword}': {e}")
            continue
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(1.0)
    
    # Display results
    if results:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = df[df['Competition Score'] != 'N/A']['Competition Score'].mean()
            st.metric("Avg Competition", f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A")
        
        with col2:
            low_comp = len(df[df['Competition Score'] < 30])
            st.metric("Low Competition", low_comp)
        
        with col3:
            high_cpc = len(df[df['CPC'] > 0.50])
            st.metric("High Monetization", high_cpc)
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Detailed cards
        st.markdown("### 🎯 Detailed Recommendations")
        
        for result in results:
            score = result['Competition Score']
            cpc = result['CPC']
            
            if score != 'N/A':
                competition, advice, monetization = get_recommendation(score, cpc)
                
                with st.expander(f"**{result['Keyword']}** - {competition}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**YouTube Competition**")
                        st.write(f"Score: {score}/100")
                        st.write(f"Direct Competition: {result['Direct Competition %']}")
                        st.write(f"Avg Views: {result['Avg Top 10 Views']:,}" if result['Avg Top 10 Views'] != 'N/A' else "Avg Views: N/A")
                    
                    with col2:
                        st.markdown("**Commercial Intent**")
                        st.write(f"Search Volume: {result['Search Volume']:,}" if result['Search Volume'] != 'N/A' else "Search Volume: N/A")
                        st.write(f"CPC: ${cpc:.2f}")
                        st.write(f"Monetization: {monetization}")
                    
                    st.info(f"💡 **Recommendation:** {advice}")
        
        # Download button
        st.markdown("---")
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="keyword_analysis_results.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit | "
    "Powered by YouTube Data API & Keywords Everywhere</div>",
    unsafe_allow_html=True
)
