import os
import math
import requests
import csv
import time
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from io import StringIO
from datetime import datetime, timezone

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
    st.markdown("### 📊 Competition Scoring")
    st.markdown("""
    **Composite Score Breakdown:**
    - 🎯 Direct Competition: **50%**
    - 📝 Broad Competition: **20%**
    - 👑 Authority Score: **30%**
      - View Count: 40%
      - Engagement Rate: 25%
      - Channel Authority: 20%
      - Video Freshness: 15%
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Score Guide")
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
    """Fetches detailed stats for videos and channels with enhanced metrics."""
    try:
        video_ids = [item['id']['videoId'] for item in video_items[:10]]
        channel_ids = list(set([item['snippet']['channelId'] for item in video_items[:10]]))

        # Fetch video stats (views, likes, comments, published date)
        video_stats_request = youtube.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids)
        )
        video_stats = video_stats_request.execute().get('items', [])
        
        # Fetch channel stats (subscriber counts)
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

def calculate_engagement_rate(video_stats_item):
    """Calculates engagement rate (likes + comments) / views."""
    try:
        stats = video_stats_item.get('statistics', {})
        views = int(stats.get('viewCount', 0))
        likes = int(stats.get('likeCount', 0))
        comments = int(stats.get('commentCount', 0))
        
        if views == 0:
            return 0.0
        
        engagement = (likes + comments) / views
        return engagement
    except:
        return 0.0

def calculate_video_freshness_score(video_stats_item):
    """Calculates freshness score based on video age (newer = higher score)."""
    try:
        published_at = video_stats_item.get('snippet', {}).get('publishedAt')
        if not published_at:
            return 0.5  # Default middle score
        
        # Parse the published date
        published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        current_date = datetime.now(timezone.utc)
        
        # Calculate age in days
        age_days = (current_date - published_date).days
        
        # Scoring: Newer videos get higher scores
        # 0-30 days: 1.0
        # 31-90 days: 0.9
        # 91-180 days: 0.8
        # 181-365 days: 0.6
        # 1-2 years: 0.4
        # 2+ years: 0.2
        
        if age_days <= 30:
            return 1.0
        elif age_days <= 90:
            return 0.9
        elif age_days <= 180:
            return 0.8
        elif age_days <= 365:
            return 0.6
        elif age_days <= 730:
            return 0.4
        else:
            return 0.2
    except:
        return 0.5

def get_channel_authority_score(channel_stats, channel_id):
    """Calculates channel authority based on subscriber count."""
    try:
        for channel in channel_stats:
            if channel['id'] == channel_id:
                subs = int(channel.get('statistics', {}).get('subscriberCount', 0))
                
                # Normalize subscriber count with logarithm
                # 1K subs: ~0.38
                # 10K subs: ~0.50
                # 100K subs: ~0.63
                # 1M subs: ~0.75
                # 10M+ subs: ~0.88+
                
                if subs == 0:
                    return 0.0
                
                authority = math.log10(subs + 1) / 8
                return min(authority, 1.0)
        
        return 0.0
    except:
        return 0.0

def calculate_youtube_competition_score(keyword, video_items, video_stats, channel_stats):
    """Calculates the composite YouTube competition score with enhanced metrics."""
    
    # === METRIC 1: Direct Competition Score (Weight: 50%) ===
    exact_matches = 0
    for item in video_items:
        title = item['snippet']['title'].lower()
        if keyword.lower() in title:
            exact_matches += 1
    direct_score = (exact_matches / len(video_items)) if video_items else 0

    # === METRIC 2: Broad Competition Score (Weight: 20%) ===
    description_matches = 0
    for item in video_items:
        description = item['snippet']['description'].lower()
        if keyword.lower() in description:
            description_matches += 1
    broad_score = (description_matches / len(video_items)) if video_items else 0

    # === METRIC 3: Authority Score (Weight: 30%) - NOW WITH 4 SUB-METRICS ===
    
    # Sub-metric 3.1: Average Views (40% of authority score)
    avg_views = sum(int(v['statistics'].get('viewCount', 0)) for v in video_stats) / len(video_stats) if video_stats else 0
    views_score = (math.log10(avg_views + 1) / 8) if avg_views > 0 else 0
    views_score = min(views_score, 1.0)
    
    # Sub-metric 3.2: Average Engagement Rate (25% of authority score)
    engagement_rates = [calculate_engagement_rate(v) for v in video_stats]
    avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
    # Normalize engagement (typical good engagement is 3-5%)
    engagement_score = min(avg_engagement / 0.05, 1.0)  # 5% = max score
    
    # Sub-metric 3.3: Channel Authority (20% of authority score)
    channel_authorities = []
    for video in video_stats:
        channel_id = video.get('snippet', {}).get('channelId')
        if channel_id:
            authority = get_channel_authority_score(channel_stats, channel_id)
            channel_authorities.append(authority)
    avg_channel_authority = sum(channel_authorities) / len(channel_authorities) if channel_authorities else 0
    
    # Sub-metric 3.4: Video Freshness (15% of authority score)
    freshness_scores = [calculate_video_freshness_score(v) for v in video_stats]
    avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.5
    
    # Combine authority sub-metrics with their weights
    authority_score = (
        (views_score * 0.40) +           # 40% weight
        (engagement_score * 0.25) +       # 25% weight
        (avg_channel_authority * 0.20) +  # 20% weight
        (avg_freshness * 0.15)            # 15% weight
    )
    
    # === FINAL COMPOSITE SCORE ===
    final_score = (direct_score * 50) + (broad_score * 20) + (authority_score * 30)
    
    return {
        "direct_competition": round(direct_score, 2),
        "broad_competition": round(broad_score, 2),
        "authority_views": int(avg_views),
        "engagement_rate": round(avg_engagement * 100, 2),  # Convert to percentage
        "channel_authority": round(avg_channel_authority, 2),
        "video_freshness": round(avg_freshness, 2),
        "composite_score": round(final_score, 2),
        # Breakdown for display
        "views_contribution": round(views_score * 0.40 * 30, 2),
        "engagement_contribution": round(engagement_score * 0.25 * 30, 2),
        "authority_contribution": round(avg_channel_authority * 0.20 * 30, 2),
        "freshness_contribution": round(avg_freshness * 0.15 * 30, 2)
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

    # Extract CPC value and convert to float
    cpc_value = 0.00
    if ke_data and ke_data.get('cpc', {}).get('value'):
        try:
            cpc_value = float(ke_data.get('cpc', {}).get('value', 0.00))
        except (ValueError, TypeError):
            cpc_value = 0.00

    # Prepare result
    result = {
        "Keyword": keyword,
        "Competition Score": yt_competition.get('composite_score', 'N/A'),
        "Direct Competition %": f"{yt_competition.get('direct_competition', 0) * 100:.0f}%" if yt_competition.get('direct_competition') != 'N/A' else 'N/A',
        "Avg Top 10 Views": yt_competition.get('authority_views', 'N/A'),
        "Engagement Rate %": yt_competition.get('engagement_rate', 'N/A'),
        "Channel Authority": yt_competition.get('channel_authority', 'N/A'),
        "Video Freshness": yt_competition.get('video_freshness', 'N/A'),
        "Search Volume": ke_data.get('vol', 'N/A') if ke_data else 'N/A',
        "CPC": cpc_value,  # Store as float, not string
        # Store detailed breakdown for expander
        "_details": yt_competition
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = df[df['Competition Score'] != 'N/A']['Competition Score'].mean()
            st.metric("Avg Competition", f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A")
        
        with col2:
            low_comp = len(df[df['Competition Score'] < 30])
            st.metric("Low Competition", low_comp)
        
        with col3:
            high_cpc = len(df[df['CPC'] > 0.50])
            st.metric("High Monetization", high_cpc)
        
        with col4:
            avg_engagement = df[df['Engagement Rate %'] != 'N/A']['Engagement Rate %'].mean()
            st.metric("Avg Engagement", f"{avg_engagement:.2f}%" if not pd.isna(avg_engagement) else "N/A")
        
        # Display summary table (without details column)
        display_df = df.drop(columns=['_details'])
        
        st.dataframe(
            display_df.style.format({
                'CPC': '${:.2f}',
                'Search Volume': '{:,.0f}',
                'Avg Top 10 Views': '{:,.0f}',
                'Engagement Rate %': '{:.2f}%',
                'Channel Authority': '{:.2f}',
                'Video Freshness': '{:.2f}'
            }, na_rep='N/A'),
            width='stretch',
            hide_index=True
        )
        
        # Detailed cards with score breakdown
        st.markdown("### 🎯 Detailed Analysis")
        
        for result in results:
            score = result['Competition Score']
            cpc = result['CPC']
            
            if score != 'N/A':
                competition, advice, monetization = get_recommendation(score, cpc)
                
                with st.expander(f"**{result['Keyword']}** - {competition} (Score: {score})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Competition Breakdown**")
                        st.write(f"**Overall Score:** {score}/100")
                        st.write(f"├─ Direct Competition: {result['Direct Competition %']}")
                        st.write(f"└─ Broad Competition: {result.get('_details', {}).get('broad_competition', 'N/A')}")
                        
                        st.markdown("**👑 Authority Metrics**")
                        details = result.get('_details', {})
                        st.write(f"├─ Avg Views: {result['Avg Top 10 Views']:,}" if result['Avg Top 10 Views'] != 'N/A' else "├─ Avg Views: N/A")
                        st.write(f"├─ Engagement: {result['Engagement Rate %']}%" if result['Engagement Rate %'] != 'N/A' else "├─ Engagement: N/A")
                        st.write(f"├─ Channel Authority: {result['Channel Authority']}" if result['Channel Authority'] != 'N/A' else "├─ Channel Authority: N/A")
                        st.write(f"└─ Video Freshness: {result['Video Freshness']}" if result['Video Freshness'] != 'N/A' else "└─ Video Freshness: N/A")
                        
                        if details:
                            st.markdown("**🔢 Score Contributions**")
                            st.write(f"├─ Views: +{details.get('views_contribution', 0):.2f} pts")
                            st.write(f"├─ Engagement: +{details.get('engagement_contribution', 0):.2f} pts")
                            st.write(f"├─ Authority: +{details.get('authority_contribution', 0):.2f} pts")
                            st.write(f"└─ Freshness: +{details.get('freshness_contribution', 0):.2f} pts")
                    
                    with col2:
                        st.markdown("**💰 Commercial Intent**")
                        st.write(f"Search Volume: {result['Search Volume']:,}" if result['Search Volume'] != 'N/A' else "Search Volume: N/A")
                        st.write(f"CPC: ${cpc:.2f}")
                        st.write(f"Monetization: {monetization}")
                        
                        st.markdown("**💡 Recommendation**")
                        st.info(f"{advice}")
                        
                        # Visual score bar
                        st.markdown("**Competition Level**")
                        if score < 30:
                            st.progress(score/100, text=f"🟢 {score}/100 - Low Competition")
                        elif score < 60:
                            st.progress(score/100, text=f"🟡 {score}/100 - Medium Competition")
                        else:
                            st.progress(score/100, text=f"🔴 {score}/100 - High Competition")
        
        # Download button
        st.markdown("---")
        
        # Format dataframe for CSV download
        df_download = display_df.copy()
        df_download['CPC'] = df_download['CPC'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
        
        csv_buffer = StringIO()
        df_download.to_csv(csv_buffer, index=False)
        
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
