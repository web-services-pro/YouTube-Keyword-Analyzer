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
import google.generativeai as genai
import re

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
    gemini_api_key = None
    
    try:
        # Try to get keys from secrets
        youtube_api_key = st.secrets.get("YOUTUBE_API_KEY", None)
        ke_api_key = st.secrets.get("KE_API_KEY", None)
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
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
    st.markdown("### 🤖 AI Content Generation")
    
    if not gemini_api_key:
        gemini_api_key = st.text_input(
            "Google AI API Key (for Gemini)",
            type="password",
            help="Get your key from Google AI Studio"
        )
    else:
        st.success("✅ Gemini API Key loaded from secrets")
    
    st.markdown("---")
    st.markdown("### 📊 Competition Scoring")
    st.markdown("""
    **Composite Score Breakdown:**
    - 🎯 Direct Competition: **50%**
    - 🔍 Broad Competition: **20%**
    - 💡 Authority Score: **30%**
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

def get_youtube_autocomplete_suggestions(seed_keyword):
    """
    Fetches YouTube's autocomplete suggestions for a seed keyword.
    
    This uses an unofficial, undocumented API endpoint.
    """
    suggestions = []
    try:
        # The URL for the unofficial YouTube autocomplete API
        url = "http://suggestqueries.google.com/complete/search"
        params = {
            "client": "youtube",
            "ds": "yt",
            "q": seed_keyword
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        # The response is not valid JSON, it's like: window.google.ac.h([...])
        # We need to find the opening bracket and closing parenthesis to extract the JSON part.
        text = response.text
        start = text.find('[')
        end = text.rfind(']')
        
        if start != -1 and end != -1:
            # The first item in the list is the original keyword, so we skip it.
            # The second item is the list of suggestions.
            json_data = text[start:end+1]
            data = requests.utils.json.loads(json_data)
            if len(data) > 1:
                suggestions = [item[0] for item in data[1]]  # Suggestions are in the second element
        
        return suggestions

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch autocomplete for '{seed_keyword}': {e}")
        return []
    except Exception as e:
        st.warning(f"Error parsing autocomplete for '{seed_keyword}': {e}")
        return []

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
    """
    Fetches comprehensive data including CPC, Volume, Related Keywords, and PAA
    from Keywords Everywhere.
    """
    try:
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            'kw[]': [keyword],
            'country': 'us',
            'currency': 'usd',
            'dataSource': 'cli'  # 'cli' provides a rich set of data
        }
        response = requests.post(
            'https://api.keywordseverywhere.com/v1/get_keyword_data',
            headers=headers,
            data=data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                # Return the entire data object for the keyword
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

def generate_youtube_assets_with_gemini(api_key, main_keyword, keyword_cluster):
    """
    Uses Google's Gemini Pro to generate a complete YouTube asset package.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Create a string from the cluster for the prompt
        cluster_string = ", ".join(keyword_cluster[:10])  # Limit to 10 for prompt clarity

        # The prompt is crucial for getting structured output
        prompt = f"""You are a world-class YouTube SEO strategist and expert copywriter. Your goal is to create a complete, highly-optimized asset package for a YouTube video.

**Main Target Keyword:** "{main_keyword}"

**Supporting Keyword Cluster (use these concepts naturally):** "{cluster_string}"

**Your Task:** Generate 3 titles, 2 SEO-optimized description variations, and a list of video tags based on the provided keywords. Follow the output format below EXACTLY using the specified markdown headers.

---

### TITLES
(Generate 3 distinct, clickable, SEO-optimized title options. Each title must be under 70 characters and in Title Case.)
1. 
2. 
3. 

---

### DESCRIPTION 1
(Generate the first SEO-optimized description variation - under 1000 characters - with the following structure:
1. The first line should be compelling and hook-focused.
2. The second line is a Call to Action, like "➡️ Get my free guide here: [LINK]".
3. Write 3-4 paragraphs that naturally incorporate the main keyword and several supporting keywords from the cluster. Provide value and entice viewers to watch.
4. Add a list of 3-5 relevant hashtags based on the keyword cluster.
5. End with a repeat of the Call to Action.)

---

### DESCRIPTION 2
(Generate the second SEO-optimized description variation - under 1000 characters - with a different angle than Description 1. Use the same structure but vary the writing style and emphasis.)

---

### TAGS
(Provide a single, comma-separated list of all relevant keywords from the supporting keyword cluster. Include variations and related terms. Keep spaces in multi-word keywords. This should be ready to copy and paste directly into YouTube's tag field. Do NOT include hashtag symbols.)
"""
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return f"Error: Could not generate assets. {str(e)}"

def parse_gemini_output(text):
    """Parses the structured text from Gemini into a dictionary."""
    try:
        # Using regex to find content under each header - more flexible patterns
        titles_match = re.search(r"### TITLES\s*\n(.*?)(?=\n---|\n###|$)", text, re.DOTALL)
        desc1_match = re.search(r"### DESCRIPTION 1\s*\n(.*?)(?=\n---|\n###|$)", text, re.DOTALL)
        desc2_match = re.search(r"### DESCRIPTION 2\s*\n(.*?)(?=\n---|\n###|$)", text, re.DOTALL)
        tags_match = re.search(r"### TAGS\s*\n(.*?)$", text, re.DOTALL)
        
        return {
            "titles": titles_match.group(1).strip() if titles_match else "Could not parse titles.",
            "description1": desc1_match.group(1).strip() if desc1_match else "Could not parse description 1.",
            "description2": desc2_match.group(1).strip() if desc2_match else "Could not parse description 2.",
            "tags": tags_match.group(1).strip() if tags_match else "Could not parse tags."
        }
    except Exception as e:
        # Fallback if parsing fails
        st.warning(f"Parsing warning: {e}")
        return {
            "titles": "Could not parse titles.",
            "description1": text[:500] if len(text) > 500 else text,
            "description2": "Could not parse description 2.",
            "tags": "Could not parse tags."
        }

# --- MAIN APP ---

# Input area
st.markdown("### 🔍 Enter Seed Keywords")
keywords_input = st.text_area(
    "Enter 1-5 seed keywords (one per line)",
    height=100,
    placeholder="advanced sourdough baking\ngaming pc build\nlandscape photography"
)

# New Feature: Keyword Expansion Options
st.markdown("#### ✨ Keyword Expansion Options")
col1, col2, col3 = st.columns(3)
with col1:
    expand_autocomplete = st.checkbox("Auto-Suggest", help="Expand with YouTube's autocomplete suggestions (~10 per keyword).")
with col2:
    expand_related = st.checkbox("Related Keywords", help="Expand with semantically related keywords from Google.")
with col3:
    expand_paa = st.checkbox("Related Questions", help="Expand with 'People Also Ask' questions from Google.")

# Parse initial seed keywords
seed_keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

# Validate seed keyword count
if len(seed_keywords) > 5:
    st.error("⚠️ Please enter a maximum of 5 seed keywords!")
    st.stop()

keywords_list = list(seed_keywords)  # Start with the original list

# Logic to expand the list based on checkbox selections
if (expand_autocomplete or expand_related or expand_paa) and seed_keywords:
    
    # Use a set to store all keywords and avoid duplicates (but preserve order)
    all_keywords = list(seed_keywords)  # Start with seeds in order
    seen = set(seed_keywords)  # Track what we've seen
    
    suggest_status = st.empty()
    
    for i, seed in enumerate(seed_keywords):
        suggest_status.text(f"Expanding '{seed}' ({i+1}/{len(seed_keywords)})...")
        
        # 1. Autocomplete Expansion
        if expand_autocomplete:
            suggestions = get_youtube_autocomplete_suggestions(seed)
            if suggestions:
                for sugg in suggestions:
                    if sugg not in seen:
                        all_keywords.append(sugg)
                        seen.add(sugg)
        
        # 2. Related & PAA Expansion (uses one API call)
        if expand_related or expand_paa:
            # Check for API key before making the call
            if not ke_api_key:
                st.error("Keywords Everywhere API key is required for Related Keywords and People Also Ask expansion!")
                st.stop()
            
            ke_data = get_keywords_everywhere_data(seed, ke_api_key)
            if ke_data:
                # Add Related Keywords (LSI-style)
                if expand_related:
                    related_data = ke_data.get('related_keywords', {})
                    if isinstance(related_data, dict):
                        related = related_data.get('keywords', [])
                    elif isinstance(related_data, list):
                        related = related_data
                    else:
                        related = []
                    
                    if related:
                        for rel in related:
                            if rel not in seen:
                                all_keywords.append(rel)
                                seen.add(rel)
                
                # Add People Also Ask Questions
                if expand_paa:
                    paa_data = ke_data.get('people_also_ask', {})
                    if isinstance(paa_data, dict):
                        paa = paa_data.get('keywords', [])
                    elif isinstance(paa_data, list):
                        paa = paa_data
                    else:
                        paa = []
                    
                    if paa:
                        for question in paa:
                            if question not in seen:
                                all_keywords.append(question)
                                seen.add(question)
        
        time.sleep(0.5)  # Small delay to be polite to the APIs
    
    suggest_status.success(f"✅ Expanded from {len(seed_keywords)} seeds to {len(all_keywords)} unique keywords.")
    
    keywords_list = all_keywords

# Check if keyword list exceeds 99
if len(keywords_list) > 99:
    st.warning(f"⚠️ You have {len(keywords_list)} keywords, but the YouTube API limit is 99. Please remove {len(keywords_list) - 99} keyword(s) below.")
    
    # Allow user to deselect keywords
    st.markdown("#### ✂️ Trim Your Keyword List")
    selected_keywords = st.multiselect(
        f"Select up to 99 keywords to analyze (currently showing {len(keywords_list)} keywords)",
        options=keywords_list,
        default=keywords_list[:99]  # Pre-select first 99
    )
    
    if len(selected_keywords) > 99:
        st.error(f"❌ Please select exactly 99 or fewer keywords. You currently have {len(selected_keywords)} selected.")
        st.stop()
    
    keywords_list = selected_keywords

# Show keyword count
if keywords_list:
    st.info(f"📊 Ready to analyze **{len(keywords_list)}** keyword(s)")

# Analyze button
if st.button("🚀 Analyze Keywords", type="primary", disabled=not keywords_list):
    
    # Validate API keys
    if not youtube_api_key or not ke_api_key:
        st.error("⚠️ Please enter both YouTube and Keywords Everywhere API keys in the sidebar!")
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
    
    # Store results in session state for AI generation
    st.session_state['analysis_results'] = results
    
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
            use_container_width=True,
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
                        
                        st.markdown("**💡 Authority Metrics**")
                        details = result.get('_details', {})
                        st.write(f"├─ Avg Views: {result['Avg Top 10 Views']:,}" if result['Avg Top 10 Views'] != 'N/A' else "├─ Avg Views: N/A")
                        st.write(f"├─ Engagement: {result['Engagement Rate %']}%" if result['Engagement Rate %'] != 'N/A' else "├─ Engagement: N/A")
                        st.write(f"├─ Channel Authority: {result['Channel Authority']}" if result['Channel Authority'] != 'N/A' else "├─ Channel Authority: N/A")
                        st.write(f"└─ Video Freshness: {result['Video Freshness']}" if result['Video Freshness'] != 'N/A' else "└─ Video Freshness: N/A")
                        
                        if details:
                            st.markdown("**📢 Score Contributions**")
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

# --- AI ASSET GENERATION SECTION ---
# Only show if we have results in session state
if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
    results = st.session_state['analysis_results']
    
    st.markdown("---")
    st.markdown("## ✏️ AI Asset Generation")
    st.markdown("Select your winning keywords to generate optimized YouTube titles, descriptions, and tags.")
    
    # Filter for successful results to populate the selector
    successful_keywords = [res['Keyword'] for res in results if res['Competition Score'] != 'N/A']
    
    if successful_keywords:
        selected_winners = st.multiselect(
            "Select winning keywords to generate AI assets for:",
            options=successful_keywords,
            help="Choose keywords with good potential for AI-powered content generation."
        )

        if st.button("✨ Generate AI Assets for Selected Keywords", disabled=not selected_winners, type="primary"):
            if not gemini_api_key:
                st.error("⚠️ Please enter your Google AI (Gemini) API key in the sidebar!")
                st.stop()

            # Create tabs for each selected keyword's assets
            tabs = st.tabs([f"🚀 {winner}" for winner in selected_winners])

            for i, winner in enumerate(selected_winners):
                with tabs[i]:
                    with st.spinner(f"🤖 Gemini is crafting assets for '{winner}'..."):
                        
                        # Re-generate the cluster for the selected keyword
                        keyword_cluster = get_youtube_autocomplete_suggestions(winner)
                        if winner not in keyword_cluster:
                            keyword_cluster.insert(0, winner)
                        
                        # Call the Gemini function
                        generated_assets_text = generate_youtube_assets_with_gemini(gemini_api_key, winner, keyword_cluster)
                        
                        if "Error:" not in generated_assets_text:
                            # Parse the structured output
                            parsed_assets = parse_gemini_output(generated_assets_text)
                            
                            st.subheader("🎬 Generated Titles")
                            st.markdown(parsed_assets["titles"])
                            
                            st.subheader("📝 Generated Description - Option 1")
                            st.text_area(
                                "Description 1 (ready to copy)",
                                value=parsed_assets["description1"],
                                height=250,
                                key=f"desc1_{i}"
                            )
                            
                            st.subheader("📝 Generated Description - Option 2")
                            st.text_area(
                                "Description 2 (ready to copy)",
                                value=parsed_assets["description2"],
                                height=250,
                                key=f"desc2_{i}"
                            )
                            
                            st.subheader("🏷️ Generated Tags")
                            st.text_area(
                                "Tags (ready to copy - comma separated, no hashtags)",
                                value=parsed_assets["tags"],
                                height=100,
                                key=f"tags_{i}"
                            )
                        else:
                            st.error(generated_assets_text)
    else:
        st.info("No successful keyword analyses found. Please run an analysis first.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit | "
    "Powered by YouTube Data API, Keywords Everywhere & Google Gemini</div>",
    unsafe_allow_html=True
)
