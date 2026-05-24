import os
import re
import requests
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from googleapiclient.discovery import build

app = FastAPI(title="YTA-AEO Dominance Suite Engine")
templates = Jinja2Templates(directory="templates")

# Remote Environment Keys Configuration Block
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")

# Client Initializer
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

def parse_iso_duration(duration_str):
    """Translates ISO 8601 formatting targets cleanly into standalone minute counts."""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 5.0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return (hours * 60) + minutes + (seconds / 60)

def fetch_estimated_volume(keyword):
    """Queries live search volumes leveraging your integrated RapidAPI wrapper database."""
    if not RAPIDAPI_KEY:
        return len(keyword) * 145  # Consistent structural backup proxy
        
    url = "https://youtube-keyword-research.p.rapidapi.com/keyword-stats"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "youtube-keyword-research.p.rapidapi.com"
    }
    try:
        response = requests.get(url, headers=headers, params={"keyword": keyword}, timeout=6)
        if response.status_code == 200:
            return int(response.json().get("monthly_volume", len(keyword) * 145))
    except Exception:
        pass
    return len(keyword) * 145

def get_longtail_suggestions(seed_keyword):
    """Harvests direct semantic expansions from Google's unauthenticated search suggest node."""
    url = f"https://suggestqueries.google.com/complete/search?client=youtube&ds=yt&q={seed_keyword}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            suggestions = re.findall(r'"([^"]*)"', response.text)
            return [s for s in suggestions if s and s != 'youtube' and s != seed_keyword]
    except Exception:
        pass
    return []

def calculate_mvi_and_competitors(keyword):
    """Processes the top ranking items to return an absolute Vulnerability Index."""
    if not youtube:
        return 0, []
    try:
        search_response = youtube.search().list(
            q=keyword, part='id,snippet', maxResults=5, type='video'
        ).execute()
        
        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
        if not video_ids:
            return 0, []
            
        video_details = youtube.videos().list(
            id=','.join(video_ids), part='snippet,statistics,contentDetails'
        ).execute()
        
        mvi_scores = []
        competitor_list = []
        
        for video in video_details.get('items', []):
            title = video['snippet']['title']
            channel_id = video['snippet']['channelId']
            channel_name = video['snippet']['channelTitle']
            published_at = datetime.strptime(video['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            
            age_days = max((datetime.utcnow() - published_at).days, 1)
            total_views = int(video['statistics'].get('viewCount', 0))
            duration_mins = parse_iso_duration(video['contentDetails']['duration'])
            
            # Sub-counting lookup to score domain authority factors
            try:
                ch_details = youtube.channels().list(id=channel_id, part='statistics').execute()
                channel_subs = int(ch_details['items'][0]['statistics'].get('subscriberCount', 0)) if ch_details.get('items') else 0
            except Exception:
                channel_subs = 15000
            
            # Core Algorithmic Vulnerability Formulations
            daily_velocity = total_views / age_days
            v_vel = 100 if daily_velocity < 15 else max(0, 100 - (daily_velocity * 2))
            v_fluff = 100 if duration_mins > 8.0 else max(0, (duration_mins / 8.0) * 100)
            v_auth = 100 if channel_subs < 15000 else (80 if channel_subs < 100000 else 30)
            
            video_mvi = (0.4 * v_vel) + (0.3 * v_fluff) + (0.3 * v_auth)
            mvi_scores.append(video_mvi)
            
            competitor_list.append({
                "title": title,
                "channel": channel_name,
                "views": f"{total_views:,}",
                "age_days": age_days,
                "duration": round(duration_mins, 1)
            })
            
        final_mvi = round(sum(mvi_scores) / len(mvi_scores), 2) if mvi_scores else 0
        return final_mvi, competitor_list
    except Exception:
        return 0, []

def generate_gemini_aeo_assets(keyword):
    """Instructs Gemini to structure content frameworks according to strict search layout metrics."""
    if not GEMINI_API_KEY:
        return "<p class='text-red-400'>Error: Gemini Operational Core Key is Missing from environment settings.</p>"
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
    
    system_prompt = (
        "You are an expert dual-engine SEO and AEO optimizer built to maximize visibility within search indexes.\n\n"
        f"Generate a definitive asset production array targeting exactly this keyword: '{keyword}'.\n"
        "Format your output with strict semantic HTML breaks (<br>, <strong>, and clear headers) using this template:\n\n"
        "<h2>[1. METADATA ENGINE PACK]</h2><br>"
        "<strong>Target Title:</strong> [Provide exact match keyword leading first, with a bracketed guide classifier tracking current guidelines]<br>"
        "<strong>AEO Snippet Description:</strong> [Provide an intensive 2-sentence direct functional answer block built to capture featured snippet blocks. Follow with systematic timestamp placeholder points]<br>"
        "<strong>Search Tags:</strong> [Provide 10 highly targeted comma-separated tags]<br><br>"
        "<h2>[2. HIGH-RETENTION CONTENT SCRIPT]</h2><br>"
        "<strong>The Solution Hook (0:00-0:15):</strong> [State problem directly, discard greeting filler]<br>"
        "<strong>The Direct Answer Block (0:15-0:30):</strong> [Clear explicit step summary matching caption crawlers]<br>"
        "<strong>Chronological Steps Walkthrough (0:30-End):</strong> [Numbered step directions focusing purely on teaching]<br><br>"
        "<h2>[3. VISUAL BLUEPRINT DESIGN]</h2><br>"
        "Provide a layout schematic for thumbnail assets (specifying left-aligned clean typography, face layout vectors, and branding icon layout rules)."
    )
    
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12)
        raw_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return raw_text.replace("\n", "<br>").replace("## ", "<h3 class='text-lg font-bold text-green-400 mt-4 mb-2'>")
    except Exception as e:
        return f"<p class='text-red-400'>Generation Fault: {str(e)}</p>"

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": False, "discovery": False})

@app.post("/analyze", response_class=HTMLResponse)
async def process_analysis_pipeline(request: Request, keyword: str = Form(...)):
    volume = fetch_estimated_volume(keyword)
    mvi_score, competitors = calculate_mvi_and_competitors(keyword)
    
    status, status_color, ai_assets = "HOLD", "text-yellow-400", ""
    if mvi_score >= 65:
        status = "LAUNCH / GREEN LIGHT"
        status_color = "text-green-400"
        ai_assets = generate_gemini_aeo_assets(keyword)
    else:
        status = "REJECT / HIGH COMPETITION INSULATION"
        status_color = "text-red-400"
        ai_assets = "<p class='text-gray-400'>Asset array bypassed. The competitive density floor prevents stable programmatic entry.</p>"
        
    return templates.TemplateResponse("index.html", {
        "request": request, "results": True, "discovery": False, "keyword": keyword,
        "volume": f"{volume:,}", "mvi_score": mvi_score, "status": status,
        "status_color": status_color, "competitors": competitors, "ai_assets": ai_assets
    })

@app.post("/discover", response_class=HTMLResponse)
async def process_bulk_discovery(request: Request, seed: str = Form(...)):
    raw_suggestions = get_longtail_suggestions(seed)
    discovered_gaps = []
    
    # Process and filter up to 8 real suggestion variants
    for phrase in raw_suggestions[:8]:
        mvi, _ = calculate_mvi_and_competitors(phrase)
        vol = fetch_estimated_volume(phrase)
        
        assessment = "EXCELLENT" if mvi >= 70 else ("VIABLE" if mvi >= 60 else "SKIPPABLE")
        badge_color = "bg-green-950 text-green-400 border-green-800" if mvi >= 70 else ("bg-yellow-950 text-yellow-400 border-yellow-800" if mvi >= 60 else "bg-red-950 text-red-400 border-red-800")
        
        discovered_gaps.append({
            "phrase": phrase, "mvi": mvi, "volume": f"{vol:,}",
            "assessment": assessment, "badge_color": badge_color
        })
        
    return templates.TemplateResponse("index.html", {
        "request": request, "results": False, "discovery": True, "seed": seed, "gaps": discovered_gaps
    })
