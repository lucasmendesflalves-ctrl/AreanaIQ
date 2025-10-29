# main.py
import os
import time
import asyncio
import logging
import sqlite3
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# carregar .env se python-dotenv estiver instalado no ambiente
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LOG = logging.getLogger("arenaiq")
logging.basicConfig(level=logging.INFO)

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPORTS_API_KEY = os.getenv("SPORTS_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
DB_FILE = os.getenv("DB_FILE", "arena.db")
# ----------------------------------------

app = FastAPI(title="ArenaIQ API (Phase 2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção restrinja para seu domínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SQLite helpers ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        sport TEXT,
        home TEXT,
        away TEXT,
        score_home INTEGER,
        score_away INTEGER,
        status TEXT,
        start_ts INTEGER,
        raw TEXT,
        updated_ts INTEGER
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS probabilities (
        key TEXT PRIMARY KEY,
        summary TEXT,
        probabilities TEXT,
        confidence REAL,
        ts INTEGER
    )
    """)
    conn.commit()
    conn.close()

def db_upsert_match(m: Dict[str, Any]):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO matches(match_id,sport,home,away,score_home,score_away,status,start_ts,raw,updated_ts)
    VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (
        m.get("match_id"),
        m.get("sport"),
        m.get("home"),
        m.get("away"),
        m.get("score_home"),
        m.get("score_away"),
        m.get("status","scheduled"),
        m.get("start_ts"),
        json.dumps(m.get("raw") or {}),
        int(time.time())
    ))
    conn.commit()
    conn.close()

def db_get_recent_matches_for_team(team: str, limit: int=20):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    SELECT home,away,score_home,score_away FROM matches
    WHERE (home=? OR away=?) AND score_home IS NOT NULL
    ORDER BY updated_ts DESC LIMIT ?
    """, (team, team, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"home":r[0],"away":r[1],"score_home":r[2],"score_away":r[3]} for r in rows]

def db_save_probability(key: str, summary: str, probabilities: Dict[str,float], confidence: float):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO probabilities(key,summary,probabilities,confidence,ts)
    VALUES(?,?,?,?,?)
    """, (key, summary, json.dumps(probabilities), confidence, int(time.time())))
    conn.commit()
    conn.close()

def db_get_matches_all():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT match_id,sport,home,away,score_home,score_away,status,start_ts,raw FROM matches ORDER BY updated_ts DESC LIMIT 200")
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "match_id": r[0], "sport": r[1], "home": r[2], "away": r[3],
            "score_home": r[4], "score_away": r[5], "status": r[6], "start_ts": r[7], "raw": json.loads(r[8] or "{}")
        })
    return out

# ---------- Probability engine ----------
def simple_prob_engine(recent_matches: List[Dict[str,Any]], teamA: str, teamB: str):
    def team_form(team):
        played = [m for m in recent_matches if m.get("home")==team or m.get("away")==team][-15:]
        if not played:
            return 0.5
        score=0.0; weight=0.0; w=1.0
        for m in reversed(played):
            isHome = (m.get("home")==team)
            gf = m.get("score_home") if isHome else m.get("score_away")
            ga = m.get("score_away") if isHome else m.get("score_home")
            res = 0.5
            if gf is not None and ga is not None:
                if gf > ga: res = 1.0
                elif gf < ga: res = 0.0
            if isHome: res += 0.05
            score += res*w; weight += w; w += 0.3
        return score/weight
    formA = team_form(teamA); formB = team_form(teamB)
    rawA = formA*0.6 + (1-formB)*0.25
    rawB = formB*0.6 + (1-formA)*0.25
    diff = abs(rawA-rawB)
    draw = max(0.12, 0.28 - diff*0.25)
    denom = (rawA+rawB) if (rawA+rawB) > 0 else 1
    winA = rawA*(1-draw)/denom
    winB = rawB*(1-draw)/denom
    s = winA + winB + draw
    return {"home": winA/s, "draw": draw/s, "away": winB/s}

# ---------- OpenAI summary (optional) ----------
async def call_openai_summary(teamA: str, teamB: str, recent: List[Dict[str,Any]], headlines: List[str]):
    if not OPENAI_API_KEY:
        return None
    prompt = f"Produza uma análise curta (3-5 frases) em português sobre {teamA} x {teamB}. Considere estes resultados recentes:\n"
    for m in recent[-8:]:
        prompt += f"- {m.get('home')} {m.get('score_home')} x {m.get('score_away')} {m.get('away')}\n"
    if headlines:
        prompt += "Principais notícias:\n"
        for h in headlines[:5]:
            prompt += f"- {h}\n"
    prompt += "Seja conciso e explique fatores-chave e uma probabilidade estimada."
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            body = {
                "model": "gpt-4o-mini",
                "messages": [{"role":"system","content":"Você é um analista esportivo conciso em português."},
                             {"role":"user","content": prompt}],
                "max_tokens": 300,
                "temperature": 0.4,
            }
            r = await client.post(url, json=body, headers=headers)
            if r.status_code == 200:
                j = r.json()
                text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
                return text
            else:
                LOG.warning("OpenAI error: %s %s", r.status_code, r.text)
                return None
    except Exception as e:
        LOG.exception("OpenAI call error: %s", e)
        return None

# ---------- News fetch ----------
async def fetch_news(sport: str='football'):
    if not NEWS_API_KEY:
        LOG.info("No NEWS_API_KEY set, skipping news fetch")
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": sport, "language":"pt", "pageSize":8, "apiKey": NEWS_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            if r.status_code==200:
                j = r.json()
                return [a.get("title") for a in j.get("articles",[])]
            else:
                LOG.warning("NewsAPI non-200 %s %s", r.status_code, r.text)
                return []
    except Exception as e:
        LOG.exception("fetch_news error: %s", e)
        return []

# ---------- Sports updates (placeholder) ----------
async def fetch_sports_updates():
    # Esta função é um placeholder.
    # Você deve adaptar para a API de sua escolha (API-Football, SportMonks, etc.)
    # Se não tiver SPORTS_API_KEY, retorna lista vazia.
    if not SPORTS_API_KEY:
        LOG.info("No SPORTS_API_KEY set, skipping sports fetch")
        return []
    # Exemplo: você colocaria aqui requests para a API escolhida e
    # retornaria uma lista de objetos com campos: match_id, sport, home, away, score_home, score_away, status, start_ts, raw
    return []

# ---------- Background poller ----------
async def poller_task():
    while True:
        try:
            updates = await fetch_sports_updates()
            for u in updates:
                db_upsert_match(u)
                teamA = u.get("home"); teamB = u.get("away")
                recentA = db_get_recent_matches_for_team(teamA)
                recentB = db_get_recent_matches_for_team(teamB)
                recent = (recentA or []) + (recentB or [])
                probs = simple_prob_engine(recent, teamA, teamB)
                headlines = await fetch_news(u.get("sport","football"))
                summary = await call_openai_summary(teamA, teamB, recent, headlines) or f"Análise: {teamA} x {teamB}"
                key = f"{teamA}__{teamB}"
                db_save_probability(key, summary, probs, 0.6)
                await manager.broadcast({"type":"match_update","payload":u})
        except Exception as e:
            LOG.exception("Poller loop error: %s", e)
        await asyncio.sleep(max(10, POLL_INTERVAL))

# ---------- WebSocket manager ----------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except Exception:
            pass
    async def broadcast(self, message: Dict[str,Any]):
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                try:
                    self.active_connections.remove(conn)
                except Exception:
                    pass

manager = ConnectionManager()

# ---------- API models ----------
class AnalysisRequest(BaseModel):
    sport: str
    teamA: str
    teamB: str
    recentMatches: Optional[List[Dict[str,Any]]] = None
    headlines: Optional[List[str]] = None

# ---------- Startup ----------
@app.on_event("startup")
async def startup():
    init_db()
    app.state.poller = asyncio.create_task(poller_task())
    LOG.info("ArenaIQ Phase2 backend started")

# ---------- Endpoints ----------
@app.get("/api/health")
async def health():
    return {"status":"ok","ts":int(time.time())}

@app.post("/api/analysis")
async def analysis(req: AnalysisRequest):
    recent = []
    recent += db_get_recent_matches_for_team(req.teamA)
    recent += db_get_recent_matches_for_team(req.teamB)
    if req.recentMatches:
        recent = recent + req.recentMatches
    probs = simple_prob_engine(recent, req.teamA, req.teamB)
    headlines = req.headlines or []
    summary = await call_openai_summary(req.teamA, req.teamB, recent, headlines) or f"Análise sintética: {req.teamA} x {req.teamB}"
    key = f"{req.teamA}__{req.teamB}"
    db_save_probability(key, summary, probs, 0.6)
    return {"summary": summary, "probabilities": probs, "confidence": 0.6}

@app.post("/api/matches")
async def ingest_match(payload: Dict[str,Any]):
    if not payload.get("match_id"):
        payload["match_id"] = f"{payload.get('home')}_{payload.get('away')}_{int(time.time())}"
    db_upsert_match(payload)
    await manager.broadcast({"type":"match_update","payload":payload})
    return {"ok":True,"match_id":payload["match_id"]}

@app.get("/api/matches")
async def list_matches():
    return {"count": len(db_get_matches_all()), "matches": db_get_matches_all()}

@app.websocket("/ws/updates")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)

# ---------- Admin protection ----------
def require_admin(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/admin/add_match")
async def admin_add_match(request: Request):
    form = await request.form()
    admin_token = form.get("admin_token") or request.headers.get("x-admin-token")
    require_admin(admin_token)
    m = {
        "match_id": form.get("match_id") or f"{form.get('home')}_{form.get('away')}_{int(time.time())}",
        "sport": form.get("sport") or "futebol",
        "home": form.get("home"),
        "away": form.get("away"),
        "score_home": int(form.get("score_home")) if form.get("score_home") else None,
        "score_away": int(form.get("score_away")) if form.get("score_away") else None,
        "status": form.get("status") or "scheduled",
        "start_ts": int(form.get("start_ts")) if form.get("start_ts") else None,
        "raw": {}
    }
    db_upsert_match(m)
    await manager.broadcast({"type":"match_update","payload":m})
    return JSONResponse({"ok":True,"match":m})

@app.post("/admin/trigger_poll")
async def admin_trigger_poll(admin_token: str = Form(...)):
    require_admin(admin_token)
    updates = await fetch_sports_updates()
    for u in updates:
        db_upsert_match(u)
        await manager.broadcast({"type":"match_update","payload":u})
    return {"ok": True, "fetched": len(updates)}

@app.get("/admin/ui", response_class=HTMLResponse)
async def admin_ui():
    html = """
    <!doctype html><html><head><meta charset='utf-8'><title>ArenaIQ Admin</title></head><body>
    <h3>ArenaIQ Admin</h3>
    <form method='post' action='/admin/add_match'>
      <input name='admin_token' placeholder='ADMIN_TOKEN' style='width:300px'><br>
      <input name='sport' placeholder='sport' value='futebol'><br>
      <input name='home' placeholder='home'><br>
      <input name='away' placeholder='away'><br>
      <input name='score_home' placeholder='score_home'><br>
      <input name='score_away' placeholder='score_away'><br>
      <button type='submit'>Adicionar partida</button>
    </form>
    <hr>
    <form method='post' action='/admin/trigger_poll'>
      <input name='admin_token' placeholder='ADMIN_TOKEN' style='width:300px'><button type='submit'>Trigger Poll</button>
    </form>
    </body></html>
    """
    return HTMLResponse(content=html)
