# main.py
import os, time, asyncio, logging, httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# carregar .env se python-dotenv estiver instalado
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LOG = logging.getLogger("arenaiq")
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

app = FastAPI(title="ArenaIQ API")

# CORS liberado em desenvolvimento (em produção restrinja)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

DB = {"matches": {}, "probabilities": {}, "news": {}}

class AnalysisRequest(BaseModel):
    sport: str
    teamA: str
    teamB: str
    recentMatches: Optional[List[Dict[str, Any]]] = None
    headlines: Optional[List[str]] = None

# WebSocket manager simples
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
    async def broadcast(self, message: Dict[str, Any]):
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                try:
                    self.active_connections.remove(conn)
                except Exception:
                    pass

manager = ConnectionManager()

# motor simples de probabilidades
def simple_prob_engine(recent_matches: List[Dict[str, Any]], teamA: str, teamB: str):
    def team_form(team):
        played = [m for m in recent_matches if m.get("home")==team or m.get("away")==team][-15:]
        if not played:
            return 0.5
        score = 0.0; w = 1.0; weight = 0.0
        for m in reversed(played):
            isHome = (m.get("home")==team)
            gf = m.get("score_home") if isHome else m.get("score_away")
            ga = m.get("score_away") if isHome else m.get("score_home")
            res = 0.5
            if gf is not None and ga is not None:
                if gf > ga: res = 1.0
                elif gf < ga: res = 0.0
            if isHome: res += 0.05
            score += res * w
            weight += w
            w += 0.3
        return score/weight
    formA = team_form(teamA); formB = team_form(teamB)
    rawA = formA*0.6 + (1-formB)*0.25
    rawB = formB*0.6 + (1-formA)*0.25
    diff = abs(rawA-rawB)
    draw = max(0.12, 0.28 - diff*0.25)
    denom = (rawA+rawB) if (rawA+rawB)>0 else 1
    winA = rawA*(1-draw)/denom
    winB = rawB*(1-draw)/denom
    s = winA + winB + draw
    return {"home": winA/s, "draw": draw/s, "away": winB/s}

# chamada simples ao OpenAI (async)
async def call_openai_summary(teamA: str, teamB: str, recent: List[Dict[str, Any]], headlines: List[str]):
    if not OPENAI_API_KEY:
        return None
    prompt = f"Produza uma análise curta (3-5 frases) em português sobre {teamA} x {teamB}. Considere estes resultados recentes:\\n"
    for m in recent[-8:]:
        prompt += f"- {m.get('home')} {m.get('score_home')} x {m.get('score_away')} {m.get('away')}\\n"
    if headlines:
        prompt += "Principais notícias:\\n"
        for h in headlines[:5]:
            prompt += f"- {h}\\n"
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
                "temperature": 0.4
            }
            r = await client.post(url, json=body, headers=headers)
            if r.status_code == 200:
                j = r.json()
                text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
                return text
            else:
                LOG.warning("OpenAI status: %s %s", r.status_code, r.text)
                return None
    except Exception as e:
        LOG.exception("OpenAI call error: %s", e)
        return None

# endpoints
@app.get("/api/health")
async def health():
    return {"status":"ok","ts":int(time.time())}

@app.post("/api/analysis")
async def analysis(req: AnalysisRequest):
    recent = []
    for m in DB["matches"].values():
        if m.get("home") in (req.teamA, req.teamB) or m.get("away") in (req.teamA, req.teamB):
            recent.append({"home": m.get("home"), "away": m.get("away"), "score_home": m.get("score_home"), "score_away": m.get("score_away")})
    if req.recentMatches:
        recent = (recent or []) + req.recentMatches
    probs = simple_prob_engine(recent, req.teamA, req.teamB)
    summary = await call_openai_summary(req.teamA, req.teamB, recent, req.headlines or [])
    if not summary:
        summary = f"Análise sintética: {req.teamA} vs {req.teamB}. Probabilidades estimadas (modelo leve)."
    key = f"{req.teamA}__{req.teamB}"
    DB["probabilities"][key] = {"ts":int(time.time()), "data":{"summary":summary,"probabilities":probs}}
    return {"summary":summary,"probabilities":probs,"confidence":0.6}

@app.post("/api/matches")
async def ingest_match(payload: Dict[str, Any]):
    match_id = str(payload.get("match_id") or f"{payload.get('home')}_{payload.get('away')}_{int(time.time())}")
    DB["matches"][match_id] = payload
    await manager.broadcast({"type":"match_update","payload":payload})
    return {"ok":True,"match_id":match_id}

@app.get("/api/matches")
async def list_matches():
    return {"count":len(DB["matches"]), "matches": list(DB["matches"].values())}

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
