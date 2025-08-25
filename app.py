#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI backend pro trendy klíčových slov (CZ/SK) + správa okruhů/klíčových slov
- SQLite (soubor data.db) pro groups/keywords
- Jednoduchá autorizace přes X-API-Key (env API_KEY)
- Endpoints:
  GET  /api/trending?geo=CZ
  GET  /api/interest?kw=...&geo=CZ&since=12m
  GET  /api/related?kw=...&geo=CZ&since=12m
  POST /api/batch_interest { geo, since, seeds[] }
  GET  /api/groups
  POST /api/groups { name }
  DELETE /api/groups/{id}
  POST /api/keywords { group_id, term }
  DELETE /api/keywords/{id}
Nasazení: uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os, time, sqlite3, json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pytrends.request import TrendReq
import numpy as np
import pandas as pd

API_KEY = os.environ.get("API_KEY", "CHANGE_ME")

app = FastAPI(title="Keyword Trends API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DB_PATH", "data.db")

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS keywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL,
        term TEXT NOT NULL,
        FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
    )""")
    con.commit()
    con.close()

init_db()

# ---- auth ----
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)

# ---- models ----
class GroupIn(BaseModel):
    name: str

class KeywordIn(BaseModel):
    group_id: int
    term: str

class BatchInterestIn(BaseModel):
    geo: str = "CZ"
    since: str = "12m"
    seeds: List[str]

# ---- helpers ----
CACHE: Dict[str, Dict] = {}
CACHE_TTL = 60 * 20

def cache_get(k: str):
    item = CACHE.get(k)
    if not item: return None
    if time.time() - item["ts"] > item.get("ttl", CACHE_TTL):
        CACHE.pop(k, None)
        return None
    return item["data"]

def cache_set(k: str, data, ttl: Optional[int] = None):
    CACHE[k] = {"data": data, "ts": time.time(), "ttl": ttl or CACHE_TTL}

def parse_since(s: str) -> str:
    s = (s or "12m").strip().lower()
    if s.endswith("m") and s[:-1].isdigit():
        m = int(s[:-1])
        return ("today 12-m" if m >= 12 else f"today {m}-m")
    if s.endswith("y") and s[:-1].isdigit():
        y = int(s[:-1])
        return ("today 5-y" if y >= 5 else f"today {y}-y")
    return "today 12-m"

def pytrends_client():
    return TrendReq(hl="cs-CZ", tz=120)

# ---- API: trends ----
@app.get("/api/trending")
def api_trending(geo: str = "CZ"):
    key = f"trending:{geo}"
    cached = cache_get(key)
    if cached: return {"items": cached}
    try:
        pt = pytrends_client()
        df = pt.trending_searches(pn=geo.lower())
        df.columns = ["query"]
        items = df["query"].dropna().head(30).tolist()
    except Exception as e:
        raise HTTPException(502, detail=f"pytrends error: {e}")
    cache_set(key, items, ttl=60*30)
    return {"items": items}

@app.get("/api/interest")
def api_interest(kw: str, geo: str = "CZ", since: str = "12m"):
    tf = parse_since(since)
    key = f"interest:{kw}:{geo}:{tf}"
    cached = cache_get(key)
    if cached: return cached
    try:
        pt = pytrends_client()
        pt.build_payload([kw], timeframe=tf, geo=geo)
        df = pt.interest_over_time()
        if df.empty or kw not in df.columns:
            raise ValueError("Empty interest")
        series = df[kw].fillna(0)
        labels = [str(i) for i in series.index]
        values = series.astype(int).tolist()
        payload = {"labels": labels, "values": values}
        cache_set(key, payload)
        return payload
    except Exception as e:
        raise HTTPException(502, detail=f"pytrends error: {e}")

@app.get("/api/related")
def api_related(kw: str, geo: str = "CZ", since: str = "12m"):
    tf = parse_since(since)
    key = f"related:{kw}:{geo}:{tf}"
    cached = cache_get(key)
    if cached: return cached
    try:
        pt = pytrends_client()
        pt.build_payload([kw], timeframe=tf, geo=geo)
        rq = pt.related_queries()
        block = rq.get(kw, {}) if isinstance(rq, dict) else {}
        out = {"top": [], "rising": []}
        if "top" in block and isinstance(block["top"], pd.DataFrame):
            out["top"] = block["top"][["query","value"]].fillna(0).to_dict(orient="records")
        if "rising" in block and isinstance(block["rising"], pd.DataFrame):
            out["rising"] = block["rising"][["query","value"]].fillna(0).to_dict(orient="records")
        cache_set(key, out, ttl=60*30)
        return out
    except Exception as e:
        raise HTTPException(502, detail=f"pytrends error: {e}")

@app.post("/api/batch_interest")
def api_batch_interest(payload: BatchInterestIn):
    tf = parse_since(payload.since)
    labels = None
    series_out = []
    for term in payload.seeds[:5]:
        try:
            pt = pytrends_client()
            pt.build_payload([term], timeframe=tf, geo=payload.geo)
            df = pt.interest_over_time()
            if df.empty or term not in df.columns: continue
            s = df[term].fillna(0)
            if labels is None: labels = [str(i) for i in s.index]
            series_out.append({"term": term, "values": s.astype(int).tolist()})
        except Exception:
            continue
    return {"labels": labels or [], "series": series_out}

# ---- API: groups/keywords ----
@app.get("/api/groups")
def list_groups():
    con = db(); cur = con.cursor()
    groups = [dict(r) for r in cur.execute("SELECT id,name FROM groups ORDER BY id DESC")]
    for g in groups:
        kws = [dict(r) for r in cur.execute("SELECT id, term FROM keywords WHERE group_id=? ORDER BY id DESC", (g["id"],))]
        g["keywords"] = kws
    con.close()
    return {"items": groups}

@app.post("/api/groups")
def create_group(g: GroupIn):
    con = db(); cur = con.cursor()
    cur.execute("INSERT INTO groups(name) VALUES(?)", (g.name.strip(),))
    con.commit()
    gid = cur.lastrowid
    con.close()
    return {"id": gid, "name": g.name}

@app.delete("/api/groups/{gid}")
def delete_group(gid: int):
    con = db(); cur = con.cursor()
    cur.execute("DELETE FROM keywords WHERE group_id=?", (gid,))
    cur.execute("DELETE FROM groups WHERE id=?", (gid,))
    con.commit(); con.close()
    return {"ok": True}

@app.post("/api/keywords")
def create_keyword(k: KeywordIn):
    con = db(); cur = con.cursor()
    cur.execute("SELECT id FROM groups WHERE id=?", (k.group_id,))
    if not cur.fetchone():
        con.close()
        raise HTTPException(404, "Skupina neexistuje")
    cur.execute("INSERT INTO keywords(group_id, term) VALUES(?,?)", (k.group_id, k.term.strip()))
    con.commit(); kid = cur.lastrowid; con.close()
    return {"id": kid, "group_id": k.group_id, "term": k.term}

@app.delete("/api/keywords/{kid}")
def delete_keyword(kid: int):
    con = db(); cur = con.cursor()
    cur.execute("DELETE FROM keywords WHERE id=?", (kid,))
    con.commit(); con.close()
    return {"ok": True}
