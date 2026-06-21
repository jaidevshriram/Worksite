from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from exa_py import Exa
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
if not EXA_API_KEY:
    raise RuntimeError("EXA_API_KEY not found in environment / .env")

exa = Exa(api_key=EXA_API_KEY)

app = FastAPI(title="Exa Search Proxy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/search")
async def search(q: str = Query(..., description="Search query")) -> dict:
    try:
        response = exa.search_and_contents(
            q,
            type="auto",
            num_results=3,
            highlights=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    results = []
    for r in response.results:
        highlights: list[str] = []
        if hasattr(r, "highlights") and r.highlights:
            highlights = list(r.highlights)
        results.append(
            {
                "title": r.title or "",
                "url": r.url or "",
                "highlights": highlights,
            }
        )

    return {"results": results}
