import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import get_db, init_db, JournalEntry
from analysis import analyse_image, analyse_journal_entry

app = FastAPI(title="JSAV — Journal Sentiment Analysis Visualisation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
def on_startup():
    init_db()


# ── Entries ────────────────────────────────────────────────────────────────────

@app.get("/entries")
def list_entries(db: Session = Depends(get_db)):
    entries = db.query(JournalEntry).order_by(JournalEntry.entry_date).all()
    return [_serialise(e) for e in entries]


@app.get("/entries/{entry_id}")
def get_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return _serialise(entry)


@app.post("/entries/upload-image")
async def upload_image_entry(
    file: UploadFile = File(...),
    entry_date: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a journal photo — Gemini reads the handwriting and scores it in one step."""
    image_bytes = await file.read()

    try:
        scores = analyse_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Analysis failed: {e}")

    raw_text = scores.pop("raw_text", "")
    return await _create_entry_with_scores(raw_text, scores, image_bytes, file.filename, entry_date, db)


@app.post("/entries/text")
async def create_text_entry(
    text: str = Form(...),
    entry_date: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Create an entry from plain text (useful for testing without a photo)."""
    return await _create_entry(text, None, None, entry_date, db)


@app.delete("/entries/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    db.delete(entry)
    db.commit()
    return {"ok": True}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _create_entry(raw_text, image_bytes, filename, entry_date_str, db):
    try:
        scores = analyse_journal_entry(raw_text)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Analysis failed: {e}")
    return await _create_entry_with_scores(raw_text, scores, image_bytes, filename, entry_date_str, db)


async def _create_entry_with_scores(raw_text, scores, image_bytes, filename, entry_date_str, db):

    image_path = None
    if image_bytes and filename:
        dest = UPLOADS_DIR / f"{datetime.utcnow().timestamp()}_{filename}"
        dest.write_bytes(image_bytes)
        image_path = str(dest)

    parsed_date = datetime.utcnow()
    if entry_date_str:
        try:
            parsed_date = datetime.fromisoformat(entry_date_str)
        except ValueError:
            pass

    entry = JournalEntry(
        entry_date=parsed_date,
        image_path=image_path,
        raw_text=raw_text,
        overall_sentiment=scores.get("overall_sentiment", 0),
        overall_mood=scores.get("overall_mood"),
        family=scores.get("family"),
        relationships=scores.get("relationships"),
        finances=scores.get("finances"),
        mental=scores.get("mental"),
        hobbies=scores.get("hobbies"),
        academics_career=scores.get("academics_career"),
        analysis_notes=scores.get("notes"),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return _serialise(entry)


def _serialise(e: JournalEntry) -> dict:
    return {
        "id": e.id,
        "entry_date": e.entry_date.isoformat(),
        "created_at": e.created_at.isoformat(),
        "raw_text": e.raw_text,
        "overall_sentiment": e.overall_sentiment,
        "overall_mood": e.overall_mood,
        "scores": {
            "family": e.family,
            "relationships": e.relationships,
            "finances": e.finances,
            "mental": e.mental,
            "hobbies": e.hobbies,
            "academics_career": e.academics_career,
        },
        "analysis_notes": e.analysis_notes,
        "image_path": e.image_path,
    }
