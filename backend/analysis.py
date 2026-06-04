import os
import json
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

PROMPT = """You are an empathetic journal analyst. Your job is to read this handwritten journal entry and score the writer's sentiment across six life categories.

First, transcribe the handwriting exactly as written. Then analyse the content.

For each category, assign a score from -1.0 (very negative) to 1.0 (very positive), or null if the entry doesn't mention that area at all.

Categories:
- family: References to family members, home life, parents, siblings
- friends: Social life, friendships, social events, loneliness vs connection
- finances: Money, work pay, expenses, financial stress or stability
- mental: Emotional wellbeing, stress, anxiety, happiness, mental health
- romance: Romantic relationships, dating, intimacy, heartbreak
- academics_career: School, studies, work, career progress, professional goals

Also provide:
- raw_text: The full transcribed text from the handwriting
- overall_sentiment: A single score from -1.0 to 1.0 for the overall tone
- overall_mood: One word capturing the dominant mood (e.g. anxious, hopeful, content, frustrated, grateful)
- notes: 1-2 sentences summarising what you observed

Respond ONLY with valid JSON in this exact shape:
{
  "raw_text": "<transcribed text>",
  "family": <float or null>,
  "friends": <float or null>,
  "finances": <float or null>,
  "mental": <float or null>,
  "romance": <float or null>,
  "academics_career": <float or null>,
  "overall_sentiment": <float>,
  "overall_mood": "<string>",
  "notes": "<string>"
}"""

TEXT_PROMPT = """You are an empathetic journal analyst. Read this journal entry and score the writer's sentiment across six life categories.

For each category, assign a score from -1.0 (very negative) to 1.0 (very positive), or null if the entry doesn't mention that area at all.

Categories:
- family: References to family members, home life, parents, siblings
- friends: Social life, friendships, social events, loneliness vs connection
- finances: Money, work pay, expenses, financial stress or stability
- mental: Emotional wellbeing, stress, anxiety, happiness, mental health
- romance: Romantic relationships, dating, intimacy, heartbreak
- academics_career: School, studies, work, career progress, professional goals

Also provide:
- overall_sentiment: A single score from -1.0 to 1.0 for the overall tone
- overall_mood: One word capturing the dominant mood (e.g. anxious, hopeful, content, frustrated, grateful)
- notes: 1-2 sentences summarising what you observed

Respond ONLY with valid JSON in this exact shape:
{
  "family": <float or null>,
  "friends": <float or null>,
  "finances": <float or null>,
  "mental": <float or null>,
  "romance": <float or null>,
  "academics_career": <float or null>,
  "overall_sentiment": <float>,
  "overall_mood": "<string>",
  "notes": "<string>"
}

Journal entry:
"""


def analyse_image(image_bytes: bytes) -> dict:
    """Send a journal photo to Gemini — it transcribes the handwriting and scores all categories."""
    image = Image.open(io.BytesIO(image_bytes))
    response = model.generate_content([PROMPT, image])
    return _parse(response.text)


def analyse_journal_entry(text: str) -> dict:
    """Score a plain-text journal entry."""
    response = model.generate_content(TEXT_PROMPT + text)
    return _parse(response.text)


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)
