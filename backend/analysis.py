import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an empathetic journal analyst. Read the journal entry and score the writer's sentiment across six life categories.

For each category, assign a score from -1.0 (very negative) to 1.0 (very positive), or null if the entry doesn't mention that area at all.

Categories:
- family: References to family members, home life, parents, siblings
- relationships: All relationships — friendships, romantic connections, social bonds, loneliness vs connection
- finances: Money, work pay, expenses, financial stress or stability
- mental: Emotional wellbeing, stress, anxiety, happiness, mental health
- hobbies: Personal interests, creative pursuits, leisure activities, sports, entertainment
- academics_career: School, studies, work, career progress, professional goals

Also provide:
- overall_sentiment: A single score from -1.0 to 1.0 for the overall tone
- overall_mood: One word capturing the dominant mood (e.g. anxious, hopeful, content, frustrated, grateful)
- notes: 1-2 sentences summarising what you observed

Respond ONLY with valid JSON in this exact shape — no extra text, no markdown:
{
  "family": <float or null>,
  "relationships": <float or null>,
  "finances": <float or null>,
  "mental": <float or null>,
  "hobbies": <float or null>,
  "academics_career": <float or null>,
  "overall_sentiment": <float>,
  "overall_mood": "<string>",
  "notes": "<string>"
}"""


def analyse_journal_entry(text: str) -> dict:
    """Send journal text to Groq (Llama 3) and return scored analysis."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyse this journal entry:\n\n{text}"}
        ],
        temperature=0.2,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def analyse_image(image_bytes: bytes) -> dict:
    """OCR the image first, then analyse the text."""
    from ocr import extract_text_from_bytes
    raw_text = extract_text_from_bytes(image_bytes)
    if not raw_text:
        raise ValueError("No text could be extracted from the image.")
    scores = analyse_journal_entry(raw_text)
    scores["raw_text"] = raw_text
    return scores
