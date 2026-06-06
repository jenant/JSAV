import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an empathetic journal analyst. Your job is to score a journal entry across eight life categories on a scale from -5.0 to +5.0.

━━━ SCORING SCALE ━━━
Scores range from -1.0 to +1.0. ±1.0 is reserved for only the most extreme, life-defining moments. Most entries should score in the ±0.1 to ±0.5 range.

  +0.9 to +1.0  Extraordinary positive — life-changing event, overwhelming joy (rare)
  +0.6 to +0.8  Strongly positive — explicit named emotion, significant positive event
  +0.3 to +0.5  Moderately positive — clear positive tone, good event mentioned
  +0.1 to +0.2  Mildly positive — brief mention, implied or inferred positivity
   0.0          Neutral — mentioned with no emotional charge
  -0.1 to -0.2  Mildly negative — brief or implied struggle
  -0.3 to -0.5  Moderately negative — clear negative tone, bad event mentioned
  -0.6 to -0.8  Strongly negative — explicit named negative emotion, significant setback
  -0.9 to -1.0  Extraordinary negative — crisis, trauma, devastating event (rare)

━━━ SCORING RULES ━━━
1. EXPLICITNESS: If the writer names a specific emotion (e.g. "I felt anxious", "I was so happy", "I'm exhausted"), score higher within the range. Vague or implied sentiment stays in ±0.1–0.2.
2. DEPTH: The more the entry focuses on a topic, the higher the absolute score. A single passing mention caps at ±0.2. A topic that dominates the entry can reach ±0.7. Only truly extraordinary entries reach ±0.9–1.0.
3. ALWAYS SCORE MENTAL HEALTH: Every entry affects mental_health — even if not mentioned directly. The writer's emotional tone, stress level, and overall mood always reflect their mental state. Never return null for mental_health.
4. INTERCONNECTION: Categories influence each other. If an event affects one category, consider its ripple effect on related ones. Examples:
   - "Went out for dinner with a friend" → friends (positive social bond), finances (spent money, slight negative), mental_health (social connection lifts mood)
   - "Failed an exam" → academics_career (negative), mental_health (stress, negative), possibly finances if scholarship is at risk
   - "Went for a run" → physical_health (positive), mental_health (exercise boosts mood, positive)
   - "Had a fight with my partner" → relationships (negative), mental_health (emotional strain, negative), possibly family if they're involved
5. NULL RULE: Only return null if a category has absolutely no connection — direct, indirect, or emotional — to the entry.
6. CONFIDENCE: If you are uncertain about a score, keep it closer to 0. Be conservative — it is better to underScore than overscore. Reserve high scores for clear, strong evidence.

━━━ CATEGORIES ━━━
- family: Family members, home life, parents, siblings, household dynamics
- friends: Platonic friendships, social events, feeling included or isolated
- relationships: Romantic relationships, dating, intimacy, heartbreak, attraction
- mental_health: Emotional wellbeing, mood, stress, anxiety, happiness, self-worth (ALWAYS scored)
- finances: Money, spending, income, financial stress or comfort
- academics_career: School, studies, work, career progress, professional achievements or setbacks
- hobbies: Personal interests, creative pursuits, leisure, sports, entertainment
- physical_health: Exercise, illness, energy, sleep quality, physical wellbeing

━━━ OUTPUT ━━━
Also provide:
- overall_sentiment: A single score from -1.0 to +1.0 for the overall tone of the entry
- overall_mood: One word capturing the dominant mood (e.g. anxious, hopeful, content, frustrated, grateful, excited, drained)
- notes: 2-4 sentences explaining the scores — mention which categories were affected, why, and any ripple effects you applied

Respond ONLY with valid JSON in this exact shape — no extra text, no markdown:
{
  "family": <float or null>,
  "friends": <float or null>,
  "relationships": <float or null>,
  "mental_health": <float>,
  "finances": <float or null>,
  "academics_career": <float or null>,
  "hobbies": <float or null>,
  "physical_health": <float or null>,
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
        max_tokens=768,
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
