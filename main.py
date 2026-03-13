"""
========================================================================
 YouTube Hair Tutorial — Communication Skills Extraction Pipeline
========================================================================

SETUP INSTRUCTIONS (run these once before executing the script):

  1. Create & activate a virtual environment:
       python -m venv env
       env\\Scripts\\activate           # Windows
       source env/bin/activate         # macOS / Linux

  2. Install dependencies:
       pip install -r requirements.txt

  3. Download the spaCy language model:
       python -m spacy download en_core_web_sm

  4. Fill in your API keys in the .env file:
       ASSEMBLYAI_API_KEY=your_key_here
       OPENAI_API_KEY=your_key_here

  5. Run the script:
       python main.py

OUTPUT:
  - output_features.csv  (quantitative features + LLM commentary per video)
========================================================================
"""

import os
import re
import tempfile
import shutil
from collections import Counter

# python-dotenv loads variables from .env into os.environ automatically.
# This keeps secrets out of the codebase and shell history.
from dotenv import load_dotenv
load_dotenv()   # reads .env from the current working directory

import yt_dlp
import assemblyai as aai
import spacy
import pandas as pd
from google import genai
from google.genai import types as genai_types
from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

# Filler words commonly found in spoken informal English
FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "literally",
    "right", "okay", "so", "anyway", "i mean", "kind of", "sort of",
    "actually", "you see", "well"
}

# Instructional verbs / sequence words typical in hair-styling tutorials
INSTRUCTIONAL_WORDS = {
    # Action verbs
    "apply", "brush", "section", "part", "clip", "comb", "blow",
    "curl", "straighten", "twist", "braid", "pin", "spray", "rinse",
    "wash", "dry", "wrap", "roll", "tease", "smooth", "detangle",
    "blowdry", "diffuse", "set", "hold", "loosen", "separate",
    # Sequence / transition words
    "first", "second", "third", "next", "then", "after", "before",
    "finally", "lastly", "start", "begin", "continue", "once",
    "step", "now", "also"
}

# ---------------------------------------------------------------------------
# Module 1 — Data Ingestion via yt-dlp
# ---------------------------------------------------------------------------

def download_audio(url: str, output_dir: str) -> tuple[str | None, float | None]:
    """
    Download only the audio track of a YouTube video using yt-dlp.

    Args:
        url        : Full YouTube URL.
        output_dir : Directory where the audio file will be saved.

    Returns:
        (audio_file_path, duration_seconds)  on success
        (None, None)                         on failure
    """
    # yt-dlp options — we request the best available audio and
    # post-process it into mp3 to keep the file small.
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",   # 96 kbps — sufficient for speech
            }
        ],
        # Template: <output_dir>/<video_id>.mp3
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        duration_seconds = info.get("duration")
        video_id         = info.get("id", "unknown")
        audio_path       = os.path.join(output_dir, f"{video_id}.mp3")

        if not os.path.exists(audio_path):
            # yt-dlp occasionally keeps the original extension; find the file
            for f in os.listdir(output_dir):
                if f.startswith(video_id):
                    audio_path = os.path.join(output_dir, f)
                    break

        print(f"  [✓] Downloaded audio → {audio_path}  ({duration_seconds}s)")
        return audio_path, float(duration_seconds)

    except Exception as exc:
        print(f"  [✗] Failed to download '{url}': {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Module 2 — Transcription via AssemblyAI
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str | None:
    """
    Upload an audio file to AssemblyAI and return the full text transcript.

    The API key is loaded from the .env file via ASSEMBLYAI_API_KEY.
    Word-level timestamps are available via transcript.words if needed
    for future features.

    Args:
        audio_path : Local path to the audio file.

    Returns:
        Full transcript string on success, or None on failure.
    """
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ASSEMBLYAI_API_KEY is not set. Add it to your .env file."
        )

    aai.settings.api_key = api_key

    try:
        print(f"  [~] Uploading {os.path.basename(audio_path)} to AssemblyAI …")
        transcriber = aai.Transcriber()
        transcript  = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"  [✗] Transcription failed: {transcript.error}")
            return None

        print(f"  [✓] Transcription complete. Words: {len(transcript.text.split())}")
        return transcript.text

    except Exception as exc:
        print(f"  [✗] AssemblyAI error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Module 3 — Feature Extraction Functions
# ---------------------------------------------------------------------------

# ── Feature 1 ── Words Per Minute (Speech Pace) ──────────────────────────

def calculate_wpm(transcript: str, duration_seconds: float) -> tuple[int, float]:
    """
    Estimate the speaker's pace in Words Per Minute.

    WPM = Total Words / (Duration in Minutes)

    Benchmark: 120–160 WPM = conversational; >180 = possibly rushing;
    <100 = overly slow or many pauses.

    Returns:
        (total_words, wpm)
    """
    words        = transcript.split()
    total_words  = len(words)
    duration_min = duration_seconds / 60.0
    wpm          = round(total_words / duration_min, 2) if duration_min > 0 else 0.0
    return total_words, wpm


# ── Feature 2 ── Filler Word Ratio (Linguistic Clarity) ──────────────────

def calculate_filler_ratio(transcript: str, total_words: int) -> float:
    """
    Measure how cluttered speech is by filler words.

    Multi-word fillers (e.g. "you know") are matched before single-word
    ones to avoid double-counting.

    Ratio = (Filler Word Count / Total Words) * 100  → %

    Benchmark: <2% = excellent; 2–5% = acceptable; >8% = may hurt clarity.

    Returns:
        filler_ratio (float, percentage)
    """
    if total_words == 0:
        return 0.0

    text_lower   = transcript.lower()
    filler_count = 0

    sorted_fillers = sorted(FILLER_WORDS, key=len, reverse=True)
    for filler in sorted_fillers:
        pattern = r"\b" + re.escape(filler) + r"\b"
        filler_count += len(re.findall(pattern, text_lower))

    return round((filler_count / total_words) * 100, 2)


# ── Feature 3 ── Instructional Density (Structure / Clarity) ─────────────

def calculate_instructional_density(
    transcript: str, total_words: int, nlp
) -> float:
    """
    Quantify how "tutorial-like" the speech is by detecting instructional
    verbs and sequence connectors via spaCy lemmatisation.

    Surface forms ("brushing", "brushed", "sections") all map to their
    base lemma so they are correctly captured.

    Density = (Instructional Tokens / Total Words) * 100  → %

    Benchmark: >8% = highly structured; 3–8% = moderate; <3% = conversational.

    Returns:
        instructional_density (float, percentage)
    """
    if total_words == 0:
        return 0.0

    doc = nlp(transcript.lower())
    instructional_count = sum(
        1 for token in doc if token.lemma_ in INSTRUCTIONAL_WORDS
    )

    return round((instructional_count / total_words) * 100, 2)


# ── Feature 4 ── Dominant Emotion (Tone / Engagement) ────────────────────

def calculate_dominant_emotion(transcript: str, emotion_classifier) -> str:
    """
    Infer the creator's dominant emotional tone using the fine-tuned model
    j-hartmann/emotion-english-distilroberta-base (CPU mode, device=-1).

    Strategy:
      1. Chunk the transcript into ~400-char sentence-bounded blocks.
      2. Run each chunk through the classifier.
      3. Return the most frequently predicted emotion label.

    Output labels: joy, sadness, anger, fear, disgust, surprise, neutral.

    Returns:
        dominant_emotion (str)
    """
    if not transcript or not transcript.strip():
        return "neutral"

    sentences = re.split(r"(?<=[.!?])\s+", transcript)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < 400:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
    if current.strip():
        chunks.append(current.strip())
    if not chunks:
        chunks = [transcript[:400]]

    emotion_votes = []
    for chunk in chunks:
        try:
            result = emotion_classifier(chunk, truncation=True, max_length=512)
            emotion_votes.append(result[0]["label"])
        except Exception:
            continue

    if not emotion_votes:
        return "neutral"

    return Counter(emotion_votes).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Module 4 — LLM Commentary Generator
# ---------------------------------------------------------------------------

def generate_llm_commentary(
    url: str,
    duration_seconds: float,
    total_words: int,
    wpm: float,
    filler_ratio: float,
    instructional_density: float,
    dominant_emotion: str,
    transcript: str,
    gemini_client,
) -> str:
    """
    Synthesise all extracted features + the raw transcript into a rich,
    expert-level commentary using Google Gemini (gemini-2.0-flash).

    The prompt provides the model with:
      - All 4 quantitative metrics with benchmark context baked in.
      - The full transcript (truncated to ~6 000 chars to stay within the
        free-tier context window while keeping API costs low).
      - A strict output format: per-dimension ratings (1-10) + recommendations.

    The Gemini SDK is configured via GOOGLE_API_KEY in the .env file.
    Falls back to a plain error string if the API call fails.

    Args:
        url                  : Source YouTube URL.
        duration_seconds     : Audio length in seconds.
        total_words          : Total word count.
        wpm                  : Words per minute.
        filler_ratio         : Filler word percentage.
        instructional_density: Instructional word percentage.
        dominant_emotion     : Most frequent predicted emotion label.
        transcript           : Full transcript text.
        gemini_client        : Configured google.genai.Client instance.

    Returns:
        commentary (str) — multi-paragraph expert review with ratings.
    """

    # Truncate the transcript to ~6 000 characters so the full prompt stays
    # well within Gemini 1.5 Flash's 1M token window and minimises latency.
    max_transcript_chars = 6000
    transcript_excerpt = transcript[:max_transcript_chars]
    if len(transcript) > max_transcript_chars:
        transcript_excerpt += "\n\n[... transcript truncated for brevity ...]"

    # ── Combined system + user prompt ─────────────────────────────────────
    # Gemini's generate_content() accepts a single string prompt, so we
    # embed the system instructions at the top and the data below.
    full_prompt = f"""You are a senior Media & Communication Analyst specialising in \
educational content evaluation. Your role is to assess YouTube hair-styling tutorial \
creators on their verbal communication skills using quantitative NLP metrics and the \
actual transcript as evidence.

When you write your commentary:
  • Be specific — quote short phrases from the transcript as evidence.
  • Be constructive — identify both strengths and improvement areas.
  • Use professional yet accessible language.
  • Do NOT pad with generic filler; every sentence must add value.

Output format (use these exact section headings, no deviations):
1. **Speech Pace (WPM)** — rated X/10
2. **Linguistic Clarity (Filler Ratio)** — rated X/10
3. **Instructional Structure (Density)** — rated X/10
4. **Emotional Tone & Engagement** — rated X/10
5. **Overall Communication Score: X/10**
6. **Key Recommendations** (3 bullet points)

=== VIDEO METADATA ===
URL              : {url}
Duration         : {int(duration_seconds // 60)}m {int(duration_seconds % 60)}s
Total Words      : {total_words}

=== QUANTITATIVE NLP METRICS ===
• Words Per Minute (WPM)      : {wpm}
  Benchmark → < 100 = slow | 120–160 = ideal | > 180 = rushed

• Filler Word Ratio           : {filler_ratio}%
  Benchmark → < 2% = excellent | 2–5% = acceptable | > 8% = distracting

• Instructional Density       : {instructional_density}%
  Benchmark → > 8% = highly structured | 3–8% = moderate | < 3% = conversational

• Dominant Emotion (ML model) : {dominant_emotion}
  Labels → joy / neutral / surprise / sadness / anger / fear / disgust

=== FULL TRANSCRIPT (evidence for your commentary) ===
{transcript_excerpt}

=== YOUR TASK ===
Write a detailed, attributed commentary following the exact output format above.
Quote specific short phrases from the transcript as evidence for each dimension.
Be honest, balanced, and actionable.
"""

    # Try models in order of capability — fall back if blocked or empty
    models_to_try = ["gemini-2.5-flash", "gemini-flash-latest"]

    for model_name in models_to_try:
        try:
            print(f"  [~] Generating LLM commentary via {model_name} …")

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=2000,
                ),
            )

            # Guard against blocked/empty responses
            text = getattr(response, "text", None)
            if not text:
                # Log finish_reason if available for debugging
                try:
                    reason = response.candidates[0].finish_reason
                    print(f"  [!] {model_name} returned empty response (finish_reason={reason}). Trying next model …")
                except Exception:
                    print(f"  [!] {model_name} returned empty response. Trying next model …")
                continue

            commentary = text.strip()
            print(f"  [✓] LLM commentary generated via {model_name}.")
            return commentary

        except Exception as exc:
            print(f"  [✗] Gemini API error with {model_name}: {exc}")
            if model_name != models_to_try[-1]:
                print(f"  [!] Retrying with next model …")
            continue

    return "Commentary unavailable (all Gemini models failed or returned empty responses)."


# ---------------------------------------------------------------------------
# Module 5 — Main Orchestration & CSV Output
# ---------------------------------------------------------------------------

def process_url(
    url: str,
    tmp_dir: str,
    nlp,
    emotion_classifier,
    gemini_client,
) -> dict | None:
    """
    Full pipeline for a single YouTube URL:
      download → transcribe → extract features → LLM commentary → return dict.

    Args:
        url                : YouTube video URL.
        tmp_dir            : Temporary directory for audio storage.
        nlp                : spaCy model instance.
        emotion_classifier : HuggingFace emotion pipeline.
        gemini_client      : Configured google.genai.Client instance.

    Returns:
        A dictionary of extracted features + LLM commentary, or None on failure.
    """
    print(f"\n{'─'*60}")
    print(f"Processing: {url}")
    print(f"{'─'*60}")

    # Step 1: Download audio + get duration from yt-dlp metadata
    audio_path, duration_seconds = download_audio(url, tmp_dir)
    if audio_path is None or duration_seconds is None:
        return None
    # After the None-guard above, both are guaranteed non-None.
    # The asserts below make that explicit so static type-checkers agree.
    assert audio_path is not None
    assert duration_seconds is not None

    # Step 2: Transcribe audio via AssemblyAI
    transcript = transcribe_audio(audio_path)
    if transcript is None:
        return None

    # Step 3: Extract quantitative communication features
    print("  [~] Extracting features …")
    total_words, wpm      = calculate_wpm(transcript, duration_seconds)
    filler_ratio          = calculate_filler_ratio(transcript, total_words)
    instructional_density = calculate_instructional_density(transcript, total_words, nlp)
    dominant_emotion      = calculate_dominant_emotion(transcript, emotion_classifier)

    # Step 4: Generate rich LLM commentary using all features + transcript
    commentary = generate_llm_commentary(
        url=url,
        duration_seconds=duration_seconds,
        total_words=total_words,
        wpm=wpm,
        filler_ratio=filler_ratio,
        instructional_density=instructional_density,
        dominant_emotion=dominant_emotion,
        transcript=transcript,
        gemini_client=gemini_client,
    )

    # Step 5: Clean up the downloaded audio file immediately after use
    try:
        os.remove(audio_path)
        print(f"  [✓] Cleaned up temp file: {os.path.basename(audio_path)}")
    except OSError:
        pass

    print(f"  [✓] Features → WPM={wpm} | Filler={filler_ratio}% | "
          f"Instructional={instructional_density}% | Emotion={dominant_emotion}")

    return {
        "Video_URL":              url,
        "Duration_Seconds":       duration_seconds,
        "Total_Words":            total_words,
        "WPM":                    wpm,
        "Filler_Word_Ratio":      filler_ratio,
        "Instructional_Density":  instructional_density,
        "Dominant_Emotion":       dominant_emotion,
        # The LLM commentary column — contains the full multi-paragraph review.
        # CSV wrapping may truncate display in Excel; open with a text editor
        # or use pd.read_csv() + print(df["LLM_Commentary"][i]) to read in full.
        "LLM_Commentary":         commentary,
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Validate required environment variables ──────────────────────────
    # Both keys are loaded from .env by load_dotenv() at the top of the file.
    for key in ("ASSEMBLYAI_API_KEY", "GOOGLE_API_KEY"):
        if not os.getenv(key):
            raise EnvironmentError(
                f"'{key}' is missing. Add it to your .env file and re-run."
            )

    # ── Sample YouTube URLs ──────────────────────────────────────────────
    # Replace these with the actual tutorial URLs you want to evaluate.
    YOUTUBE_URLS = [
        "https://youtu.be/dJliorOTMAY?si=gNMab0kCzjS-sjOz",   # Tutorial sample 1
        "https://youtu.be/hI0KiAK_HrQ?si=N4bxxo0-WjlU_6g9",   # Tutorial sample 2
        "https://youtu.be/IRYOU5Nx10c?si=Bf77xL94ReOOdrB-",   # Tutorial sample 3
    ]

    # ── One-time model / client loading ──────────────────────────────────
    print("Loading spaCy model (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm")

    print("Loading emotion classifier (CPU mode) …")
    emotion_classifier = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        device=-1,   # Force CPU — no GPU required
    )

    # Configure Gemini 2.5 Pro with the new google.genai SDK.
    print("Initialising Gemini client (gemini-2.5-pro) …")
    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # ── Temporary directory for audio downloads ──────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="moxie_audio_")
    print(f"\nTemporary audio directory: {tmp_dir}")

    # ── Process each URL ─────────────────────────────────────────────────
    all_results = []

    for url in YOUTUBE_URLS:
        result = process_url(url, tmp_dir, nlp, emotion_classifier, gemini_client)
        if result:
            all_results.append(result)

    # ── Clean up the temp directory itself ──────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n[✓] Removed temporary directory: {tmp_dir}")

    # ── Build DataFrame & export CSV ────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results, columns=[
            "Video_URL",
            "Duration_Seconds",
            "Total_Words",
            "WPM",
            "Filler_Word_Ratio",
            "Instructional_Density",
            "Dominant_Emotion",
            "LLM_Commentary",        # ← new column: rich expert review
        ])

        output_path = "output_features.csv"
        df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print(f"✅  Pipeline complete!")
        print(f"    Processed : {len(df)} video(s)")
        print(f"    Output    : {os.path.abspath(output_path)}")
        print(f"{'='*60}")

        # Print a summary table (without the long commentary column)
        print(df.drop(columns=["LLM_Commentary"]).to_string(index=False))

        # Print each commentary separately for readability in the terminal
        for i, row in df.iterrows():
            print(f"\n{'═'*60}")
            print(f"  LLM Commentary — Video {i+1}")
            print(f"  {row['Video_URL']}")
            print(f"{'═'*60}")
            print(row["LLM_Commentary"])

    else:
        print("\n[!] No videos were successfully processed. "
              "Check your API keys in .env and the YouTube URLs.")
