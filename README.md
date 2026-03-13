# Assessing Communication Skills in YouTube Hair Styling Tutorials
**Submitted for: Machine Learning Specialist Role — Moxie Beauty**

---

## Defining "Communication Skills" Programmatically

Communication skills in tutorial content are a composite of **clarity**, **pace**, **structure**, and **engagement**. Rather than treating this as a single subjective score, the approach decomposes it into independently measurable signals extracted directly from the video's audio and transcript. Each signal acts as a proxy for a distinct dimension of effective communication.

---

## System Flow

```
YouTube URLs
     │
     ▼
[yt-dlp] — Audio Extraction (MP3)
     │
     ▼
[AssemblyAI] — Speech-to-Text Transcription
     │
     ▼
[Feature Extraction Layer]
 ├── spaCy        → Lexical richness & instructional density
 ├── Rule-based   → WPM, filler word ratio
 └── Transformers → Dominant emotional tone
     │
     ▼
[Google Gemini 2.5 Flash] — Expert LLM Commentary (per-dimension ratings)
     │
     ▼
CSV Output  (one row per video, all features + commentary)
```

---

## Features Extracted & Rationale

### 1. Words Per Minute (WPM) — *Speech Pace*
**Why:** Pace is the most direct signal of comprehension support. A creator speaking at 120–160 WPM gives viewers time to process and follow along. Too fast (>180 WPM) overwhelms; too slow (<100 WPM) loses attention. WPM is computed from total transcript word count divided by audio duration — a clean, interpretable metric with well-established benchmarks from public speaking research.

### 2. Filler Word Ratio — *Linguistic Clarity*
**Why:** Filler words ("um", "uh", "like", "you know", "basically") are a reliable indicator of verbal fluency and preparation. High filler density signals nervousness or lack of rehearsal, directly hurting how professional and confident a creator sounds. Measured as a percentage of total words using multi-word pattern matching (to avoid double-counting "you know").

### 3. Instructional Density — *Structural Quality*
**Why:** Hair tutorials are procedural content. Creators who use action verbs ("apply", "section", "clip") and sequence connectors ("first", "next", "then", "finally") produce clearer step-by-step guidance. Low instructional density suggests a casual, unstructured delivery that's harder for viewers to follow. spaCy's lemmatisation was used so that inflected forms ("brushing", "sectioned") are correctly captured.

### 4. Dominant Emotion — *Tone & Engagement*
**Why:** Tone drives engagement and trust. A consistently neutral or negative emotional tone makes a tutorial feel flat, while warmth (joy, positive surprise) keeps viewers motivated. The `j-hartmann/emotion-english-distilroberta-base` model (7-class classifier, CPU-friendly) was chosen because it's purpose-built for emotional nuance in everyday English text — more precise than sentiment polarity alone.

### 5. LLM Commentary Score (Gemini 2.5 Flash) — *Holistic Expert Review*
**Why:** Individual metrics capture isolated signals but miss holistic quality. Gemini 2.5 Flash synthesises all four numerical metrics alongside the raw transcript to produce dimension-by-dimension ratings (1–10) with quoted transcript evidence. This mirrors how a human evaluator would assess a tutorial — making the output both quantitative and explainable.

---

## Validation Approach

| Method | Purpose |
|--------|---------|
| **Correlation with viewer engagement** | Check if WPM, filler ratio, and instructional density correlate with like/comment ratios (via YouTube Data API) — a crowd-sourced proxy for communication quality |
| **Inter-rater agreement** | Have 3–5 human evaluators score the same 10 videos on a 1–10 scale; compute Spearman's ρ between human scores and the pipeline's composite score |
| **Ablation testing** | Remove features one at a time and measure the drop in correlation with human ratings to confirm each feature adds genuine signal |
| **Model calibration** | Apply the pipeline to creators at known skill extremes (viral MasterClass-style content vs. low-engagement beginner videos) and verify the scores separate correctly |

---

## Tools & Libraries

| Tool | Role | Why Chosen |
|------|------|------------|
| `yt-dlp` | Audio download | Most reliable YouTube downloader; actively maintained |
| `AssemblyAI` | Transcription | High accuracy on informal speech; word-level timestamps for future features |
| `spaCy` (`en_core_web_sm`) | NLP / lemmatisation | Fast, lightweight, ideal for CPU-only feature extraction |
| `transformers` (HuggingFace) | Emotion classification | Access to domain-specific fine-tuned models without training from scratch |
| `google-genai` (Gemini 2.5 Flash) | LLM commentary | State-of-the-art reasoning with structured output; cost-effective on pay-as-you-go |
| `pandas` | Output | Universal, interoperable CSV/DataFrame format |

---

## Key Design Decisions

- **Batch transcription over real-time:** AssemblyAI's async batch API was chosen for reliability and accuracy over streaming — we're not doing live analysis, so there's no trade-off.
- **CPU-only inference:** The emotion model runs on CPU (`device=-1`) so the pipeline works without GPU infrastructure, keeping it reproducible on any machine.
- **Separation of features from LLM:** Numerical features are deterministic and reproducible. The LLM commentary layer is additive — the CSV remains valid and usable even if the LLM call fails.
