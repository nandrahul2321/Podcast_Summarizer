# 🎙️ PodcastAI — Intelligent Podcast Summarizer

> **An AI-powered web application that transforms YouTube podcast videos into rich, structured summaries with sentiment analysis, confidence scoring, interactive Q&A, and exportable PDF reports.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Working Flow / Execution Flow](#5-working-flow--execution-flow)
6. [Features](#6-features)
7. [Technology Stack](#7-technology-stack)
8. [Folder Structure](#8-folder-structure)
9. [Installation & Setup](#9-installation--setup)
10. [Configuration Details](#10-configuration-details)
11. [API Endpoints](#11-api-endpoints)
12. [Database Schema](#12-database-schema)
13. [Deployment Process](#13-deployment-process)
14. [Screenshots](#14-screenshots)
15. [Future Enhancements](#15-future-enhancements)
16. [Conclusion](#16-conclusion)

---

## 1. Project Overview

**PodcastAI** is a full-stack web application built with **Flask** that allows users to paste any YouTube podcast/video URL and instantly receive:

- A **structured executive summary** with key takeaways (10-15 bullet points)
- **Sentiment & emotion analysis** for both the raw transcript and the generated summary
- **Confidence/accuracy metrics** evaluating transcription quality and summary faithfulness
- An **interactive Q&A chatbot** (PodcastAI Assistant) that answers questions strictly from the podcast content
- **PDF export** of the complete analysis report
- A **personal history library** to revisit all previously analyzed podcasts

The application leverages **Ollama** (local LLM inference) via **LangChain** for all NLP tasks, **YouTube Transcript API** for caption extraction, and **MongoDB** for persistent storage.

---

## 2. Problem Statement

Podcasts are one of the most popular content formats globally, yet they present a significant knowledge-extraction challenge:

| Challenge | Impact |
|---|---|
| **Long duration** — Episodes typically run 1-3+ hours | Users cannot quickly determine if content is relevant |
| **Audio-only format** — No scannable text or headings | Searching for specific insights requires listening end-to-end |
| **Lack of structured notes** — Raw transcripts are noisy and unformatted | Professionals cannot efficiently archive or reference key takeaways |
| **Emotional context is lost** — Written summaries strip tone and sentiment | Readers miss whether the speaker was excited, critical, or neutral |
| **No interactive exploration** — Static summaries don't allow follow-up questions | Users must re-listen to clarify specific points |

**PodcastAI** solves all of these problems through automated NLP-powered analysis that condenses hours of audio into a digestible, actionable, and interactive intelligence report — in minutes.

---

## 3. Objectives

1. **Automate Transcript Extraction** — Seamlessly retrieve captions from any YouTube video (manual or auto-generated, with multi-language and proxy support).
2. **Generate Structured Summaries** — Use a distributed chunked summarization pipeline to produce a concise summary paragraph and 10-15 key takeaways.
3. **Analyze Sentiment & Emotion** — Classify the emotional tone (Excited, Serious, Motivational, Sad, Informative, Angry, Neutral) and sentiment polarity (Positive, Negative, Neutral) with confidence scores.
4. **Measure Confidence/Accuracy** — Provide LLM-evaluated quality scores for both the transcription and the generated summary.
5. **Enable Interactive Q&A** — Allow users to ask natural-language questions about the podcast content with context-grounded answers.
6. **Export Reports** — Generate downloadable PDF reports containing the summary, sentiment data, and accuracy metrics.
7. **Maintain History** — Persist all analyses per user for future retrieval and re-analysis.
8. **Secure Access** — Implement user authentication with hashed passwords and session management.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Browser)                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Landing  │  │  Login   │  │  Signup  │  │Dashboard │  │ History  │     │
│  │  Page    │  │  Page    │  │  Page    │  │  (Home)  │  │  Page    │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│        HTML / CSS / JavaScript (Inter font, CSS Variables, Animations)      │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │ HTTP (GET/POST/JSON)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLASK APPLICATION SERVER                          │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  app.py — Main Entry Point                                          │   │
│  │  • Flask app initialization & secret key                            │   │
│  │  • Flask-Login setup (UserMixin, session protection)                │   │
│  │  • Authentication routes: /, /signup, /login, /logout               │   │
│  │  • Blueprint registration (home_bp)                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │  home.py — Core Business Logic Blueprint                            │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │   │
│  │  │  Transcript     │  │  Summarization  │  │  Sentiment &     │   │   │
│  │  │  Extraction     │  │  Engine         │  │  Emotion Engine  │   │   │
│  │  │  (YT Captions)  │  │  (Chunked LLM)  │  │  (LLM JSON)     │   │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │   │
│  │           │                    │                     │              │   │
│  │  ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼─────────┐   │   │
│  │  │  Accuracy       │  │  PDF           │  │  Chat Service    │   │   │
│  │  │  Calculator     │  │  Generator     │  │  (Q&A Bot)       │   │   │
│  │  │  (LLM Scoring)  │  │  (ReportLab)   │  │  (LangChain)     │   │   │
│  │  └─────────────────┘  └────────────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
└─────────────────────────────────────┼───────────────────────────────────────┘
          │                           │                          │
          ▼                           ▼                          ▼
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  YouTube         │   │  Ollama LLM Server   │   │  MongoDB             │
│  Transcript API  │   │  (localhost:11434)    │   │  (localhost:27017)   │
│                  │   │                      │   │                      │
│  • Caption fetch │   │  • llama3.2:1b       │   │  DB: Podcast_        │
│  • Multi-lang    │   │  • gpt-oss:latest    │   │      Summarizer      │
│  • Proxy support │   │  • Any Ollama model  │   │  Collections:        │
│                  │   │                      │   │  • users              │
│                  │   │  LangChain Ollama    │   │  • summaries          │
│                  │   │  Integration         │   │  • history            │
└──────────────────┘   └──────────────────────┘   └──────────────────────┘
```

### Architecture Highlights

| Layer | Technology | Responsibility |
|---|---|---|
| **Presentation** | HTML5, CSS3, JavaScript | Responsive UI with sidebar layout, animations, and interactive chat |
| **Application** | Flask + Blueprints | Request routing, authentication, session management, template rendering |
| **NLP Engine** | LangChain + Ollama (ChatOllama) | Summarization, sentiment analysis, accuracy scoring, Q&A chat |
| **Data** | MongoDB (PyMongo) | User accounts, cached summaries, per-user analysis history |
| **External** | YouTube Transcript API | Caption extraction with proxy and multi-language support |
| **Export** | ReportLab | PDF report generation with styled tables and formatted content |

---

## 5. Working Flow / Execution Flow

```
┌───────────┐     ┌───────────┐     ┌──────────────────┐
│  User     │     │  User     │     │  User pastes     │
│  visits   │────▶│  signs up │────▶│  YouTube URL on  │
│  landing  │     │  or logs  │     │  Dashboard       │
│  page     │     │  in       │     │                  │
└───────────┘     └───────────┘     └────────┬─────────┘
                                              │
                                              ▼
                                   ┌──────────────────┐
                                   │  Check MongoDB   │
                                   │  cache for this  │
                                   │  video_id        │
                                   └─────┬──────┬─────┘
                                   Cached│      │Not Cached
                                         ▼      ▼
                              ┌──────────┐  ┌──────────────────────┐
                              │  Return  │  │  Fetch YouTube       │
                              │  cached  │  │  transcript          │
                              │  summary │  │  (captions API)      │
                              └──────────┘  └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │  Split text into     │
                                            │  chunks (3000 chars, │
                                            │  200 overlap)        │
                                            └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │  Async parallel      │
                                            │  summarize each      │
                                            │  chunk via LLM       │
                                            └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │  Combine chunk       │
                                            │  summaries → Final   │
                                            │  LLM call for        │
                                            │  executive summary   │
                                            │  + key takeaways     │
                                            └──────────┬───────────┘
                                                       │
                                          ┌────────────┼────────────┐
                                          ▼            ▼            ▼
                                ┌──────────────┐ ┌──────────┐ ┌──────────┐
                                │  Sentiment & │ │ Accuracy │ │  Store   │
                                │  Emotion     │ │ Scoring  │ │  in      │
                                │  Analysis    │ │ (LLM)    │ │  MongoDB │
                                └──────────────┘ └──────────┘ └──────────┘
                                          │            │            │
                                          └────────────┼────────────┘
                                                       ▼
                                            ┌──────────────────────┐
                                            │  Render Dashboard    │
                                            │  with Summary,       │
                                            │  Sentiment, Accuracy │
                                            │  + Chat + PDF Export │
                                            └──────────────────────┘
```

### Step-by-Step Execution

| Step | Action | Component |
|---|---|---|
| 1 | User visits the landing page | `index.html` via `/` route |
| 2 | User signs up or logs in | `app.py` → `/signup` or `/login` |
| 3 | User pastes a YouTube URL and clicks "Start Analysis" | `home.html` form → POST to `/home` |
| 4 | Server extracts video ID from URL | `get_video_id()` with regex patterns |
| 5 | Server checks MongoDB cache for existing summary | `summaries_collection.find_one()` |
| 6 | If not cached: fetch transcript via YouTube Transcript API | `fetch_available_captions()` |
| 7 | Text is split into 3000-character chunks | `RecursiveCharacterTextSplitter` |
| 8 | Each chunk is summarized asynchronously via Ollama LLM | `summarize_chunk()` with `asyncio.gather()` |
| 9 | Chunk summaries are combined and a final summary is generated | `generate_distributed_summary_async()` |
| 10 | Sentiment & emotion analysis runs on transcript and summary | `analyze_sentiment_emotion_async()` |
| 11 | Accuracy/confidence scores are calculated | `calculate_accuracy_scores_async()` |
| 12 | All results are stored in MongoDB | `summaries_collection.insert_one()` |
| 13 | User's history is updated | `history_collection.update_one()` with upsert |
| 14 | Dashboard renders with full results | `home.html` with Jinja2 templating |
| 15 | User can ask questions via interactive chat | `/chat` → `ChatService.get_chat_response()` |
| 16 | User can download a PDF report | `/download-pdf/<video_id>` → `generate_pdf()` |

---

## 6. Features

### 6.1 Core Features

| Feature | Description |
|---|---|
| **🔐 User Authentication** | Secure signup/login with hashed passwords (Werkzeug), Flask-Login session management, and strong session protection |
| **🎯 Smart URL Parsing** | Supports multiple YouTube URL formats: standard watch, youtu.be short links, /shorts/, /embed/, /live/, and raw 11-character video IDs |
| **📝 Transcript Extraction** | Multi-language support with configurable preferred languages, automatic/manual caption detection, optional proxy configuration (Webshare or generic), and translation support |
| **🤖 Distributed AI Summarization** | Chunked text splitting (3000 chars, 200 overlap) → async parallel chunk summarization → final combined executive summary with 10-15 key takeaways |
| **📊 Sentiment & Emotion Analysis** | LLM-powered analysis returning sentiment (Positive/Negative/Neutral), sentiment score (0-100%), dominant emotion (Excited, Serious, Motivational, Sad, Informative, Angry, Neutral), and emotion confidence |
| **📈 Confidence Metrics** | LLM-evaluated transcription quality score and summary faithfulness score (0-100%) |
| **💬 Interactive Q&A Chat** | Context-grounded chatbot that answers questions only from the podcast summary, with polite greeting handling and strict no-hallucination rules |
| **📄 PDF Export** | Professionally styled PDF reports with ReportLab, including summary, key takeaways, sentiment tables, and accuracy tables |
| **📚 Analysis History** | Per-user history tracking with timestamps, sortable by most recent, and one-click re-analysis |
| **⚡ Intelligent Caching** | MongoDB-based caching keyed by video ID — repeat requests return instant results without re-processing |

### 6.2 UI/UX Features

| Feature | Description |
|---|---|
| **🎨 Premium Design System** | Custom CSS design system with CSS variables, Inter font, indigo accent palette, glassmorphism effects |
| **📱 Sidebar Dashboard** | Persistent sidebar navigation with workspace and library sections, user profile avatar, and logout |
| **✨ Animations** | Fade-up entrance animations with staggered delays for smooth page transitions |
| **📋 Copy Actions** | One-click copy of full summary or just key takeaways to clipboard with toast notifications |
| **🖨️ Print Support** | Native browser print with dedicated print header styling |
| **🎯 Color-Coded Metrics** | Progress bars with green (≥75%), yellow (≥50%), red (<50%) thresholds for sentiment and accuracy scores |
| **💬 Real-time Chat UI** | Chat bubbles with loading indicator ("Thinking..."), auto-scroll, and keyboard (Enter) support |
| **⏳ Loading States** | Form submission disables button and shows processing message for long-running analyses |

---

## 7. Technology Stack

### Backend

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Core programming language |
| **Flask** | Latest | Web framework for routing, templating, and request handling |
| **Flask-Login** | Latest | User session management and authentication |
| **Werkzeug** | 3.1.5 | Password hashing (`generate_password_hash`, `check_password_hash`) |
| **PyMongo** | 4.16+ | MongoDB driver for Python |
| **LangChain Core** | Latest | Prompt templates and document abstractions |
| **LangChain Ollama** | Latest | ChatOllama integration for local LLM inference |
| **LangChain Text Splitters** | Latest | `RecursiveCharacterTextSplitter` for intelligent document chunking |
| **youtube-transcript-api** | 1.2.3 | YouTube caption/transcript extraction |
| **ReportLab** | 4.0.9 | PDF generation with styled layouts and tables |
| **python-dotenv** | 1.2.1 | Environment variable management |

### Frontend

| Technology | Purpose |
|---|---|
| **HTML5** | Semantic page structure with Jinja2 templating |
| **CSS3** | Custom design system with CSS variables, animations, responsive layout |
| **JavaScript (ES6+)** | Summary formatting, clipboard API, chat functionality, form handling |
| **Google Fonts (Inter)** | Modern sans-serif typography |

### Database

| Technology | Purpose |
|---|---|
| **MongoDB** | NoSQL document database for users, summaries, and history |
| **Default connection** | `mongodb://localhost:27017/` |
| **Database name** | `Podcast_Summarizer` |

### AI/ML Infrastructure

| Technology | Purpose |
|---|---|
| **Ollama** | Local LLM inference server (default: `http://localhost:11434`) |
| **Supported Models** | `llama3.2:1b`, `gpt-oss:latest`, or any Ollama-compatible model |
| **LangChain** | Orchestration framework for LLM prompts, chains, and document processing |

### APIs & External Services

| Service | Purpose |
|---|---|
| **YouTube Transcript API** | Extracts captions (manual and auto-generated) from YouTube videos |
| **Webshare Proxy** (optional) | Proxy support for YouTube API calls in restricted environments |

---

## 8. Folder Structure

```
Podcast_Summarizer/
│
├── app.py                      # Flask application entry point
│                                # - App initialization & secret key
│                                # - Flask-Login configuration
│                                # - Authentication routes (/, /signup, /login, /logout)
│                                # - Blueprint registration
│
├── db.py                       # MongoDB connection setup
│                                # - MongoClient connection to localhost:27017
│                                # - Exports: users_collection, summaries_collection,
│                                #   history_collection
│
├── home.py                     # Core business logic blueprint (768 lines)
│                                # - YouTube URL parsing & transcript fetching
│                                # - Distributed async summarization engine
│                                # - Sentiment & emotion analysis
│                                # - Accuracy/confidence scoring
│                                # - PDF generation (ReportLab)
│                                # - Routes: /home, /chat, /history, /download-pdf,
│                                #   /clear_summary
│
├── requirements.txt            # Python dependencies
│
├── test.ipynb                  # Jupyter notebook for testing/experimentation
│
├── .gitignore                  # Git ignore rules (venv, __pycache__, .env, etc.)
│
├── services/                   # Service layer modules
│   ├── __init__.py             # Package initializer
│   └── chat_service.py        # ChatService class
│                                # - LangChain + Ollama Q&A with context injection
│                                # - PodcastAI assistant prompt with strict rules
│
├── templates/                  # Jinja2 HTML templates
│   ├── index.html              # Landing page with hero section and feature cards
│   ├── login.html              # Login page with authentication form
│   ├── signup.html             # Signup page with client-side validation
│   ├── home.html               # Main dashboard with sidebar, URL input, results
│   │                            # display, sentiment cards, accuracy cards, and chat
│   └── history.html            # Analysis history page with table view
│
├── static/                     # Static assets
│   ├── style.css               # Main design system (1105 lines)
│   │                            # - CSS variables (colors, spacing, typography)
│   │                            # - Component styles (buttons, forms, cards)
│   │                            # - Layout (sidebar, main content, landing page)
│   │                            # - Sentiment & accuracy card styles
│   │                            # - Chat UI styles
│   │                            # - Animations & transitions
│   │                            # - Print media queries
│   ├── new_styles.css          # Additional/override styles
│   ├── chat_styles.css         # Chat-specific styles
│   └── script.js               # Client-side JavaScript (227 lines)
│                                # - Summary markdown→HTML formatter
│                                # - Clipboard copy functionality
│                                # - Print handler
│                                # - Chat send/receive with fetch API
│                                # - Form submission loading state
│
└── venv/                       # Python virtual environment (gitignored)
```

---

## 9. Installation & Setup

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or higher |
| **MongoDB** | Running on `localhost:27017` (default) |
| **Ollama** | Installed and running on `localhost:11434` |
| **Git** | For cloning the repository |

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nachiket858/Podcast_Summarizer.git
cd Podcast_Summarizer
```

### Step 2: Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start MongoDB

```bash
# If using local MongoDB
mongod

# Or if using MongoDB as a service (Windows)
net start MongoDB
```

### Step 5: Install and Start Ollama

```bash
# Install Ollama (https://ollama.com/download)
# Pull the required model
ollama pull llama3.2:1b

# Start Ollama server
ollama serve
```

### Step 6: Configure Environment Variables (Optional)

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key-here
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
TRANSCRIPT_LANGS=en
TRANSCRIPT_TRANSLATE_TO=
PROXY_USERNAME=
PROXY_PASSWORD=
PROXY_URL=
```

### Step 7: Run the Application

```bash
python app.py
```

The application will start on **http://127.0.0.1:5000** in debug mode.

---

## 10. Configuration Details

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key` | Flask session encryption key (change in production) |
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model to use for summarization and analysis |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `TRANSCRIPT_LANGS` | `en` | Comma-separated preferred transcript languages |
| `TRANSCRIPT_TRANSLATE_TO` | _(empty)_ | Target language for transcript translation |
| `PROXY_USERNAME` | _(empty)_ | Webshare proxy username (for YouTube API in restricted environments) |
| `PROXY_PASSWORD` | _(empty)_ | Webshare proxy password |
| `PROXY_URL` | _(empty)_ | Generic proxy URL (alternative to Webshare) |

### LLM Configuration

- **Summarization Temperature**: `0.7` (balanced creativity and accuracy)
- **Accuracy Scoring Temperature**: `0.5` (more deterministic for scoring)
- **Chunk Size**: `3000` characters with `200` character overlap
- **Sentiment Text Truncation**: Capped at `5000` characters for efficiency
- **Accuracy Text Truncation**: Capped at `3000` characters

### Flask-Login Configuration

- **Session Protection**: `"strong"` — regenerates session on IP/user-agent change
- **Login View**: Redirects unauthenticated users to `/login`
- **Remember Me**: Enabled for persistent sessions

---

## 11. API Endpoints

### Authentication Routes (`app.py`)

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/` | No | Landing page |
| `GET/POST` | `/signup` | No | User registration (GET: form, POST: create account) |
| `GET/POST` | `/login` | No | User login (GET: form, POST: authenticate) |
| `GET` | `/logout` | Yes | Log out and redirect to login |

### Application Routes (`home.py` — Blueprint: `home_bp`)

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET/POST` | `/home` | Yes | Main dashboard. GET: show form/results. POST: process YouTube URL |
| `GET` | `/clear_summary` | Yes | Clears the current session summary and redirects to home |
| `POST` | `/chat` | Yes | Interactive Q&A chat endpoint |
| `GET` | `/history` | Yes | View user's analysis history |
| `GET` | `/download-pdf/<video_id>` | Yes | Download PDF report for a specific video |

### Chat Endpoint Details

**POST `/chat`**

Request Body (JSON):
```json
{
    "message": "What was the main topic discussed?",
    "video_id": "dQw4w9WgXcQ"
}
```

Response (JSON):
```json
{
    "response": "The main topic discussed in this podcast was..."
}
```

Error Responses:
- `400` — Video context missing (no `video_id` provided)
- `404` — No summary found for the given video ID

---

## 12. Database Schema

### Database: `Podcast_Summarizer`

#### Collection: `users`

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Auto-generated MongoDB ID |
| `username` | String | Unique username (min 4 characters) |
| `password` | String | Werkzeug-hashed password (min 6 characters raw) |

#### Collection: `summaries`

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Auto-generated MongoDB ID |
| `video_id` | String | 11-character YouTube video ID (cache key) |
| `video_url` | String | Full YouTube URL as submitted by user |
| `summary` | String | Generated executive summary with key takeaways (markdown formatted) |
| `full_text` | String | Complete raw transcript text |
| `transcript_sentiment` | Object | `{sentiment, sentiment_score, emotion, emotion_confidence}` |
| `summary_sentiment` | Object | `{sentiment, sentiment_score, emotion, emotion_confidence}` |
| `transcription_confidence` | Integer | Transcription quality score (0-100) |
| `summary_confidence` | Integer | Summary faithfulness score (0-100) |
| `created_at` | DateTime | UTC timestamp of first analysis |

**Sentiment Object Schema:**

```json
{
    "sentiment": "Positive | Negative | Neutral",
    "sentiment_score": 0-100,
    "emotion": "Excited | Serious | Motivational | Sad | Informative | Angry | Neutral",
    "emotion_confidence": 0-100
}
```

#### Collection: `history`

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Auto-generated MongoDB ID |
| `user_id` | String | Reference to user's `_id` (as string) |
| `video_id` | String | YouTube video ID |
| `video_url` | String | Full YouTube URL |
| `viewed_at` | DateTime | UTC timestamp of the analysis (updated on re-analysis via upsert) |

**Indexes/Constraints:**
- `history` uses a compound key `{user_id, video_id}` for upsert operations
- `summaries` uses `video_id` as a de facto unique key for caching
- `users` checks `username` uniqueness at the application level

---

## 13. Deployment Process

### Local Development

```bash
# 1. Start MongoDB
mongod --dbpath /path/to/data

# 2. Start Ollama
ollama serve

# 3. Run Flask
python app.py
# Server starts at http://127.0.0.1:5000 (debug mode)
```

### Production Deployment

#### Option A: Gunicorn (Linux/macOS)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Option B: Waitress (Windows)

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

#### Production Checklist

| Item | Action |
|---|---|
| **Secret Key** | Set `SECRET_KEY` to a strong random value |
| **Debug Mode** | Set `app.run(debug=False)` or use WSGI server |
| **MongoDB** | Use authenticated connection string with replica set |
| **Ollama** | Ensure Ollama server is accessible and model is pre-pulled |
| **HTTPS** | Deploy behind a reverse proxy (Nginx/Apache) with SSL |
| **Environment** | Use `.env` file or system environment variables |
| **Firewall** | Restrict MongoDB (27017) and Ollama (11434) ports to localhost |

### Docker Deployment (Recommended for production)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - SECRET_KEY=your-production-key
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - mongodb
      - ollama

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  mongo_data:
  ollama_data:
```

---

## 14. Screenshots

> **Note:** Screenshots can be captured by running the application locally and navigating to each page. Key screens to capture:

| Screen | Route | Description |
|---|---|---|
| **Landing Page** | `/` | Hero section with feature cards and CTA |
| **Login Page** | `/login` | Clean authentication form |
| **Signup Page** | `/signup` | Registration form with validation |
| **Dashboard (Empty)** | `/home` | URL input card before analysis |
| **Dashboard (Results)** | `/home` | Full results with summary, sentiment, accuracy, and chat |
| **History Page** | `/history` | Table of previously analyzed podcasts |
| **PDF Report** | `/download-pdf/<id>` | Downloaded PDF with formatted report |

---

## 15. Future Enhancements

| Enhancement | Description | Priority |
|---|---|---|
| **🎤 Audio Upload Support** | Accept direct audio file uploads (MP3, WAV) in addition to YouTube URLs using OpenAI Whisper for transcription | High |
| **🌐 Multi-Platform Support** | Extend to Spotify, Apple Podcasts, and RSS feed URLs | High |
| **📊 Advanced Analytics Dashboard** | Word clouds, topic modeling, speaker diarization visualization, and trend analysis across multiple podcasts | Medium |
| **🔔 Email Notifications** | Send summary reports via email after processing completes | Medium |
| **👥 Team Collaboration** | Shared workspaces, shared history, and collaborative annotations | Medium |
| **🔍 Full-Text Search** | Elasticsearch integration for searching across all analyzed podcast transcripts | Medium |
| **🎨 Theme Customization** | Dark mode toggle and customizable color themes | Low |
| **📱 Mobile App** | React Native or Flutter mobile application for on-the-go analysis | Low |
| **🔄 Real-time Processing** | WebSocket-based progress updates during long-running analyses instead of page reload | Medium |
| **🧠 Model Selection** | Allow users to choose from available Ollama models for different quality/speed tradeoffs | Low |
| **📑 Chapter Detection** | Automatic chapter segmentation based on topic shifts in the transcript | Medium |
| **🔗 API Access** | REST API with API key authentication for programmatic access | Medium |
| **🗓️ Scheduled Analysis** | Auto-analyze new episodes from subscribed podcast channels | Low |

---

## 16. Conclusion

**PodcastAI** is a comprehensive, production-ready web application that solves a real-world problem — extracting actionable intelligence from long-form audio content. By combining the **YouTube Transcript API** for data acquisition, **LangChain + Ollama** for distributed NLP processing, and **MongoDB** for persistent caching, the application delivers:

- **Speed** — Cached results return instantly; new analyses complete in minutes
- **Depth** — Goes beyond simple summaries to include sentiment, emotion, confidence metrics, and interactive Q&A
- **Quality** — Distributed chunked summarization ensures no content is lost from long podcasts
- **Accessibility** — Runs entirely locally with no API costs (using open-source models via Ollama)
- **Security** — Hashed passwords, strong session protection, and per-user data isolation
- **Exportability** — Professional PDF reports suitable for sharing and archiving

The modular architecture (Flask Blueprints, service layer, separated templates) makes the codebase maintainable and extensible for future enhancements.

---

<div align="center">

**Built with ❤️ using Flask, LangChain, Ollama, and MongoDB**

*PodcastAI — Unlock the Knowledge Hidden in Podcasts*

</div>
