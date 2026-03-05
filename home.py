from flask import Blueprint, render_template, request, flash, session, redirect, url_for, send_file, make_response, jsonify
from flask_login import login_required, current_user
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from db import summaries_collection, history_collection, comments_collection, chat_history_collection
from datetime import datetime, timedelta, timezone

# IST timezone offset
IST = timezone(timedelta(hours=5, minutes=30))
import re
import os
import asyncio
import json
import threading
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
from services.chat_service import ChatService
from services.output_cleaner import clean_text, parse_summary_sections, build_clean_response

# Initialize Services
chat_service = ChatService()

home_bp = Blueprint('home_bp', __name__)

# -------------------- Helpers --------------------

def get_video_id(url):
    if not url:
        return None

    # Normalize common copy/paste artifacts
    cleaned = url.strip()
    cleaned = cleaned.strip("[]()<>")
    cleaned = cleaned.split()[0]

    patterns = [
        r"v=([0-9A-Za-z_-]{11})",                 # watch?v=VIDEO_ID
        r"youtu\.be/([0-9A-Za-z_-]{11})",         # youtu.be/VIDEO_ID
        r"/shorts/([0-9A-Za-z_-]{11})",           # /shorts/VIDEO_ID
        r"/embed/([0-9A-Za-z_-]{11})",            # /embed/VIDEO_ID
        r"/live/([0-9A-Za-z_-]{11})",             # /live/VIDEO_ID
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    # Fallback: raw 11-char ID
    match = re.search(r"\b([0-9A-Za-z_-]{11})\b", cleaned)
    return match.group(1) if match else None


def fetch_available_captions(video_url):
    video_id = get_video_id(video_url)
    if not video_id:
        return None

    try:
        proxy_config = None
        if os.getenv("PROXY_USERNAME") and os.getenv("PROXY_PASSWORD"):
            proxy_config = WebshareProxyConfig(
                proxy_username=os.getenv("PROXY_USERNAME"),
                proxy_password=os.getenv("PROXY_PASSWORD")
            )
        elif os.getenv("PROXY_URL"):
            proxy_config = GenericProxyConfig(
                http_url=os.getenv("PROXY_URL"),
                https_url=os.getenv("PROXY_URL")
            )

        api = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        transcripts = list(transcript_list)

        preferred_langs = os.getenv("TRANSCRIPT_LANGS", "en")
        preferred_langs = [l.strip() for l in preferred_langs.split(",") if l.strip()]

        transcript = None

        for lang in preferred_langs:
            transcript = next(
                (t for t in transcripts if (not t.is_generated) and t.language_code == lang),
                None
            )
            if transcript:
                break

        if not transcript:
            for lang in preferred_langs:
                transcript = next(
                    (t for t in transcripts if t.is_generated and t.language_code == lang),
                    None
                )
                if transcript:
                    break

        if not transcript:
            transcript = next((t for t in transcripts if not t.is_generated), None)

        if not transcript and transcripts:
            transcript = transcripts[0]

        if not transcript:
            return None

        target_lang = os.getenv("TRANSCRIPT_TRANSLATE_TO", "").strip()
        if (
            target_lang
            and getattr(transcript, "is_translatable", False)
            and transcript.language_code != target_lang
        ):
            try:
                transcript = transcript.translate(target_lang)
            except Exception:
                pass

        formatter = TextFormatter()
        formatted_text = formatter.format_transcript(transcript.fetch())
        
        # return dict to be consistent
        return {'text': formatted_text, 'chapters': []}  # No semantic chapters for standard captions

    except Exception as e:
        print("Caption error:", e)
        return None


# -------------------- Async summarization --------------------

async def summarize_chunk(chunk_text, index):
    try:
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7
        )

        prompt = PromptTemplate.from_template("""
        Summarize this transcript chunk into 2-3 concise bullet points.

        Chunk {num}:
        {text}
        """)

        response = await llm.ainvoke(
            prompt.format(
                text=chunk_text,
                num=index + 1
            )
        )

        return index, response.content

    except Exception as e:
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg:
            print(f"⚠️ Ollama connection failed for chunk {index+1}. Is Ollama running?")
        else:
            print(f"Chunk {index+1} failed:", e)
        return index, None


async def generate_distributed_summary_async(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        docs = splitter.split_documents([Document(page_content=text)])

        tasks = []
        for i, doc in enumerate(docs):
            tasks.append(
                summarize_chunk(doc.page_content, i)
            )

        results = await asyncio.gather(*tasks)

        summaries = [r for r in results if r[1]]
        summaries.sort(key=lambda x: x[0])

        if not summaries:
            return None

        combined = "\n".join(s[1] for s in summaries)

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7
        )

        final_prompt = PromptTemplate.from_template("""
You are an expert content summarizer.
Please provide a response in English in the following format:

### Summary
(A concise paragraph summarizing the entire video content.)

### Key Takeaways
(Exactly 10-15 bullet points containing the most important insights.)

Input text:
{text}
        """)

        final_response = await llm.ainvoke(final_prompt.format(text=combined))
        return final_response.content
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            print("⚠️ ERROR: Cannot connect to Ollama. Please ensure Ollama is running on http://localhost:11434")
            print("   Run 'ollama serve' to start Ollama")
        else:
            print(f"⚠️ Summary generation error: {e}")
        return None


def generate_distributed_summary(text):
    return asyncio.run(generate_distributed_summary_async(text))


# -------------------- Sentiment & Emotion Analysis --------------------

async def analyze_sentiment_emotion_async(text, text_type="transcript"):
    """
    Analyzes sentiment and emotion of the given text using LLM.
    Returns: {
        'sentiment': 'Positive|Negative|Neutral',
        'sentiment_score': 0-100,
        'emotion': 'Excited|Serious|...',
        'emotion_confidence': 0-100
    }
    """
    try:
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7
        )

        # Truncate text if too long to save processing
        if len(text) > 5000:
            text = text[:5000]

        prompt = PromptTemplate.from_template("""
Analyze the sentiment and emotion of this {text_type} text. Respond ONLY with valid JSON, no other text.

Expected JSON format:
{{"sentiment": "Positive|Negative|Neutral", "sentiment_score": 0-100, "emotion": "Excited|Serious|Motivational|Sad|Informative|Angry|Neutral", "emotion_confidence": 0-100}}

Text:
{text}
        """)

        response = await llm.ainvoke(
            prompt.format(text=text, text_type=text_type)
        )

        # Parse JSON response
        response_text = response.content.strip()
        # Extract JSON from response (in case it has extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Ensure sentiment_score is present and valid
            if 'sentiment_score' not in result or not isinstance(result.get('sentiment_score'), (int, float)):
                result['sentiment_score'] = 50
            else:
                result['sentiment_score'] = max(0, min(100, int(result['sentiment_score'])))
            return result
        else:
            # Fallback if JSON parsing fails
            return {
                'sentiment': 'Neutral',
                'sentiment_score': 50,
                'emotion': 'Informative',
                'emotion_confidence': 50
            }

    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        # Return default values if analysis fails
        return {
            'sentiment': 'Neutral',
            'sentiment_score': 50,
            'emotion': 'Informative',
            'emotion_confidence': 50
        }


def analyze_sentiment_emotion(text, text_type="transcript"):
    return asyncio.run(analyze_sentiment_emotion_async(text, text_type))


# -------------------- Accuracy Calculation --------------------

async def calculate_accuracy_scores_async(full_text, summary):
    """
    Calculates transcription and summary confidence scores using LLM.
    Returns: {
        'transcription_confidence': 0-100,
        'summary_confidence': 0-100
    }
    """
    try:
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.5
        )

        # Truncate text if too long
        analysis_text = full_text if len(full_text) <= 3000 else full_text[:3000]

        transcription_prompt = PromptTemplate.from_template("""
Rate the quality and completeness of this transcript (0-100). Consider grammar, punctuation, sentence structure, and comprehensiveness.
Respond with ONLY a number 0-100, nothing else.

Transcript:
{text}
        """)

        trans_response = await llm.ainvoke(
            transcription_prompt.format(text=analysis_text)
        )

        # Parse transcription confidence
        trans_score = 50  # default
        try:
            trans_score = int(''.join(filter(str.isdigit, trans_response.content.strip()[:3])))
            trans_score = max(0, min(100, trans_score))
        except:
            pass

        # Summary confidence
        summary_prompt = PromptTemplate.from_template("""
Rate how well this summary captures the original transcript content (0-100). Consider completeness, accuracy, and relevance.
Respond with ONLY a number 0-100, nothing else.

Summary:
{summary}
        """)

        summary_response = await llm.ainvoke(
            summary_prompt.format(summary=summary if len(summary) <= 2000 else summary[:2000])
        )

        # Parse summary confidence
        summary_score = 50  # default
        try:
            summary_score = int(''.join(filter(str.isdigit, summary_response.content.strip()[:3])))
            summary_score = max(0, min(100, summary_score))
        except:
            pass

        return {
            'transcription_confidence': trans_score,
            'summary_confidence': summary_score
        }

    except Exception as e:
        print(f"Accuracy calculation error: {e}")
        # Return default values
        return {
            'transcription_confidence': 70,
            'summary_confidence': 70
        }


def calculate_accuracy_scores(full_text, summary):
    return asyncio.run(calculate_accuracy_scores_async(full_text, summary))


# -------------------- Topic Detection & Q&A Generation --------------------

def generate_topics_qa(full_text):
    """
    Extracts key topics from the transcript and generates related Q&A pairs.
    Returns: list of { 'topic': str, 'questions': [{'q': str, 'a': str}] }
    """
    try:
        if not full_text or len(full_text) < 100:
            return []

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.5
        )

        # Truncate for manageable processing
        analysis_text = full_text if len(full_text) <= 5000 else full_text[:5000]

        prompt = PromptTemplate.from_template("""
Analyze this podcast transcript and extract exactly 5 key topics discussed.
For each topic, generate 2 relevant questions and their answers based ONLY on the transcript.

Transcript:
{text}

Respond ONLY with valid JSON in this EXACT format, no other text:
[{{
    "topic": "Topic Name",
    "questions": [
        {{"q": "Question 1?", "a": "Answer 1"}},
        {{"q": "Question 2?", "a": "Answer 2"}}
    ]
}}]
        """)

        response = llm.invoke(prompt.format(text=analysis_text))
        response_text = response.content.strip()

        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            topics = json.loads(json_match.group())
            # Validate structure
            valid_topics = []
            for t in topics[:5]:
                if isinstance(t, dict) and 'topic' in t and 'questions' in t:
                    valid_qs = []
                    for q in t['questions'][:2]:
                        if isinstance(q, dict) and 'q' in q and 'a' in q:
                            valid_qs.append({'q': str(q['q']), 'a': str(q['a'])})
                    if valid_qs:
                        valid_topics.append({'topic': str(t['topic']), 'questions': valid_qs})
            return valid_topics

        return []

    except Exception as e:
        print(f"Topic detection error: {e}")
        return []

# -------------------- PDF Generation --------------------

def generate_pdf(video_id, record):
    """
    Generates a PDF with summary, sentiment analysis, and accuracy scores.
    Returns: BytesIO object containing the PDF
    """
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4E7FFF'),
            spaceAfter=30,
            alignment=1  # center
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1F2937'),
            spaceAfter=12,
            spaceBefore=12
        )

        # Title
        title_text = "Podcast Summary Report"
        elements.append(Paragraph(title_text, title_style))
        elements.append(Spacer(1, 0.3*inch))

        # Podcast Title (Video ID)
        video_url = record.get('video_url', '')
        if video_url:
            url_para = Paragraph(f"<b>Podcast URL:</b> {video_url}", styles['Normal'])
            elements.append(url_para)

        # Date
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_para = Paragraph(f"<b>Generated:</b> {date_str}", styles['Normal'])
        elements.append(date_para)
        elements.append(Spacer(1, 0.3*inch))

        # Summary Section - extract summary and key takeaways separately
        summary_text = record.get('summary', 'N/A')
        
        # Clean the summary text first
        summary_text = clean_text(summary_text)
        
        summary_part = summary_text
        key_takeaways = []

        # Try to parse out summary and key takeaways from the markdown-formatted text
        if '### Summary' in summary_text or '### Key Takeaways' in summary_text or '### सारांश' in summary_text or '### मुख्य' in summary_text or 'Summary' in summary_text or 'Key Takeaways' in summary_text:
            parsed = parse_summary_sections(summary_text)
            summary_part = parsed['summary']
            key_takeaways = parsed['keypoints']
        
        if not summary_part:
            summary_part = summary_text

        elements.append(Paragraph("Summary", heading_style))
        # Clean HTML-unsafe characters for ReportLab
        safe_summary = summary_part.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        summary_para = Paragraph(safe_summary, styles['BodyText'])
        elements.append(summary_para)
        elements.append(Spacer(1, 0.3*inch))

        # Key Takeaways Section
        if key_takeaways:
            elements.append(Paragraph("Key Takeaways", heading_style))
            bullet_style = ParagraphStyle(
                'BulletStyle',
                parent=styles['BodyText'],
                bulletIndent=18,
                leftIndent=36,
                spaceBefore=4,
                spaceAfter=4
            )
            for takeaway in key_takeaways:
                safe_takeaway = takeaway.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                elements.append(Paragraph(f"• {safe_takeaway}", bullet_style))
            elements.append(Spacer(1, 0.3*inch))

        # Sentiment & Emotion Section
        trans_sent = record.get('transcript_sentiment', {})
        summary_sent = record.get('summary_sentiment', {})

        if trans_sent or summary_sent:
            elements.append(Paragraph("Sentiment &amp; Emotion Analysis", heading_style))

            sentiment_data = []
            sentiment_data.append(['Type', 'Sentiment', 'Score', 'Emotion'])

            if trans_sent:
                sentiment_data.append([
                    'Transcript',
                    trans_sent.get('sentiment', 'N/A'),
                    f"{trans_sent.get('sentiment_score', 0)}%",
                    trans_sent.get('emotion', 'N/A')
                ])

            if summary_sent:
                sentiment_data.append([
                    'Summary',
                    summary_sent.get('sentiment', 'N/A'),
                    f"{summary_sent.get('sentiment_score', 0)}%",
                    summary_sent.get('emotion', 'N/A')
                ])

            # Use dynamic column widths based on available page width
            avail_width = letter[0] - 144  # page width minus margins (72+72)
            col_count = len(sentiment_data[0])
            col_w = avail_width / col_count

            # Wrap cell content in Paragraphs for text wrapping
            cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=9, alignment=1)
            header_cell_style = ParagraphStyle('HeaderCellStyle', parent=styles['Normal'], fontSize=10,
                                               alignment=1, textColor=colors.whitesmoke)
            wrapped_data = []
            for row_idx, row in enumerate(sentiment_data):
                style = header_cell_style if row_idx == 0 else cell_style
                wrapped_data.append([Paragraph(str(cell), style) for cell in row])

            sentiment_table = Table(wrapped_data, colWidths=[col_w] * col_count)
            sentiment_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4E7FFF')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(sentiment_table)
            elements.append(Spacer(1, 0.3*inch))

        # Accuracy Section
        trans_acc = record.get('transcription_confidence', 0)
        summary_acc = record.get('summary_confidence', 0)

        if trans_acc or summary_acc:
            elements.append(Paragraph("Analysis Accuracy", heading_style))

            accuracy_data = []
            accuracy_data.append(['Type', 'Confidence'])
            accuracy_data.append(['Transcription', f"{trans_acc}%"])
            accuracy_data.append(['Summary', f"{summary_acc}%"])

            # Dynamic column widths for accuracy table
            avail_width = letter[0] - 144
            acc_col_count = len(accuracy_data[0])
            acc_col_w = avail_width / acc_col_count

            # Wrap cell content in Paragraphs
            acc_cell_style = ParagraphStyle('AccCellStyle', parent=styles['Normal'], fontSize=9, alignment=1)
            acc_header_style = ParagraphStyle('AccHeaderStyle', parent=styles['Normal'], fontSize=10,
                                               alignment=1, textColor=colors.whitesmoke)
            wrapped_acc = []
            for row_idx, row in enumerate(accuracy_data):
                style = acc_header_style if row_idx == 0 else acc_cell_style
                wrapped_acc.append([Paragraph(str(cell), style) for cell in row])

            accuracy_table = Table(wrapped_acc, colWidths=[acc_col_w] * acc_col_count)
            accuracy_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4E7FFF')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(accuracy_table)

        # Build PDF
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer

    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

# -------------------- Background Analysis --------------------

def _run_background_analysis(video_id, full_text, summary):
    """
    Runs sentiment, accuracy, and topic analysis in a background thread.
    Updates the DB record progressively as each step completes.
    """
    try:
        # Step 1: Sentiment analysis
        print(f"[PodcastAI] [{video_id}] Background: Starting sentiment analysis...")
        transcript_sentiment = analyze_sentiment_emotion(full_text, "transcript")
        summary_sentiment = analyze_sentiment_emotion(summary, "summary")
        summaries_collection.update_one(
            {'video_id': video_id},
            {'$set': {
                'transcript_sentiment': transcript_sentiment,
                'summary_sentiment': summary_sentiment,
                'analysis_progress': 'sentiment_done'
            }}
        )
        print(f"[PodcastAI] [{video_id}] Background: Sentiment analysis complete.")

        # Step 2: Accuracy scores
        print(f"[PodcastAI] [{video_id}] Background: Calculating accuracy scores...")
        accuracy_scores = calculate_accuracy_scores(full_text, summary)
        summaries_collection.update_one(
            {'video_id': video_id},
            {'$set': {
                'transcription_confidence': accuracy_scores['transcription_confidence'],
                'summary_confidence': accuracy_scores['summary_confidence'],
                'analysis_progress': 'accuracy_done'
            }}
        )
        print(f"[PodcastAI] [{video_id}] Background: Accuracy scores complete.")

        # Step 3: Topic detection + Q&A
        print(f"[PodcastAI] [{video_id}] Background: Generating topics...")
        topics = generate_topics_qa(full_text)
        summaries_collection.update_one(
            {'video_id': video_id},
            {'$set': {
                'topics': topics,
                'analysis_progress': 'complete'
            }}
        )
        print(f"[PodcastAI] [{video_id}] Background: \u2705 All analysis complete.")

    except Exception as e:
        print(f"[PodcastAI] [{video_id}] Background analysis error: {e}")
        summaries_collection.update_one(
            {'video_id': video_id},
            {'$set': {'analysis_progress': 'complete'}}
        )


# -------------------- Routes --------------------

@home_bp.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        video_id = get_video_id(youtube_url)

        if not video_id:
            flash("Invalid YouTube URL", "error")
            return redirect(url_for('home_bp.dashboard'))

        # Check cache first
        cached = summaries_collection.find_one({'video_id': video_id})

        if cached:
            # If cached but missing background analysis, kick it off
            if cached.get('analysis_progress') != 'complete':
                full_text = cached.get('full_text', '')
                summary = cached.get('summary', '')
                if full_text and summary:
                    thread = threading.Thread(
                        target=_run_background_analysis,
                        args=(video_id, full_text, summary),
                        daemon=True
                    )
                    thread.start()
        else:
            content_data = fetch_available_captions(youtube_url)
            if not content_data:
                flash("Could not fetch captions/audio. Check URL or try again.", "error")
                return redirect(url_for('home_bp.dashboard'))

            full_text = content_data['text']

            # Generate summary ONLY (the user sees this immediately)
            print(f"[PodcastAI] [{video_id}] Generating summary...")
            raw_summary = generate_distributed_summary(full_text)

            if not raw_summary:
                flash("\u274c Summary generation failed. Make sure Ollama is running (ollama serve).", "error")
                return redirect(url_for('home_bp.dashboard'))

            summary = clean_text(raw_summary)
            print(f"[PodcastAI] [{video_id}] Summary ready. Redirecting to results...")

            # Save summary immediately with progress marker
            summaries_collection.insert_one({
                'video_id': video_id,
                'video_url': youtube_url,
                'summary': summary,
                'full_text': full_text,
                'analysis_progress': 'summary_done',
                'created_at': datetime.utcnow()
            })

            # Kick off background analysis in a separate thread
            thread = threading.Thread(
                target=_run_background_analysis,
                args=(video_id, full_text, summary),
                daemon=True
            )
            thread.start()

        history_collection.update_one(
            {'user_id': current_user.id, 'video_id': video_id},
            {'$set': {'video_url': youtube_url, 'viewed_at': datetime.utcnow()}},
            upsert=True
        )

        # Redirect to the results page IMMEDIATELY (summary is ready)
        return redirect(url_for('home_bp.results', video_id=video_id))

    # --- GET request ---
    recent_history = list(
        history_collection.find({'user_id': current_user.id}).sort('viewed_at', -1).limit(4)
    )

    response = make_response(render_template(
        'dashboard.html',
        username=current_user.username,
        recent_history=recent_history
    ))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@home_bp.route('/results/<video_id>')
@login_required
def results(video_id):
    """Display analysis results for a specific video."""
    record = summaries_collection.find_one({'video_id': video_id})

    if not record or 'summary' not in record:
        flash("No summary found for this video. Please analyze it first.", "error")
        return redirect(url_for('home_bp.dashboard'))

    summary = record['summary']
    progress = record.get('analysis_progress', 'complete')
    sentiment_data = {
        'transcript': record.get('transcript_sentiment', {}),
        'summary': record.get('summary_sentiment', {})
    }
    accuracy_data = {
        'transcription_confidence': record.get('transcription_confidence', 0),
        'summary_confidence': record.get('summary_confidence', 0)
    }
    topics = record.get('topics', [])

    return render_template(
        'results.html',
        username=current_user.username,
        summary=summary,
        video_id=video_id,
        sentiment_data=sentiment_data,
        accuracy_data=accuracy_data,
        topics=topics,
        analysis_progress=progress
    )


@home_bp.route('/analysis-status/<video_id>')
@login_required
def analysis_status(video_id):
    """API endpoint for frontend to poll background analysis progress."""
    record = summaries_collection.find_one({'video_id': video_id})
    if not record:
        return jsonify({'progress': 'not_found'}), 404

    progress = record.get('analysis_progress', 'complete')
    data = {'progress': progress}

    # Include completed data sections so the frontend can render them
    if progress in ('sentiment_done', 'accuracy_done', 'complete'):
        data['sentiment'] = {
            'transcript': record.get('transcript_sentiment', {}),
            'summary': record.get('summary_sentiment', {})
        }

    if progress in ('accuracy_done', 'complete'):
        data['accuracy'] = {
            'transcription_confidence': record.get('transcription_confidence', 0),
            'summary_confidence': record.get('summary_confidence', 0)
        }

    if progress == 'complete':
        data['topics'] = record.get('topics', [])

    return jsonify(data)


@home_bp.route('/chat-page/<video_id>')
@login_required
def chat_page(video_id):
    """Dedicated interactive Q&A page for a video."""
    record = summaries_collection.find_one({'video_id': video_id})

    if not record or 'summary' not in record:
        flash("No summary found for this video. Please analyze it first.", "error")
        return redirect(url_for('home_bp.dashboard'))

    topics = record.get('topics', [])

    return render_template(
        'chat.html',
        username=current_user.username,
        video_id=video_id,
        topics=topics
    )


@home_bp.route('/clear_summary')
@login_required
def clear_summary():
    session.clear()
    return redirect(url_for('home_bp.dashboard'))


@home_bp.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    message = data.get('message')
    video_id = data.get('video_id')

    if not video_id:
        return {'response': 'Error: Video context missing.'}, 400

    # Retrieve full transcript and summary from DB
    record = summaries_collection.find_one({'video_id': video_id})

    if not record or 'summary' not in record:
        return {'response': 'Error: No summary found for this video. Please generate it first.'}, 404

    full_text = record.get('full_text', '')
    summary = record.get('summary', '')

    # Get response with confidence score using full transcript
    result = chat_service.get_chat_response(full_text, summary, message)

    # Store chat history in DB
    try:
        chat_history_collection.insert_one({
            'user_id': current_user.id,
            'video_id': video_id,
            'question': message,
            'answer': result['answer'],
            'confidence_score': result['confidence_score'],
            'timestamp': datetime.now(IST)
        })
    except Exception as e:
        print(f"Chat history save error: {e}")

    return {
        'response': result['answer'],
        'confidence_score': result['confidence_score']
    }


@home_bp.route('/chat-history/<video_id>', methods=['GET'])
@login_required
def get_chat_history(video_id):
    """Fetch previous Q&A history for the current user and video."""
    records = list(
        chat_history_collection.find(
            {'user_id': current_user.id, 'video_id': video_id}
        ).sort('timestamp', 1).limit(50)
    )

    history = []
    for r in records:
        history.append({
            'question': r.get('question', ''),
            'answer': r.get('answer', ''),
            'confidence_score': r.get('confidence_score', 0),
            'timestamp': r.get('timestamp', datetime.utcnow()).strftime('%b %d, %Y at %H:%M')
        })

    return jsonify({'history': history})


# -------------------- Comments API --------------------

@home_bp.route('/comments/<video_id>', methods=['GET'])
@login_required
def get_comments(video_id):
    """Fetch all comments for a video, sorted newest first."""
    comments = list(
        comments_collection.find({'video_id': video_id})
            .sort('created_at', -1)
            .limit(100)
    )
    
    result = []
    for c in comments:
        result.append({
            'username': c.get('username', 'Anonymous'),
            'comment_text': c.get('comment_text', ''),
            'created_at': (c.get('created_at', datetime.utcnow()).replace(tzinfo=timezone.utc).astimezone(IST)).strftime('%b %d, %Y at %H:%M')
        })
    
    return jsonify({'comments': result})


@home_bp.route('/comments', methods=['POST'])
@login_required
def add_comment():
    """Add a new comment for a video."""
    data = request.get_json()
    video_id = data.get('video_id')
    comment_text = data.get('comment_text', '').strip()
    
    if not video_id or not comment_text:
        return jsonify({'error': 'Video ID and comment text are required.'}), 400
    
    if len(comment_text) > 2000:
        return jsonify({'error': 'Comment too long (max 2000 characters).'}), 400
    
    comment = {
        'video_id': video_id,
        'username': current_user.username,
        'comment_text': comment_text,
        'created_at': datetime.now(IST)
    }
    
    comments_collection.insert_one(comment)
    
    return jsonify({
        'success': True,
        'comment': {
            'username': comment['username'],
            'comment_text': comment['comment_text'],
            'created_at': comment['created_at'].strftime('%b %d, %Y at %H:%M')
        }
    })


@home_bp.route('/history')
@login_required
def history():
    history = list(
        history_collection.find({'user_id': current_user.id}).sort('viewed_at', -1)
    )

    return render_template(
        'history.html',
        username=current_user.username,
        history=history
    )


@home_bp.route('/download-pdf/<video_id>')
@login_required
def download_pdf(video_id):
    """Download podcast summary as PDF"""
    try:
        record = summaries_collection.find_one({'video_id': video_id})

        if not record or 'summary' not in record:
            flash("Summary not found.", "error")
            return redirect(url_for('home_bp.dashboard'))

        pdf_buffer = generate_pdf(video_id, record)

        if not pdf_buffer:
            flash("Error generating PDF.", "error")
            return redirect(url_for('home_bp.dashboard'))

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'podcast_summary_{video_id}.pdf'
        )


    except Exception as e:
        print(f"Download PDF error: {e}")
        flash("Error downloading PDF.", "error")
        return redirect(url_for('home_bp.dashboard'))
