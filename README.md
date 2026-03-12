# HR-Breaker

Resume optimization tool that transforms any resume into a job-specific, ATS-friendly PDF.

![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)

## Features

- **Any format in** — LaTeX, plain text, Markdown, HTML, PDF
- **Profile archives** — Persist reusable candidate profiles with uploaded documents and notes
- **Folder import** — Ingest an entire local folder into a profile in one click
- **Document extraction** — Structured facts (experience, skills, contact info, etc.) are extracted from each document and used for retrieval-based tailoring
- **Retrieval-based tailoring** — Documents ranked against the target job with lexical, keyword, and vector signals before synthesis
- **Direct upload fallback** — One-off resume upload flow without an archive
- **Optimized PDF out** — Single-page, professionally formatted
- **Multi-filter validation** — ATS simulation, keyword matching, hallucination detection, semantic similarity, AI-text detection
- **No fabrication** — Hallucination detection prevents made-up claims
- **User instructions** — Guide the optimizer with extra context ("Focus on Python", "I have K8s experience")
- **Profile instructions** — Permanent per-profile notes the optimizer always receives
- **Multi-language output** — Optimize in English, then translate (e.g. `-l ru`)
- **Web UI + CLI** — Streamlit dashboard or command-line
- **Runtime log** — Live token usage, iteration progress, and filter results

## How It Works

1. **Profile mode:** create a profile, add files/notes, wait for extraction to complete
2. Provide a job posting URL or paste the description
3. The system ranks your profile documents against the job and synthesizes a tailored source
4. The optimizer generates an HTML resume from the ranked evidence
5. Filters validate it (ATS simulation, keywords, hallucination, semantic similarity)
6. If filters fail, the optimizer refines with feedback — repeat up to N iterations
7. When all checks pass, HTML is rendered to PDF via WeasyPrint

**Direct upload** (no profile): steps 1–2 collapsed into uploading a single resume file.

## Quick Start

```bash
# Install
uv sync

# Configure
cp .env.example .env
# Edit .env — set your API key (see Configuration below)

# Run web UI
uv run streamlit run src/hr_breaker/main.py
```

### Using Moonshot AI (Kimi)

To use Moonshot AI instead of Gemini:

1. Get a Moonshot AI API key from https://platform.moonshot.ai/
2. Set `MOONSHOT_API_KEY` in your `.env` file
3. Configure models to use Moonshot:
   ```bash
   PRO_MODEL=moonshot/kimi-k2-5
   FLASH_MODEL=moonshot/kimi-k2-5
   ```

Moonshot AI models work via LiteLLM — see [LiteLLM Moonshot docs](https://docs.litellm.ai/docs/providers/moonshot) for available models.

## Usage

### Web UI

Launch with `uv run streamlit run src/hr_breaker/main.py`

**Profile archive workflow:**
1. Create a profile (name, optional full name, optional standing instructions)
2. Upload files, import a folder, or add notes — extraction runs automatically in the background
3. Select which documents are in scope for this run
4. Enter a job URL or paste the posting
5. Optionally add one-off session instructions
6. Click **Optimize** and download the PDF

If documents were added before extraction existed, click **Extract facts** to backfill.

**Direct upload:** skip profile creation and just upload a resume file.

### CLI

```bash
# Optimize from file or URL
uv run hr-breaker optimize resume.txt https://example.com/job
uv run hr-breaker optimize resume.txt job.txt

# Debug mode — saves each iteration
uv run hr-breaker optimize resume.txt job.txt -d

# One-off instructions
uv run hr-breaker optimize resume.txt job.txt -i "Focus on Python, I have K8s experience"

# Translate output
uv run hr-breaker optimize resume.txt https://example.com/job -l ru

# Lenient mode — relaxes hallucination/AI checks (use with caution)
uv run hr-breaker optimize resume.txt job.txt --no-shame

# Backfill extraction for a profile or all profiles
uv run hr-breaker backfill
uv run hr-breaker backfill --profile <profile-id>
uv run hr-breaker backfill --force   # re-extract even if already done

# List generated PDFs
uv run hr-breaker list
```

## Output

- Final PDFs: `output/<name>_<company>_<role>.pdf`
- Debug iterations: `output/debug_<company>_<role>/` (with `-d`)
- Records: `output/index.json`

## Configuration

Copy `.env.example` to `.env`. All settings are optional except an API key.

### LLM provider

The default provider is Google Gemini. To use a different provider, set the shared-mode variables:

```env
# Google Gemini (default)
LLM_SHARED_PROVIDER=gemini
LLM_SHARED_API_KEY=your-gemini-key
PRO_MODEL=gemini/gemini-2.5-pro-preview-05-06
FLASH_MODEL=gemini/gemini-2.5-flash-preview-05-20

# OpenAI
LLM_SHARED_PROVIDER=openai_compatible
LLM_SHARED_API_KEY=your-openai-key
PRO_MODEL=openai/gpt-4o
FLASH_MODEL=openai/gpt-4o-mini

# Any OpenAI-compatible endpoint (Ollama, NVIDIA NIM, etc.)
LLM_SHARED_PROVIDER=openai_compatible
LLM_SHARED_BASE_URL=http://localhost:11434/v1
LLM_SHARED_API_KEY=ollama
PRO_MODEL=openai/llama3.3:70b
FLASH_MODEL=openai/llama3.1:8b
```

Model names follow [LiteLLM format](https://docs.litellm.ai/docs/providers).

### Embedding model (optional — used for vector similarity filter)

```env
EMBEDDING_MODEL=gemini/text-embedding-004
EMBEDDING_API_KEY=your-key
# or for a different provider:
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=your-openai-key
```

If no embedding key is set, the vector similarity filter is skipped.

### Other notable settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HR_BREAKER_MAX_ITERATIONS` | `5` | Max optimizer iterations |
| `REASONING_EFFORT` | `medium` | `none/low/medium/high` for reasoning models |
| `RETRY_MAX_ATTEMPTS` | `5` | Retries on rate limit errors |
| `LOG_LEVEL` | `WARNING` | Set to `INFO` or `DEBUG` for verbose output |

See `.env.example` for the full list.

---

## Architecture

```
src/hr_breaker/
├── agents/          # Pydantic-AI agents (optimizer, reviewer, extractor, etc.)
├── filters/         # Validation plugins (ATS, keywords, hallucination, AI-text)
├── services/        # Rendering, scraping, caching, profile store, extraction worker
│   └── scrapers/    # Job scraper implementations
├── models/          # Pydantic data models
├── orchestration.py # Core optimization loop
├── main.py          # Streamlit UI
└── cli.py           # Click CLI
```

**Filters** (run in parallel by default, priority order for `--seq` mode):

| Priority | Filter | Purpose |
|----------|--------|---------|
| 0 | ContentLengthChecker | Pre-render size check |
| 1 | DataValidator | HTML structure validation |
| 3 | HallucinationChecker | Detect fabricated claims |
| 4 | KeywordMatcher | TF-IDF keyword matching |
| 5 | LLMChecker | Visual + ATS simulation |
| 6 | VectorSimilarityMatcher | Semantic similarity |
| 7 | AIGeneratedChecker | Detect AI-sounding text |

## Security

### API key persistence

When you save provider settings via the web UI sidebar, HR-Breaker writes API keys and model names to your local `.env` file (the same file used by `cp .env.example .env`). This is intentional — settings survive app restarts — but it means **API keys are stored in plaintext on disk**.

If you are running HR-Breaker on a shared machine or in a container, use environment variables directly instead of the UI persistence feature, and ensure the `.env` file has appropriate file permissions (`chmod 600 .env`). The `.env` file is already in `.gitignore` and will not be committed.

## Deployment

### Streamlit Cloud

`packages.txt` lists the system-level dependencies required by WeasyPrint for PDF rendering (Cairo, Pango, GDK-Pixbuf). Streamlit Cloud reads this file automatically during build. No changes needed for local development — these libraries are typically pre-installed on Linux hosts.

## Development

```bash
uv run pytest tests/
uv sync --group dev
```
