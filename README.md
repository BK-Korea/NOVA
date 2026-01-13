# NOVA - News & Investor Analysis System

A terminal-based AI research tool that analyzes corporate IR materials, news, and investor presentations using LangChain and LLMs.

## Overview

NOVA is an AI-powered research assistant that:
- **Matches companies** from user queries to official company information
- **Scrapes IR materials** from official company websites (news, presentations, IR pitches)
- **Parses documents** from various formats (HTML, PDF, PPT, DOCX)
- **Answers questions** using RAG with quality filtering (8+/10 CEO-level reports)

## Features

- **Intelligent Company Matching**: LLM-powered company name resolution with IR website detection
- **Advanced IR Scraping**: 
  - Recursive crawling with subdomain detection (ir.*, investors.*)
  - Automatic discovery of IR sections from homepage links
  - Support for shareholder letters, quarterly reports, presentations, news
  - **SEC filings automatically excluded** to reduce costs
- **Multi-format Document Parsing**: HTML, PDF, PPT, DOCX with fallback strategies
- **Smart Caching**: Reuses indexed materials to avoid redundant API calls
- **Vector Store**: ChromaDB with persistent storage
- **Iterative Quality Refinement**: Aurora-style quality evaluation (8+/10 CEO-level reports)
- **Modern CLI**: Rich terminal UI with magenta theme

## Installation

```bash
# Navigate to NOVA directory
cd /Users/bk/NOVA

# Install package in editable mode
python3 -m pip install --user -e .

# Add to PATH (optional, for easier access)
export PATH="/Users/bk/Library/Python/3.9/bin:$PATH"
```

## Configuration

Create a `.env` file with your API keys (already configured):

```bash
GLM_API_KEY=your_glm_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
OPENAI_API_KEY=your_openai_api_key_here
CHAT_MODEL=glm-4.7
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_PROVIDER=openai
```

**Note:** The `.env` file contains sensitive API keys and is already in `.gitignore`.

## Usage

### Start NOVA (Interactive Mode)

```bash
# Direct path
/Users/bk/Library/Python/3.9/bin/nova

# Or if added to PATH
nova
```

You'll be prompted to enter a company name, then NOVA will:
1. Find the official company information
2. Scrape IR materials from their website
3. Index the materials
4. Enter Q&A mode where you can ask questions

### Example Flow

```
$ nova

[NOVA Banner]

✓ System initialized

? Enter company name: Apple

Searching for: Apple
✓ Found: Apple Inc.
Website: https://www.apple.com

Is this correct? [Y/n]: Y

Fetching IR materials...

Materials Summary
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Material Type    ┃ Count  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ news             │ 15     │
│ presentation     │ 8      │
│ ir_pitch         │ 3      │
├──────────────────┼────────┤
│ Total Chunks     │ 156    │
└──────────────────┴────────┘

Q&A Mode
Ready for questions!

? Your question: What are the key growth drivers?

[Answer with quality score]
```

## Project Structure

```
NOVA/
├── .env                    # API keys (not in git)
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Project configuration
├── setup.py                # Package setup
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   ├── raw/               # Downloaded raw files
│   ├── processed/         # Processed documents
│   └── vectordb/          # ChromaDB storage
└── src/
    ├── __init__.py
    ├── main.py            # CLI entry point
    ├── config.py          # Configuration management
    ├── agents/
    │   ├── __init__.py
    │   ├── graph.py       # Main NovaAgent
    │   ├── state.py       # Agent state
    │   ├── company_matcher.py  # Company matching
    │   └── quality_evaluator.py  # Quality scoring
    ├── scrapers/
    │   ├── __init__.py
    │   └── ir_scraper.py  # IR materials scraper
    ├── document/
    │   ├── __init__.py
    │   └── parser.py      # Document parser
    ├── vectorstore/
    │   ├── __init__.py
    │   └── chroma_store.py  # ChromaDB wrapper
    └── llm/
        ├── __init__.py
        └── glm_client.py  # GLM/OpenAI clients
```

## How It Works

1. **Company Resolution**: User inputs company name → LLM finds official name, website, and IR website
2. **Intelligent Scraping**: 
   - Tries IR subdomains (ir.*, investors.*)
   - Crawls homepage for IR links
   - Recursively follows IR-related pages
   - **Excludes SEC filings** automatically
3. **Document Processing**: 
   - Downloads and parses materials (PDF, HTML, PPT)
   - Extracts text into searchable chunks
   - Checks if already indexed (caching)
4. **Indexing**: Chunks are embedded (OpenAI/GLM) and stored in ChromaDB
5. **Question Answering**:
   - Retrieve relevant chunks using semantic search
   - Generate initial answer with GLM-4.7
   - Evaluate quality (1-10 score) on 5 criteria
   - Refine iteratively if score < 8
   - Return only if 8+ quality (CEO-level)

## Quality Evaluation

Answers are evaluated on:
- **Specificity** (2 points): Concrete numbers and data
- **Citation Quality** (2 points): Proper source attribution
- **Direct Quotes** (2 points): Original text citations
- **Depth** (2 points): Insightful analysis
- **Completeness** (2 points): Comprehensive response

Only answers scoring 8/10 or higher are displayed.

## API Keys Used

- **GLM-4.7** (Zhipu AI): Primary LLM for chat (supports reasoning models)
- **OpenAI**: Optional embeddings (text-embedding-3-small, recommended for better quality)

### Cost Optimization

- **Embeddings**: ~$0.003 per 100 chunks (very cheap)
- **Q&A**: ~$0.01-0.05 per question (depends on complexity)
- **Caching**: Already indexed materials are reused (no additional cost)
- **SEC Exclusion**: Automatically filters out SEC filings to reduce processing

## Security Notes

- `.env` file contains API keys and is excluded from git
- Never commit `.env` to version control
- Use `.env.example` as template for setup

## Recent Updates

### v0.2.0 (Latest)
- ✅ **SEC Filing Exclusion**: Automatically filters out SEC documents to reduce costs
- ✅ **Enhanced IR Scraper**: Recursive crawling with subdomain detection
- ✅ **GLM-4.7 Reasoning Support**: Proper handling of reasoning models
- ✅ **Shareholder Letters**: Added support for shareholder/CEO letters
- ✅ **Smart Caching**: Reuses indexed materials across sessions
- ✅ **Improved UI**: Unified magenta theme throughout
- ✅ **Comprehensive Tests**: Full pipeline test suite

## Differences from Aurora

| Feature | Aurora | Nova |
|---------|--------|-------|
| Data Source | SEC Filings (EDGAR) | Company IR materials (excludes SEC) |
| Focus | Regulatory documents | News, presentations, IR pitches, shareholder letters |
| Website | sec.gov | Company official IR websites |
| Scraper | sec-edgar-downloader | Custom intelligent web scraper |
| Caching | Basic | Persistent with reuse detection |

## Testing

Run the test suite to verify functionality:

```bash
# Test scraper only
python3 tests/test_scraper.py --company coca-cola

# Test full pipeline (scraping → parsing → indexing → Q&A)
python3 tests/test_full_pipeline.py
```

## Troubleshooting

### SSL Warning
You may see this warning (can be ignored):
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+
```

### Module Not Found Error
If you get import errors, reinstall the package:
```bash
python3 -m pip install --user -e . --force-reinstall
```

### No Materials Found
- Check if company has an IR website (some companies use different structures)
- Try using the full company name (e.g., "The Coca-Cola Company" instead of "Coca-Cola")
- Check network connectivity (some sites may block automated access)

### GLM API Issues
- Verify API key is correct in `.env`
- Check Zhipu AI console for usage/quota: https://open.bigmodel.cn/
- GLM-4.7 is a reasoning model - responses may take longer

## License

Internal use only.
