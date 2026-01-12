# NOVA - News & Investor Analysis System

A terminal-based AI research tool that analyzes corporate IR materials, news, and investor presentations using LangChain and LLMs.

## Overview

NOVA is an AI-powered research assistant that:
- **Matches companies** from user queries to official company information
- **Scrapes IR materials** from official company websites (news, presentations, IR pitches)
- **Parses documents** from various formats (HTML, PDF, PPT, DOCX)
- **Answers questions** using RAG with quality filtering (8+/10 CEO-level reports)

## Features

- Company name resolution using LLM
- Web scraping for IR materials
- Document parsing for multiple formats
- Vector store with ChromaDB
- Iterative quality refinement (Aurora-style)
- CLI interface with rich output

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

1. **Company Resolution**: User inputs company name → LLM finds official name & website
2. **Scraping**: Scraper finds IR sections and downloads materials
3. **Parsing**: Documents are parsed into searchable chunks
4. **Indexing**: Chunks are embedded and stored in ChromaDB
5. **Question Answering**:
   - Retrieve relevant chunks
   - Generate initial answer
   - Evaluate quality (1-10 score)
   - Refine if score < 8
   - Return only if 8+ quality

## Quality Evaluation

Answers are evaluated on:
- **Specificity** (2 points): Concrete numbers and data
- **Citation Quality** (2 points): Proper source attribution
- **Direct Quotes** (2 points): Original text citations
- **Depth** (2 points): Insightful analysis
- **Completeness** (2 points): Comprehensive response

Only answers scoring 8/10 or higher are displayed.

## API Keys Used

- **GLM-4.7** (Zhipu AI): Primary LLM for chat and embeddings
- **OpenAI**: Optional embeddings (higher quality)

## Security Notes

- `.env` file contains API keys and is excluded from git
- Never commit `.env` to version control
- Use `.env.example` as template for setup

## Differences from Aurora

| Feature | Aurora | Nova |
|---------|--------|-------|
| Data Source | SEC Filings (EDGAR) | Company IR materials |
| Focus | Regulatory documents | News, presentations, IR pitches |
| Website | sec.gov | Company official websites |
| Scraper | sec-edgar-downloader | Custom web scraper |

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

## License

Internal use only.
