# AI Data Analyst

AI Data Analyst is a web-based tool for automated analysis of tabular and semi-structured datasets.

The system combines deterministic Python analysis with LLM-based interpretation. Python performs the calculations and extracts structured evidence, while AI agents generate and review the final analytical report.

## Features

- CSV, Excel, JSON, JSONL, Parquet and text-based dataset support
- Automatic schema and semantic type detection
- Numerical, categorical, datetime and text analysis
- Geospatial data analysis
- Multi-valued field detection
- Missing-value and data-quality analysis
- Correlation and cross-type relationship analysis
- Classification and regression target detection
- Automatic chart generation
- AI-generated analytical reports
- AI reviewer for checking unsupported claims and inconsistencies
- Interactive FastAPI dashboard

## Architecture

Dataset
↓
Data Loading
↓
Schema Inference
↓
Semantic Router
↓
Specialized Analyzers
↓
Structured Evidence
↓
AI Analyst
↓
AI Reviewer
↓
Dashboard

The LLM does not analyze the raw dataset directly. It receives structured evidence produced by the deterministic analysis pipeline.

## Installation

```bash
git clone <repository-url>
cd "AI Data Analyst"

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Project Structure

```text
app/
├── agents/        # Analyst and reviewer agents
├── services/      # Loading, routing, profiling and chart generation
├── tools/         # Statistical and semantic analysis modules
├── web/           # Dashboard frontend
└── main.py        # FastAPI application
```

## Current Limitations

- Some unusual identifier-like columns may require better semantic detection.
- Missingness mechanisms are not automatically inferred.
- AI-generated reports should still be reviewed for high-stakes use.
- The project focuses on structured and semi-structured tabular data, not raw image, audio or video analysis.
