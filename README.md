# Hansel - APS Website Auditor (Single-Page, GitHub-Only)
He's stylish and helpful. What more could you want?

This repo runs an automated one-page audit against APS Style guidance.

## How it works
1. **Scrape** → Clean HTML via Readability → **Markdown**
2. **Layer 1** AST → Headings, lists, etc. + rules (e.g. H2 must be Title Case)
3. **Layer 2** Regex → citations, `et al` punctuation, spacing, DOIs/URLs
4. **Layer 3** Heuristics → sentence length, passive voice; **Semantic search** via TF‑IDF over the APS chunked KB to attach relevant guidance

## Inputs
- Place your chunked JSONL under `pipeline/kb/chunks/` (glob configurable).
- Place `kb/urls.map.csv` (maps filename → source URL).
- Optionally use `kb/ChunkingExample.xlsx` (first column contains JSON rows) instead of JSONL.

## Running (one page per action run)
- Go to **Actions** → **Audit one page** → **Run workflow**
  - `target_url` = URL to audit (required)
  - `kb_glob` = `pipeline/kb/chunks/**/*.jsonl` (default)
  - `kb_xlsx` = `kb/ChunkingExample.xlsx` (leave blank if you’re using JSONL)
  - `urls_map` = `kb/urls.map.csv`

## Outputs
- `reports/<timestamp>_<slug>.json` and `.md` committed to repo and uploaded as an artifact.

## Example rule hits (from Layer 1 + 2)
Input heading:
The Study's conclusions (Jones et al 2024)
Findings:
- H2 not in Title Case → suggest `The Study's Conclusions`
- Citation in heading → suggest moving it to body text
- `et al` missing period → suggest `et al.`

## Local testing


python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

Build KB (from JSONL)

python -m pipeline.kb.build_kb_index --chunks-glob "pipeline/kb/chunks/**/*.jsonl" --urls-map kb/urls.map.csv

Or build KB (from provided XLSX example)

python -m pipeline.kb.build_kb_index --xlsx kb/ChunkingExample.xlsx --urls-map kb/urls.map.csv

Audit a local Markdown file

python -m pipeline.audit_one --markdown-file samples/example.md --kb-index pipeline/kb/kb_index.json
