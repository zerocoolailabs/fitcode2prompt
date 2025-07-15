# fitcode2prompt

A tool to reduce the size of a codebase. You get complete fine-grained control over which files to compress and by how much, or you can set a budget and let an LLM figure out the best way to get there.

Inspired by the awesome [code2prompt](https://github.com/mufeedvh/code2prompt).

## Compression Levels

| Level  | Reduction | Description                                                                                                                          |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| none   | 0%        | No compression, full original content.                                                                                               |
| trim   | 5%        | Removes excess whitespace, empty lines, and imports while keeping code intact.                                                       |
| light  | 15%       | Builds on trim by removing unnecessary or redundant comments                                                                         |
| medium | 50%       | Progressive summarization; may replace canonical functions with signature and/or one-sentence description, essential code intact.    |
| heavy  | 90%       | Aggressive approach where many functions become just a signature with one-sentence description; retains critical code when possible. |
| max    | 100%      | Summarizes file to one to three sentences.                                                                                           |

## CLI Usage Examples

By default final result is copied to clipboard and saved to fitcode2prompt.out (both options are configurable).

```bash
fitcode2prompt target/
# Applies trim (default) compression to all files.
```

```bash
fitcode2prompt target/ -b 8000
# Uses LLM planner to compress files to meet token budget.
```

```bash
fitcode2prompt . -i "*.py,*.md" --compression-0 "*.md" --compression-50 "*.json"
# Compresses .py files with default (trim), no compression on .md, 50% on JSON.
```

```bash
fitcode2prompt . -i "*.py,*.md" --compression-0 "*.md" --compression-50 "lib.py" -b 5000
# No compression for Markdown files, 50% reduction for lib.py,
# LLM planner compresses the rest of the files to meet budget
```

```bash
fitcode2prompt . --count-only
# Outputs total token count for the codebase.
```

Run `fitcode2prompt --help` for full command options.

## Python Library Usage

```python
from fitcode2prompt.summarizer import Summarizer

summarizer = Summarizer(
    path=".",
    llm_model_planner="gpt-4",
    llm_model_summarizer="gpt-3.5-turbo",
    exclude_patterns=[
        "**/node_modules/**",
        "**/venv/**",
        "**/dist/**",
        "**/build/**"
    ],
    budget=budget,
    return_results=True,
    no_clipboard=True,
    output_dir="./artifacts"  # CI artifact directory
)

summary = summarizer.run()

```

## Installation

### 1. Install with UV (Recommended)

We recommend using [UV](https://github.com/astral-sh/uv) for fast, reliable package management:

```bash
# Clone and install fitcode2prompt
git clone https://github.com/ZeroCoolAILabs/fitcode2prompt.git
cd fitcode2prompt
uv sync
```

### 2. Set up API Keys

fitcode2prompt uses LiteLLM for model access. You can either use a `.env` file or set environment variables directly.

**Required**: `OPENAI_API_KEY` for GPT models

fitcode2prompt uses LiteLLM, so other providers may work. [See LiteLLM docs](https://docs.litellm.ai/docs/providers) for details.
