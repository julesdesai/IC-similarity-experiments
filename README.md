# Graph Merger: Philosophical Position Similarity Analysis

A comprehensive toolkit for analyzing semantic similarity between philosophical positions using multiple embedding-based approaches, designed to evaluate position merging algorithms and detect false positives/negatives in philosophical reasoning systems.

## Overview

This project provides three complementary similarity analysis tools for evaluating philosophical positions:

1. **Basic Cosine Similarity** - Standard embedding-based similarity using OpenAI's text embeddings
2. **LLM-Based Similarity** - Advanced semantic similarity using language model evaluation 
3. **Hierarchical Similarity** - Sophisticated analysis that considers both main positions and their objections/children using the Hungarian algorithm for optimal matching

## Features

- 🔍 **Multi-modal Similarity Analysis**: Three different approaches to measure philosophical position similarity
- 📊 **Comprehensive Reporting**: Generate detailed JSON, text, and markdown reports
- 🎯 **False Positive/Negative Detection**: Specialized evaluation for position merging algorithms
- 🌳 **Hierarchical Analysis**: Advanced similarity measurement considering position objections and sub-arguments
- 📈 **Statistical Analysis**: Detailed statistics and comparative metrics across all similarity measures
- 🔄 **Batch Processing**: Efficient processing of large evaluation datasets

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd graph-merger
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Basic Similarity Analysis
```bash
python similarity_evaluator.py --experiment both
```

### LLM-Based Similarity Analysis
```bash
python llm_similarity_evaluator.py --experiment both
```

### Hierarchical Similarity Analysis
```bash
python hierarchical_similarity_evaluator.py --experiment both --alpha 0.6
```

## Usage

### Command Line Options

All evaluators support these common options:
- `--experiment {negatives,positives,both}`: Which experiment to run (default: both)
- `--negatives-file`: Path to false negatives file (default: evals/falsenegatives.json)
- `--positives-file`: Path to false positives file (default: evals/falsepositives.json)

#### Hierarchical Similarity Additional Options:
- `--alpha FLOAT`: Weight for node vs children similarity (default: 0.6)
- `--negatives-file`: Path to false negatives with children (default: evals/falsenegativeswithchildren.json)
- `--positives-file`: Path to false positives with children (default: evals/falsepositiveswithchildren.json)

### Examples

```bash
# Run only false negatives analysis
python similarity_evaluator.py --experiment negatives

# Run hierarchical analysis with custom alpha weight
python hierarchical_similarity_evaluator.py --alpha 0.8

# Run LLM analysis on custom dataset
python llm_similarity_evaluator.py --negatives-file custom_negatives.json
```

## Data Format

### Basic Evaluation Data
False negatives and false positives files should follow this structure:

```json
{
  "false_negative_pairs": [
    {
      "pair_id": "fn_001",
      "original_position": {
        "position_id": "act_utilitarianism",
        "summary": "Act Utilitarianism",
        "theses": ["thesis1", "thesis2", ...]
      },
      "reformulated_position": {
        "position_id": "act_utilitarianism_reformulated",
        "summary": "Direct Outcome-Based Ethics",
        "theses": ["reformulated thesis1", ...],
        "reformulation_strategies": ["strategy1", "strategy2"]
      }
    }
  ]
}
```

### Hierarchical Data (with Children)
For hierarchical analysis, positions include objections/children:

```json
{
  "original_position": {
    "summary": "Position summary",
    "theses": ["thesis1", "thesis2"],
    "children": [
      {
        "summary": "Objection 1",
        "content": "Detailed objection content"
      }
    ]
  }
}
```

## Output Files

All results are saved to `evals/results/` directory:

### Generated Files
- **JSON Results**: `{method}_similarity_results_{timestamp}.json` - Detailed results with embeddings and metadata
- **Text Reports**: `{method}_similarity_report_{timestamp}.txt` - Human-readable analysis reports  
- **Markdown Reports**: `{method}_similarity_report_{timestamp}.md` - Formatted reports for documentation

### Report Contents
- Summary statistics (mean, std dev, min, max, median)
- Detailed pair-by-pair analysis
- Results sorted by similarity score
- Results in original order
- Comparative analysis between false negatives and false positives

## Similarity Methods

### 1. Basic Cosine Similarity (`similarity_evaluator.py`)
- Uses OpenAI's `text-embedding-3-large` model
- Calculates cosine similarity between position embeddings
- Fast and efficient for large datasets
- Baseline similarity measurement

### 2. LLM-Based Similarity (`llm_similarity_evaluator.py`)
- Uses language models for semantic similarity assessment
- Supports multiple OpenAI models (GPT-4, GPT-4 mini, etc.)
- More nuanced understanding of philosophical concepts
- Slower but more sophisticated analysis

### 3. Hierarchical Similarity (`hierarchical_similarity_evaluator.py`)
- Considers both main positions and their objections/children
- Uses Hungarian algorithm for optimal objection matching
- Weighted combination: `α × node_similarity + (1-α) × children_similarity`
- Most comprehensive analysis for complex philosophical arguments

## Project Structure

```
graph-merger/
├── similarity_evaluator.py           # Basic cosine similarity analysis
├── llm_similarity_evaluator.py       # LLM-based similarity analysis  
├── hierarchical_similarity_evaluator.py  # Hierarchical similarity analysis
├── generate_children_datasets.py     # Utility for generating hierarchical datasets
├── evals/                            # Evaluation datasets
│   ├── falsenegatives.json          # Basic false negatives
│   ├── falsepositives.json          # Basic false positives
│   ├── falsenegativeswithchildren.json  # Hierarchical false negatives
│   ├── falsepositiveswithchildren.json  # Hierarchical false positives
│   ├── prompts/                      # Evaluation prompts
│   └── results/                      # Generated analysis results
├── prompts/                          # System prompts for LLM analysis
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Dependencies

- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities (cosine similarity)
- **scipy**: Scientific computing (Hungarian algorithm)
- **openai**: OpenAI API client for embeddings and LLM calls
- **python-dotenv**: Environment variable management
