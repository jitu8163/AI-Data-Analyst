# Autonomous Data Analyst Agent

Production-oriented AI data analyst service built with FastAPI, LangChain OpenAI function-calling agents, pandas, scikit-learn, and matplotlib.

## Architecture

```text
autonomous_analyst/
│
├── main.py                  # FastAPI app and /analyze endpoint
├── config.py                # Runtime configuration
├── agent/
│   ├── agent_builder.py     # OpenAI function-calling agent assembly/execution
│   ├── prompts.py           # Agent system prompt
│   ├── tools.py             # Structured LangChain tools
│
├── ml/
│   ├── trainer.py           # Problem-type detection, preprocessing, training
│   ├── evaluator.py         # Metrics
│
├── utils/
│   ├── data_loader.py       # Secure CSV upload parsing
│   ├── plotting.py          # EDA chart generation
│   ├── validation.py        # File/data/output safety checks
│
├── outputs/                 # Generated charts only
├── requirements.txt
└── README.md
```

## Features

- CSV upload via API
- Automatic EDA (shape, dtypes, missing values, basic stats, column typing)
- Plot generation:
  - Correlation heatmap (if >1 numeric feature)
  - Target distribution plot
  - Top-3 feature distribution plots
- Automatic task detection:
  - Classification: `LogisticRegression`, `RandomForestClassifier`
  - Regression: `LinearRegression`, `RandomForestRegressor`
- 80/20 split, baseline training, and evaluation metrics
- Model comparison (stretch feature) and best-model selection
- Preprocessing summary in API response (dropped noisy columns/rows, final training shape)
- Natural-language explanation of model quality and risks
- Structured JSON response only

## Safety Controls

- No `os.system`, `subprocess`, `eval`, `exec`
- No arbitrary code execution
- No file deletion
- Upload parsing is CSV-only
- Writes restricted to `autonomous_analyst/outputs/`

## Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API key:

```bash
export OPENAI_API_KEY="your_key"
```

Optional runtime settings:

- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_TEMPERATURE` (default: `0.0`)
- `MAX_FILE_SIZE_MB` (default: `25`)
- `MAX_ROWS` (default: `250000`)

## Run

```bash
uvicorn autonomous_analyst.main:app --host 0.0.0.0 --port 8000 --reload
```

## Example API Call

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@/path/to/dataset.csv" \
  -F "target_column=target"
```

If the dataset contains a `target` column, `target_column` is optional.

## Example Output

```json
{
  "eda_summary": "{\"shape\": {\"rows\": 1000, \"columns\": 12}, \"numeric_columns\": [\"age\", \"income\"], \"categorical_columns\": [\"city\"], \"missing_values\": {\"age\": 0}}",
  "plots": [
    "outputs/correlation_heatmap.png",
    "outputs/target_distribution.png",
    "outputs/feature_1_age.png"
  ],
  "problem_type": "classification",
  "model_used": "RandomForestClassifier",
  "metrics": {
    "accuracy": 0.87,
    "f1_score": 0.85,
    "cv_score_mean": 0.84,
    "cv_score_std": 0.03
  },
  "preprocessing_summary": {
    "dropped_feature_columns": [
      "Unnamed: 0"
    ],
    "dropped_target_rows_count": 0,
    "final_training_rows": 1000,
    "final_training_features": 11
  },
  "explanation": "Accuracy=0.870, F1=0.850, indicating moderate predictive quality. Overfitting risk: Tree-based models can overfit on small datasets; validate with cross-validation. Key features: age (0.320), income (0.210), city_NewYork (0.110)."
}
```

## Workflow

1. Load CSV from upload
2. Validate input/constraints
3. Agent calls tools in sequence using function-calling:
   - `analyze_dataframe`
   - `generate_plots`
   - `train_model`
   - `explain_model`
4. API returns deterministic structured JSON assembled from tool outputs
# AI-Data-Analyst
# AI-Data-Analyst
