# Titanic Dataset Chat Agent

A production-ready chatbot that answers natural language questions about the Titanic
passenger dataset, generates visualizations, and displays everything in a Streamlit UI.

Built with FastAPI, LangChain (ReAct agent), Groq LLM, and Streamlit.

---

## Architecture

```
User → Streamlit UI → FastAPI Backend → LangChain ReAct Agent → Pandas / Plotting Tools
                                                ↓
                                        Groq LLM (llama-3.3-70b-versatile)
```

---

## Project Structure

```
titanic_agent/
├── backend/
│   ├── config.py          # Environment-driven settings
│   ├── data_loader.py     # Dataset fetch, validation, cleaning
│   ├── tools.py           # LangChain tools - stats + plotting
│   ├── agent.py           # ReAct agent setup
│   └── main.py            # FastAPI app - POST /chat
├── frontend/
│   └── streamlit_app.py   # Streamlit chat UI
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/GURRALASAIHANEESH/titanic_agent.git
cd titanic_agent
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0

BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501

AGENT_MAX_ITERATIONS=5
AGENT_VERBOSE=false

BACKEND_URL=http://localhost:8000
```

Get a free Groq API key at: https://console.groq.com

---

## Running Locally

### Start the backend — Terminal 1

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify the backend is running at: http://localhost:8000/health

### Start the frontend — Terminal 2

```bash
cd frontend
streamlit run streamlit_app.py
```

Open the UI at: http://localhost:8501

---

## API Reference

### `POST /chat`

**Request body:**

```json
{
  "question": "What percentage of passengers were male?"
}
```

**Response:**

```json
{
  "answer": "35.24% of passengers were female and 64.76% were male.",
  "has_chart": false,
  "chart_base64": null,
  "processing_time_ms": 1647.6
}
```

### `GET /health`

```json
{
  "status": "ok",
  "dataset_rows": 891,
  "dataset_columns": 15,
  "model": "llama-3.3-70b-versatile"
}
```

---

## Example Questions

| Question | Response Type |
|---|---|
| What percentage of passengers were male? | Text |
| Show me a histogram of passenger ages | Chart + Text |
| What was the average ticket fare? | Text |
| How many passengers embarked from each port? | Text |
| Show survival rate by gender | Chart + Text |
| Show a bar chart of passenger class distribution | Chart + Text |
| What was the survival rate by passenger class? | Text |
| Show the correlation between all features | Chart + Text |
| What was the median age of survivors? | Text |
| How many passengers traveled alone? | Text |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| Agent framework | LangChain 0.3.7 ReAct agent |
| LLM | Groq — llama-3.3-70b-versatile (free tier) |
| Data processing | Pandas, NumPy |
| Visualizations | Matplotlib, Seaborn |
| Frontend | Streamlit 1.39.0 |
| Language | Python 3.12 |

---

## Dataset

The Titanic dataset is downloaded automatically at startup from:
`https://raw.githubusercontent.com/datasciencedboys/datasets/master/titanic.csv`

891 rows, 12 original columns. Three derived columns are added at load time:

- **AgeGroup** — age band categorization
- **FamilySize** — SibSp + Parch + 1
- **IsAlone** — 1 if traveling alone, 0 otherwise

---

## Notes

- No API keys are hardcoded. All secrets are loaded from the `.env` file.
- The dataset is cached in memory after the first load — no repeated downloads.
- Charts are returned as base64-encoded PNG strings in the API response.
- The agent uses a ReAct loop with a maximum of 5 iterations per query.
- Out-of-scope questions (movies, general knowledge) are rejected immediately
  without hitting the LLM API.
