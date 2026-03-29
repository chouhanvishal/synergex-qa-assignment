# Synergex Med — AI-Powered Call QA API

A production-ready FastAPI service that receives phone call transcripts and returns structured quality analysis using an LLM. Built as a technical assessment for the Synergex Med Senior Python/AI Engineer role.

---

## Quick Start

```bash
# 1. Clone / unzip, enter directory
cd synergex-qa

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 5. Start the API
uvicorn app.main:app --reload --port 8000

# 6. Test with a sample transcript
curl -X POST http://localhost:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d @sample_transcripts/01_clean_call.json | python3 -m json.tool
```

Interactive docs available at: `http://localhost:8000/docs`

---

## Endpoints

### `POST /analyze-call`
Analyze a single call transcript.

**Request body:**
```json
{
  "call_id": "CALL-001",
  "agent_name": "Maria Santos",
  "call_date": "2025-06-10",
  "call_duration_seconds": 187,
  "department": "Scheduling",
  "transcript": "Agent: Thank you for calling...\nCaller: Hi, I need..."
}
```

**Response:** Structured `QAAnalysis` JSON (see `app/schema.py` for full schema).

### `POST /batch-analyze`
Analyze a list of transcripts concurrently.

**Request body:**
```json
{ "calls": [ { ...transcript 1... }, { ...transcript 2... } ] }
```

### `GET /health`
Returns `{"status": "ok"}`.

---

## How It Works

### Architecture

```
POST /analyze-call
       │
       ▼
CallTranscript (Pydantic validation)
       │
       ▼
build_user_prompt()   ← department-aware rules injected here
       │
       ▼
llm_complete()        ← provider abstraction (Anthropic or OpenAI)
  │  retry logic (3 attempts, exponential back-off)
       │
       ▼
_parse_llm_response() ← strips markdown fences, JSON.parse, Pydantic validation
       │
       ▼
_enforce_business_rules() ← ensures escalation consistency
       │
       ▼
QAAnalysis (returned to caller)
```

### Prompting Strategy

The system is designed around **two competing failure modes**:

1. **False positives** — flagging minor imperfections as serious issues, damaging staff morale (the explicit problem Synergex is solving).
2. **False negatives** — missing genuine HIPAA violations or dangerous misinformation.

The prompt addresses both:

- **Clear severity hierarchy**: The system prompt defines exactly what qualifies as `critical` (confirmed HIPAA violation, dangerous misinformation, explicit hostility) vs `minor` (awkward phrasing, small omission with no impact). This prevents the model from reaching for "escalate" on routine calls.
- **Evidence-first instruction**: The model is explicitly told *"only flag what you can directly observe"* and *"if ambiguous, note the ambiguity — do not assume the worst."* This reduces hallucinated issues.
- **Department-specific rules**: Each department has tailored quality criteria injected at prompt time. A Records call checks HIPAA identity verification; a Scheduling call checks appointment confirmation. This grounds the model in concrete, observable criteria rather than abstract quality.
- **Duration notes**: Very short calls (< 30 s) receive a note to keep assessments tentative, preventing over-confident analysis from minimal data.
- **Temperature = 0**: Deterministic outputs for reproducible QA.
- **JSON-only instruction**: The model is told to return raw JSON with no markdown. Pydantic validates the schema; the parser strips fences defensively if the model adds them anyway.

### Edge Case Handling

| Scenario | How it's handled |
|---|---|
| Very short call (< 30 s) | Duration note injected into prompt; model told to be tentative |
| Transcript with no issues | Model instructed not to invent issues; `compliance_flags` may be empty or contain `positive_interaction` only |
| Ambiguous phrasing | Prompt instructs model to note ambiguity rather than assume the worst |
| LLM returns malformed JSON | Parser strips fences, re-attempts JSON.parse; raises descriptive error with raw preview |
| LLM API failure / timeout | Exponential back-off retry (3 attempts by default, configurable via env) |
| Pydantic validation failure | Clean `ValueError` with field-level details |
| Escalation inconsistency (e.g. model says "pass" but includes a critical flag) | `_enforce_business_rules()` post-processes and corrects the assessment |
| Invalid `call_date` | Pydantic rejects the request unless the value is a real calendar date in `YYYY-MM-DD` |
| `assessment_reasoning` not 2–4 sentences | Response fails schema validation until the model follows the prompt |
| Full LLM prompt and raw response in logs | Set `LOG_LEVEL=DEBUG` (defaults to truncated / metadata only at `INFO`) |

---

## Provider Abstraction

Switch LLM providers with a single env var:

```env
LLM_PROVIDER=anthropic   # default — uses claude-opus-4-5-20251101
LLM_PROVIDER=openai      # uses gpt-4o (or OPENAI_MODEL override)
```

Both providers share the same interface: `llm_complete(system, user) → (content, usage)`. Adding a new provider (e.g. Google Gemini) requires implementing `_call_gemini()` in `app/llm_client.py` and adding it to the dispatch block.

OpenAI calls use `response_format={"type": "json_object"}` for guaranteed JSON output. Anthropic calls rely on explicit prompt instructions (Anthropic does not yet support constrained JSON output via the API in the same way, though structured outputs are available via tool use — a future improvement path).

---

## Running the Eval Script

The eval script loads all sample transcripts, hits the live API, validates schemas, and asserts expected outcomes:

```bash
# API must be running on port 8000
python eval.py

# Custom URL
python eval.py --url http://localhost:8000
```

Expected results:

| Call | Expected | Reason |
|---|---|---|
| CALL-001 | `pass` | Clean scheduling call, all steps followed correctly |
| CALL-002 | `escalate` | HIPAA violation (PHI disclosed before release verified) + unprofessional tone |
| CALL-003 | `pass` | Very short transfer call, no issues observable |

---

## Tradeoffs Made

### Structured outputs via prompt vs. tool use
I used explicit JSON instructions in the system prompt rather than Anthropic's tool-use / function-calling structured output feature. This is simpler to implement and works across both providers with a single prompt. The tradeoff: the model can occasionally add markdown fences despite instructions, so the parser strips them defensively. A production v2 would use tool-calling / `json_schema` response format for strict schema enforcement.

### Single-call analysis (no chain-of-thought scratchpad)
The prompt asks for a single JSON response rather than a CoT scratchpad + final answer. This is faster and cheaper per call. The tradeoff is that the model cannot show its reasoning steps. For a higher-stakes deployment, a two-step approach (reasoning → structured output) would improve reliability on edge cases.

### No streaming
The API returns the full analysis synchronously. For a UI, streaming the response would improve perceived latency. Omitted here as the spec is for a backend API.

### Scores are subjective floats
`professionalism_score`, `accuracy_score`, and `resolution_score` are LLM-generated floats. They are useful as relative signals but should not be treated as precise measurements. In production, calibration against human-labelled examples would be needed.

### No persistent storage
As specified, the endpoint is stateless. In production, each analysis would be persisted to PostgreSQL with the call_id as the primary key, enabling trend analysis, agent dashboards, and audit trails.

---

## Project Structure

```
synergex-qa/
├── app/
│   ├── main.py          # FastAPI app, endpoints
│   ├── schema.py        # Pydantic input/output schemas
│   ├── analyzer.py      # Core analysis pipeline
│   ├── llm_client.py    # Provider abstraction + retry logic
│   ├── prompts.py       # System prompt + user prompt builder + dept rules
│   └── logger.py        # Structured logging setup
├── sample_transcripts/
│   ├── 01_clean_call.json
│   ├── 02_problematic_call.json
│   └── 03_edge_case_short_transfer.json
├── eval.py              # Evaluation + schema validation script
├── requirements.txt
├── .env.example
└── README.md
```
