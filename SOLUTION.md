# SOLUTION

## 1. Design
A ReAct (Reason + Act) loop in `tam_agent/agent.py`, with no framework. At each step the LLM replies with JSON: either a tool call (`action`, `input`, `thought`) or the final decision. We run the tool, send back a short observation plus the tool data, and the LLM decides what to do next.

```
ticket -> LLM -> tool call -> Python tool -> observation -> LLM -> ... -> final decision -> validate -> output
```

**Tools** (`tam_agent/tools.py`, plain functions over pandas/KB data):
| Tool | What it does |
|---|---|
| `lookup_account` | Resolves the account by id, else by email domain (unique, so similar company names don't fool it). Returns tier, days to renewal, escalations and recent SSO changes as signals. |
| `get_usage_metrics` | 3-month active-user and document trend; flags a decline of more than 25% (KB-2914 override). |
| `search_kb` | Exact error-code match first, then keyword search with IDF weighting plus a module boost; optional `kb_type` filter (e.g. `limit`). Deprecated articles are replaced by their `superseded_by` article. Returns the top 5 as short excerpts. |
| `get_kb_article` | Full text of one article plus its `related` ids, so the agent can follow links (e.g. KB-4795 -> KB-7889). |
| `check_related_tickets` | The account's recent tickets; flags an open or pending ticket with the same module or error from the last 7 days (KB-5930). |

The three policy articles (KB-2914, KB-5930, KB-6401) apply to every ticket, so they are summarised in the system prompt. Everything else is retrieved. The agent picks its own path: the injection ticket used 1 lookup, the E-3310 ticket used 3 (search -> read article -> account, which revealed the SSO change).
**Stopping:** the LLM stops when it emits `final_decision`. A hard cap of 6 tool calls forces a decision. Safety nets: tolerant JSON parsing, `kb_references` limited to articles actually retrieved (so it cannot cite invented ids), schema validation with safe defaults, hard KB-2914 rules enforced in code (P1 or security means escalate), and ticket text treated as untrusted (prompt-injection rule).

## 2. Evaluation
- Gold set of about 300 tickets labelled by senior TAMs; per-field accuracy for category, priority, route, escalate and duplicate_of, with confusion matrices.
- Cost-weighted priority errors: under-prioritising a P1 counts far more than over-prioritising.
- Hard gates: 100% recall on security/data-exposure tickets, and a regression suite of prompt-injection tickets.
- Retrieval: recall@5 and precision of `kb_references` against the gold articles; duplicate detection precision and recall.
- Reply quality: rubric (human + LLM judge) checking for made-up steps and any other customer's data.
- Efficiency: tool calls, tokens, latency and cost per ticket; check that confidence is calibrated.
- Shadow mode next to human triagers before go-live; review the disagreements.

## 3. Scale (30k tickets, 4k articles)
- **Cost:** grows per ticket, not with data size. Tools return top-k results, so the prompt stays at a few thousand tokens and about 3-5 LLM calls. Prompt caching of the system prompt and a cheaper model for simple tickets would cut cost further.
- **Latency:** today's tools scan pandas tables and every KB article on each call (O(N)). Fine at 4k articles, but I would index tickets by account_id, pre-compute account health nightly, precompute IDF/BM25 indexes at load time, and run tickets in parallel workers with rate-limit handling.
- **Retrieval quality:** this breaks first. More articles means more near-duplicates and outdated versions, and keyword search misses meaning ("greyed out" vs "disables the Approve button"). I would move to hybrid search (BM25 + embeddings in a vector store), filter on module/status/type, add a reranker, prefer `current`/`active_incident` articles, and track recall@k on the gold set.

## 4. Limitations
- Keyword retrieval only; no semantic search, so the agent sometimes needs a second search.
- Policies live in the prompt; a policy change needs a prompt change.
- Duplicate check uses the dataset's "today" (2026-10-20), not the ticket's creation time, and relies on the LLM to judge "same symptom".
- No native tool-calling API (JSON-in-text), and the results vary between runs on a small free model.
- Confidence is self-reported by the LLM and not calibrated.

## 5. How to run
```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set LLM_PROVIDER, LLM_MODEL, API key (OPENAI_BASE_URL for OpenAI-compatible APIs)
python scripts/validate_data.py
python run.py --set review
python scripts/check_outputs.py
```
