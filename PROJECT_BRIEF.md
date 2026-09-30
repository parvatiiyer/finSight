# FinSight — Project Context & Improvement Brief

This document is written to be handed to another AI coding tool (or a new
engineer) with zero prior context. It explains what the project is, why it's
built the way it is, what exists today, what's known to be broken or
missing, and concrete suggestions for models, architecture, and testing.

---

## 1. What this project actually is

FinSight is **not** a stock-prediction tool. It's a research-desk tool that
answers one narrow question well: *"why did this stock move, and how sure
are we?"* — with the causal explanation checked against evidence before
it's ever shown, rather than an LLM freely narrating a plausible-sounding
story.

The whole design exists to solve one specific, well-known failure mode:
**an LLM asked "why did this stock move" will confidently attribute it to
any nearby headline, related or not.** Every architectural decision below
traces back to preventing that.

### Non-negotiable design invariants — preserve these in any changes

1. **Evidence proposal and evidence judgment are different steps, done by
   different things.** Agents (news/fundamentals/technical) *propose*
   evidence as structured, falsifiable claims — a date, a direction, a
   magnitude estimate. They never get to decide something is "the cause."
2. **The adjudicator is deterministic, not an LLM.** It checks each piece
   of evidence against three explicit gates — temporal alignment,
   magnitude adequacy, directional consistency — before it's "accepted."
   This must stay auditable: every confidence number must be traceable
   back to which evidence passed which gate and by how much.
3. **The system is allowed to say "unexplained."** If nothing clears the
   bar, it reports that plainly rather than manufacturing a story. Never
   change this to always produce a narrative.
4. **The LLM only narrates already-decided facts.** `thesis_writer.py`'s
   system prompt explicitly forbids citing anything the adjudicator didn't
   already accept. Any new narration feature must preserve this
   constraint.
5. **A move is factor-decomposed before anyone tries to explain it.** Only
   the residual (idiosyncratic) return — what's left after removing
   market and sector effects — is something an agent is asked to explain.
   This stops the system from writing a stock-specific story about a move
   that was actually just the whole market moving.

---

## 2. Architecture & file map (current state)

```
config/universe.py
    25 curated NSE large-caps across 6 sectors, for offline demo/testing.
    Also: YFINANCE_SECTOR_MAP, mapping yfinance's free-text .info['sector']
    (e.g. "Technology", "Financial Services") onto NSE sector indices, so
    an ARBITRARY (non-curated) ticker can still get a sector factor.

src/data/
    ingest.py            Production yfinance ingestion: OHLCV, fundamentals,
                         quarterly fundamentals history (for ratio-delta
                         evidence), earnings dates, company sector lookup.
                         Retries with backoff, caches to parquet.
    resolve_ticker.py    Turns ANY typed input into a validated ticker.
                         Layered: (1) curated exact ticker match, offline,
                         (2) curated fuzzy name match, offline, (3) direct
                         yfinance validation (as typed, then with .NS/.BO
                         suffixes), (4) yfinance name search (yf.Search).
    synthetic.py         Dev-only single-stock OHLCV/fundamentals fixture
                         (GBM + volatility clustering). Not used in
                         production; exists because this environment can't
                         always reach live data.
    synthetic_factors.py Dev-only MULTI-stock, factor-driven fixture: a
                         market index, sector indices, and stock returns
                         mechanically driven by them, PLUS injected labeled
                         idiosyncratic events (a fake headline + a
                         corresponding fake price shock on a known date).
                         This is what lets tests check the adjudicator
                         actually recovers a known cause and rejects
                         planted noise — not just "the code runs."

src/features/
    technical.py         ~20 technical indicators, all strictly causal
                         (backward-looking only — no lookahead).
    fundamental.py       Fundamental ratios + derived metrics (net debt,
                         FCF yield) from a point-in-time snapshot.
    pipeline.py          Combines technical + fundamental into one feature
                         table per ticker; also builds forward-return
                         labels (up/flat/down) for a NEVER-TRAINED
                         XGBoost classifier — see Section 5.
    factor_model.py      THE core quant piece. decompose_move() splits a
                         return into market-explained + sector-explained +
                         residual via rolling OLS. Falls back to a
                         market-only single-factor model when no sector
                         index applies. Handles MISMATCHED trading
                         calendars/timezones (see Section 4 — this bit
                         hard, twice). find_most_significant_window() scans
                         recent history for the most statistically unusual
                         move.

src/agents/
    evidence_types.py    Shared contracts: Evidence, MoveWindow,
                         EvidenceVerdict (now carries `contribution`, its
                         literal share of the confidence score),
                         AdjudicationResult.
    news_agent.py        Synthetic path (ground-truth events, for testing)
                         + a REAL production path: Google News RSS search
                         (no API key needed) + a lightweight keyword-based
                         sentiment heuristic (NOT FinBERT yet — see
                         Section 6).
    fundamentals_agent.py Earnings-date proximity + quarter-over-quarter
                         ratio-delta evidence (ROE, margins, debt/equity,
                         current ratio, revenue growth).
    technical_agent.py   Volume spikes, RSI extremes, Bollinger Band
                         breakouts — mostly NEUTRAL-direction evidence
                         (confirms a move was "real," doesn't say why).
    adjudicator.py        THE deterministic scorer. Three gates per
                         evidence item (temporal_alignment_score,
                         magnitude_adequate, directional_consistency).
                         Confidence = directional_strength × (1 +
                         corroboration_bonus), capped at 1.0.
                         corroboration_bonus rewards independent AGENT
                         TYPES agreeing, not raw evidence count.
    thesis_writer.py      write_narrative() = real Claude API call
                         (untested live in this sandbox — needs
                         ANTHROPIC_API_KEY). write_narrative_offline() =
                         deterministic template fallback, same
                         evidence-only constraint.
    orchestrator.py       explain_move() = synthetic-data path (demo/test).
                         explain_move_live() = the "any real stock" path:
                         resolves sector, fetches real data, gathers real
                         evidence, adjudicates. Runs the 3 evidence agents
                         concurrently (ThreadPoolExecutor).

src/api/main.py
    FastAPI. GET /health, /universe, /resolve?query=, /explain?query=
    (live, any stock), /explain_demo?ticker= (synthetic, curated universe
    only — zero-setup demo path).

frontend/dashboard.html
    Single self-contained file (vanilla JS + Plotly.js via CDN). Beige/
    white palette, minimal typography. Landing view (single search input,
    optional date-range toggle) → report view (Plotly waterfall for
    decomposition, Plotly scatter for evidence timeline, accepted/rejected
    evidence ledger with per-item confidence CONTRIBUTION shown, a
    plain-language "why is confidence X%, not higher" diagnostics panel
    that references specific evidence and offers a one-click "widen
    window" action). Falls back to 3 bundled example reports when no
    backend is reachable.

tests/
    test_no_leakage.py    Walk-forward CV correctness (no train/test
                         boundary leakage).
    test_adjudicator.py   The three gates + confidence formula, including
                         rejection of off-topic/mistimed evidence.
    test_factor_model.py  Calendar/timezone alignment regression tests
                         (this is where two real production bugs were
                         found and fixed — see Section 4).

src/models/walk_forward.py
    Rolling-origin (walk-forward) CV harness — built, tested, NEVER WIRED
    to an actual model. See Section 5.
```

---

## 3. Tech stack currently in use

- **Data:** yfinance (OHLCV, fundamentals, quarterly financials, earnings
  dates), Google News RSS (no key required)
- **Quant:** pandas/numpy (OLS via `np.linalg.lstsq`), `ta` library for
  technical indicators
- **Backend:** FastAPI + uvicorn
- **LLM:** Anthropic API (`anthropic` Python SDK) — wired but never
  exercised live in this build environment (no key configured here)
- **Frontend:** vanilla JS, Plotly.js (CDN), no build step, no framework
- **Testing:** pytest

## 4. Two real production bugs already found and fixed — don't reintroduce these

1. **Boolean mask cross-contamination.** `decompose_move()` built a
   window mask from `stock_returns.index` and applied it to
   `market_returns`, which can have a different length (different
   national trading calendars — a US stock has ~500 trading days over 2
   years, the NSEI Indian index has ~494). Fixed by aligning all series
   onto a shared, timezone-naive, calendar-date index (`_align_returns()`)
   before any cross-series operation.
2. **The same bug, independently, in `estimate_betas()`** — it sliced
   stock/market/sector Series by *integer position*, which silently picks
   different calendar dates from each Series the moment calendars diverge.
   Fixing only `decompose_move()` didn't fix this — it's a separate
   function with its own copy of the same mistake. **Lesson embedded in
   the code now:** alignment logic lives in exactly one place
   (`_align_returns()`), used by both functions, specifically so this
   class of bug can't reappear by drifting out of sync.
3. **A third variant in `orchestrator.py`**: sector returns were
   `.reindex(stock_returns.index).ffill()`'d onto the stock's own
   (differently-timezoned) index before ever reaching the factor model —
   which for two tz-aware DatetimeIndexes matches by exact UTC instant, so
   it silently produced an ALL-NaN sector series for any foreign-market
   ticker. Fixed by not pre-aligning at all — `_align_returns()` already
   does this correctly.

**Takeaway for whoever extends this:** any new code that combines two or
more return Series MUST go through `_align_returns()`. Do not add a new
`.reindex()`, boolean mask, or `.iloc[]` positional slice across multiple
Series anywhere in this codebase.

## 5. Known gaps — planned in the original spec, never built

- **XGBoost return-bucket classifier.** The walk-forward CV harness
  (`src/models/walk_forward.py`) and the labels
  (`build_return_bucket_labels` in `technical.py`) exist and are tested.
  No model has actually been trained. This was deliberately deprioritized
  in favor of building the evidence-attribution system, which is the
  actual novel contribution of this project.
- **GARCH(1,1) volatility model** — planned, not built. Would replace or
  augment the current `realized_vol_20` technical feature.
- **Isolation Forest / anomaly detection** — planned, not built. The
  `technical_agent.py`'s volume-ratio and RSI-extreme checks are a partial,
  much cruder substitute.
- **RAG over annual report PDFs** (Chroma/FAISS + embeddings) — not built
  at all. No annual-report ingestion exists yet.
- **The "monitoring loop"** — re-running analysis over time and diffing
  old vs. new explanations, attributing *changes* in confidence/evidence
  to specific new information ("prior explanation cited NPA concerns;
  this quarter's earnings show NPA improved — partially invalidated").
  This is the single most valuable missing feature and is exactly what a
  graph database is good at — see Section 7.
- **Peer-comparison mode.** `config/universe.py`'s `peers_of()` helper
  exists; nothing consumes it yet.
- **Strategy backtest** (long stocks with confidence/health above a
  threshold vs. buy-and-hold) — not built.
- **FinBERT sentiment** — `news_agent.py` currently uses a hand-rolled
  keyword list (`_keyword_polarity`), explicitly as a placeholder. Swap
  point is isolated (see Section 6).

## 6. Suggested models, per component

| Component | Current | Suggested upgrade | Why |
|---|---|---|---|
| News sentiment | Keyword list (`_keyword_polarity`) | `ProsusAI/finbert` via `transformers` | Already stubbed as `score_headline_sentiment_production()`; keyword matching misses negation, sarcasm, and financial jargon FinBERT was trained on |
| Sector inference for arbitrary tickers | Static dict (`YFINANCE_SECTOR_MAP`) | Sentence-embedding similarity (e.g. `sentence-transformers/all-MiniLM-L6-v2`) between the company's business description and each sector index's description | The static map only covers ~6 broad Yahoo sector strings; real coverage gaps exist for Real Estate, Basic Materials, Communication Services, etc. — these silently fall to market-only right now |
| Thesis narration | Claude via Anthropic API (wired, untested live) | Keep Claude; consider a smaller/cheaper model for this task since the prompt is short and heavily constrained | Narration is low-complexity by design (see invariant #4) — doesn't need a frontier model |
| Return classification | Not built | XGBoost (as originally planned) on the existing feature table, evaluated ONLY via `walk_forward.py`'s CV, never random k-fold | Framing as direction/return-bucket classification is more defensible than raw price regression; infrastructure already exists |
| Volatility | `realized_vol_20` (rolling std) | GARCH(1,1) via the `arch` package | Captures volatility clustering that a simple rolling std misses; complements rather than replaces the current feature |
| Anomaly detection | None (proxied by technical_agent's heuristics) | Isolation Forest on volume/price/volatility jointly | Cheap, unsupervised, gives a principled "unusual day" flag instead of hand-tuned thresholds |
| RAG over filings | Not built | Chroma (simplest to stand up) + any local embedding model, chunked by filing section | Needed before annual-report evidence can exist at all |

## 7. Where a graph database actually helps — the monitoring loop

This is the single highest-value architectural addition, and it's
specifically what's needed for the never-built "monitoring loop": tracking
how an explanation for a stock's move evolves as new evidence arrives, and
being able to say "this earlier explanation was later contradicted."

**Suggested: Neo4j** (or a lighter embedded option — **Kuzu** or even
**NetworkX** for a first pass, if a hosted graph DB is overkill for now).

Model it as:

```
(:Company {ticker, name, sector})
(:MoveWindow {start_date, end_date, raw_return, residual_zscore, confidence})
(:Evidence {source_agent, event_date, direction, magnitude_hint, raw_text, contribution})

(:Company)-[:HAD_MOVE]->(:MoveWindow)
(:MoveWindow)-[:EXPLAINED_BY]->(:Evidence)          // accepted evidence
(:MoveWindow)-[:CONSIDERED_AND_REJECTED]->(:Evidence)  // rejected, with reason as a property
(:Company)-[:PEER_OF]->(:Company)                    // replaces the flat peers_of() dict
(:Evidence)-[:CONTRADICTS]->(:Evidence)              // NEW: a later fact invalidating an earlier one
(:Evidence)-[:CORROBORATES]->(:Evidence)             // cross-source agreement, explicit as an edge
```

This directly enables queries that are painful in the current stateless,
recompute-everything design:
- "Show every move for this stock that was later contradicted by new
  evidence" — literally a graph traversal (`MoveWindow -> Evidence
  <-[:CONTRADICTS]- NewEvidence`).
- "How often does this specific news source's evidence end up
  contradicted?" — a source-reliability signal that could feed back into
  `Evidence.reliability_weight`, which is currently a static number set at
  creation time in each agent.
- Peer-comparison mode falls out almost for free once `PEER_OF` is a real
  edge instead of a static dict lookup.

This is also the natural place to persist analysis history at all — right
now every `/explain` call recomputes everything from scratch; nothing is
stored anywhere.

## 8. Other architecture gaps worth flagging

- **No persistence layer at all.** Every API call is stateless. A
  Postgres table of past `MoveWindow`/`AdjudicationResult` runs (even
  without the graph DB) would already unlock history and the monitoring
  loop's simpler version.
- **No caching beyond `ingest.py`'s parquet file cache.** Repeated
  `/explain` calls for the same ticker/window re-run the entire pipeline.
  A simple Redis or in-memory TTL cache keyed on `(ticker, window)` would
  help before this sees real traffic.
- **Google News RSS is unauthenticated scraping of a public endpoint** —
  fine for a demo, fragile for production. No rate limiting, retry, or
  backoff currently exists on that path (unlike `ingest.py`'s yfinance
  calls, which do have retry/backoff). Worth aligning the two.
- **`find_most_significant_window()` is brute-force** — it re-fits betas
  (a full OLS regression) for every candidate window in the scan (~20-40
  windows over a 90-day lookback). Fine at current scale; would need
  vectorizing or caching trailing betas if this runs across many tickers
  at once.
- **No auth, no multi-user concept.** Fine for a personal/demo tool; a
  concrete gap if this becomes a shared research-desk tool.

## 9. Testing — what exists and what's actually missing

**Exists:**
- `test_no_leakage.py` — walk-forward CV boundary correctness
- `test_adjudicator.py` — the three gates, confidence formula, rejection
  of off-topic/mistimed evidence, corroboration-count vs raw-count
  distinction
- `test_factor_model.py` — calendar/timezone alignment regressions
  (reproduces the exact AAPL-vs-NSEI bug with synthetic mismatched-length,
  mismatched-timezone Series)
- Manual Playwright-driven checks of the dashboard (landing→report
  transition, Plotly chart rendering, error states) — **not yet a
  committed test file**, should be formalized

**Missing, in priority order:**

1. **Confidence calibration testing — the biggest methodological gap.**
   Nothing currently validates that a 70% confidence score is actually
   right 70% of the time. All current tests check that the *formula* is
   computed correctly, not that the *formula means anything predictive*.
   Concrete plan: pull N historically significant moves across M real
   stocks, run the pipeline, hand-label (using hindsight/actual news)
   whether each accepted explanation was correct, and plot a reliability
   diagram (predicted confidence bucket vs. observed correctness rate).
   This is the test that would tell you whether the confidence *number*
   deserves to be trusted, not just whether it's internally consistent.
2. **Formalize the Playwright frontend tests** into a committed suite:
   landing→report transition, date-range validation (start after end,
   one-sided dates), the "widen window" diagnostic action, Plotly chart
   presence, offline-fallback behavior when no backend is reachable.
3. **Live integration tests, network-gated.** Everything in `ingest.py`,
   `resolve_ticker.py`'s layers 3-4, and `news_agent.py`'s production path
   has only been verified to *fail cleanly* in this sandbox (no network
   access to Yahoo Finance / Google News here) — never actually exercised
   against live data. These need a real test run, ideally in CI with a
   dedicated job that has network access, separate from the fast offline
   unit tests.
4. **Property-based/fuzz testing for the adjudicator** (e.g. with
   `hypothesis`): generate random `MoveWindow`/`Evidence` combinations and
   assert invariants hold regardless of input — confidence is always in
   `[0, 1]`, `unexplained` is never `True` when evidence was accepted,
   rejected evidence always has a non-empty `rejection_reason`, etc.
5. **API/frontend contract tests.** The dashboard's JS reads fields like
   `has_sector_factor`, `contribution`, `temporal_alignment` directly off
   the JSON response with no schema validation — a backend field rename
   would silently break the frontend. A shared Pydantic schema (or JSON
   Schema) checked on both sides would catch this before it ships.
6. **Data-quality tests against yfinance.** yfinance's `.info` schema
   drifts between versions and sometimes returns partially-empty data for
   valid tickers. No test currently checks for this — `fetch_fundamentals`
   would silently store a bunch of `None` fields.
7. **Load/performance test on `/explain`** for a cross-section of
   tickers, given the brute-force window scan noted in Section 8.

---

## 10. Suggested priority order for a next work session

1. Confidence calibration test harness (Section 9.1) — this determines
   whether the core product claim ("we tell you how sure to be") is true.
2. Formalize the frontend Playwright suite (Section 9.2) — cheap, prevents
   regressions in a file that's grown large and is edited by hand.
3. Wire FinBERT into `news_agent.py` (Section 6) — isolated, well-scoped
   swap, meaningfully improves evidence quality.
4. Persistence layer (Postgres, even without the graph DB yet) — unlocks
   history, which unlocks the monitoring loop, which is the most valuable
   unbuilt feature.
5. Graph DB modeling for the monitoring loop (Section 7) — build once
   persistence exists and there's real history to model relationships
   over.
