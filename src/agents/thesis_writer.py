"""LLM thesis writer — constrained narration layer (see earlier turns for full rationale)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SYSTEM_PROMPT = """You are writing a short move-explanation note for an equity research desk.
Only reference ACCEPTED evidence as support. Never invent a cause not in the provided lists.
State the confidence score exactly as given. If unexplained=True, say plainly that no evidence
explains the move. Keep it to one tight paragraph."""

def write_narrative(result, model="claude-sonnet-4-6"):
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    client = anthropic.Anthropic(api_key=api_key)
    m = result.move
    lines = [f"Ticker: {m.ticker}", f"Confidence: {result.confidence:.0%}", f"Unexplained: {result.unexplained}"]
    for e in result.accepted_evidence:
        lines.append(f"ACCEPTED [{e.source_agent}]: {e.raw_text}")
    response = client.messages.create(model=model, max_tokens=400, system=SYSTEM_PROMPT,
                                       messages=[{"role": "user", "content": "\n".join(lines)}])
    return response.content[0].text

def write_narrative_offline(result):
    m = result.move
    if result.unexplained:
        return (f"{m.ticker} moved {m.raw_return:+.2%} over {m.start_date}–{m.end_date} "
                f"({m.residual_return:+.2%} idiosyncratic, z={m.residual_zscore:+.2f}), but no "
                f"evidence considered sufficiently explains it (confidence {result.confidence:.0%}). "
                f"This move should be flagged for manual analyst review rather than assumed explained.")
    if not result.accepted_evidence:
        return (f"{m.ticker}'s {m.raw_return:+.2%} move over {m.start_date}–{m.end_date} was "
                f"{m.systematic_fraction:.0%} explained by market/sector factors alone; the residual "
                f"was within normal noise for this stock, so no stock-specific story is warranted.")
    sector_note = "" if m.has_sector_factor else " (sector factor unavailable for this stock — market-only decomposition)"
    n_ev = len(result.accepted_evidence)
    return (f"{m.ticker} moved {m.raw_return:+.2%} over {m.start_date}–{m.end_date} "
            f"({m.systematic_fraction:.0%} explained by market/sector factors{sector_note}). The residual "
            f"{m.residual_return:+.2%} move (z={m.residual_zscore:+.2f}) is explained with "
            f"{result.confidence:.0%} confidence by {result.corroboration_count} corroborating "
            f"source type(s) ({n_ev} accepted evidence item{'s' if n_ev != 1 else ''}).")
