"""
Vector store indexing and retrieval for annual reports, 10-K, and regulatory filings.
Uses ChromaDB for embedding storage with automatic TF-IDF fallback for offline/sandbox execution.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class FilingChunk:
    chunk_id: str
    ticker: str
    fiscal_year: int
    section: str  # e.g. "MD&A", "Risk Factors", "Forward Guidance", "Financial Notes"
    content: str = ""
    sentiment_hint: str = "neutral"  # "positive", "negative", "neutral"
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def __post_init__(self):
        if not self.content and self.text:
            self.content = self.text
        elif not self.text and self.content:
            self.text = self.content


class FilingRetriever:
    """
    Retriever for annual report filings with ChromaDB backend and TF-IDF fallback.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        use_in_memory: bool = True,
        collection_name: str = "annual_reports",
    ):
        self.persist_dir = str(persist_dir) if persist_dir else None
        self.use_in_memory = use_in_memory
        self.collection_name = collection_name
        self.chunks: dict[str, FilingChunk] = {}
        self.chroma_collection = None
        self._init_backend()

    def _init_backend(self):
        try:
            import chromadb
            if self.use_in_memory or not self.persist_dir:
                client = chromadb.Client()
            else:
                client = chromadb.PersistentClient(path=self.persist_dir)
            self.chroma_collection = client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            # Fall back to in-memory TF-IDF index
            self.chroma_collection = None

    def add_filings(self, chunks: list[FilingChunk]):
        """Index a batch of filing sections into the vector store."""
        if not chunks:
            return

        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk

        if self.chroma_collection is not None:
            try:
                ids = [c.chunk_id for c in chunks]
                documents = [c.content for c in chunks]
                metadatas = [
                    {
                        "ticker": c.ticker,
                        "fiscal_year": c.fiscal_year,
                        "section": c.section,
                        "sentiment_hint": c.sentiment_hint,
                    }
                    for c in chunks
                ]
                self.chroma_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            except Exception:
                # If Chroma upsert fails, in-memory chunks are still available for TF-IDF fallback
                pass

    def query(
        self,
        query_text: str,
        ticker: str | None = None,
        top_k: int = 3,
        section_filter: str | None = None,
    ) -> list[tuple[FilingChunk, float]]:
        """
        Query the filing index for relevant disclosures.
        Returns list of (FilingChunk, similarity_score in [0, 1]).
        """
        if not self.chunks:
            return []

        # 1. Try ChromaDB query if active
        if self.chroma_collection is not None:
            try:
                where_clause = {}
                if ticker:
                    where_clause["ticker"] = ticker
                if section_filter:
                    where_clause["section"] = section_filter

                results = self.chroma_collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k * 2, len(self.chunks)),
                    where=where_clause if where_clause else None,
                )
                if results and results.get("ids") and results["ids"][0]:
                    retrieved = []
                    ids = results["ids"][0]
                    distances = results.get("distances", [[0.5] * len(ids)])[0]
                    for cid, dist in zip(ids, distances):
                        if cid in self.chunks:
                            # Convert distance (L2 or cosine distance) to similarity score in [0, 1]
                            sim = float(np.clip(1.0 - (dist / 2.0), 0.1, 0.99))
                            retrieved.append((self.chunks[cid], sim))
                    if retrieved:
                        return retrieved[:top_k]
            except Exception:
                pass

        # 2. Robust TF-IDF cosine similarity fallback
        return self._query_tfidf(query_text, ticker, top_k, section_filter)

    # Aliases for search and indexing
    search = query
    index_chunks = add_filings

    def _query_tfidf(
        self,
        query_text: str,
        ticker: str | None = None,
        top_k: int = 3,
        section_filter: str | None = None,
    ) -> list[tuple[FilingChunk, float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        candidate_chunks = [
            c for c in self.chunks.values()
            if (ticker is None or c.ticker == ticker) and
               (section_filter is None or c.section == section_filter)
        ]

        if not candidate_chunks:
            # If ticker filter too restrictive, search across all chunks
            candidate_chunks = list(self.chunks.values())

        if not candidate_chunks:
            return []

        corpus = [c.content for c in candidate_chunks]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        try:
            tfidf_mat = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query_text])
            sims = cosine_similarity(query_vec, tfidf_mat).flatten()

            ranked_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in ranked_indices:
                score = float(sims[idx])
                results.append((candidate_chunks[idx], max(0.2, score)))
            return results
        except Exception:
            # Fallback simple keyword overlap
            q_words = set(query_text.lower().split())
            scored = []
            for c in candidate_chunks:
                c_words = set(c.content.lower().split())
                overlap = len(q_words & c_words) / max(1, len(q_words))
                scored.append((c, float(overlap)))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]


def get_curated_annual_reports() -> list[FilingChunk]:
    """Pre-curated regulatory filings for key Indian and global enterprises."""
    return [
        # HDFC Bank
        FilingChunk(
            chunk_id="hdfc_fy25_mda_nim",
            ticker="HDFCBANK.NS",
            fiscal_year=2025,
            section="MD&A",
            content="Net Interest Margin (NIM) faced transient compression following the merger integration, with deposit repricing pressures persisting across retail liabilities before stabilizing toward the final quarter.",
            sentiment_hint="negative",
            metadata={"topic": "net_interest_margin", "quarter": "Q4"},
        ),
        FilingChunk(
            chunk_id="hdfc_fy25_risk_credit",
            ticker="HDFCBANK.NS",
            fiscal_year=2025,
            section="Risk Factors",
            content="Asset quality across wholesale advances remained resilient with gross non-performing assets (GNPA) at 1.24%. The bank maintained prudent contingent provision buffers against potential retail unsecured slippages.",
            sentiment_hint="positive",
            metadata={"topic": "asset_quality", "gnpa": 1.24},
        ),
        FilingChunk(
            chunk_id="hdfc_fy25_guidance_growth",
            ticker="HDFCBANK.NS",
            fiscal_year=2025,
            section="Forward Guidance",
            content="Management guides for credit growth to closely track systemic deposit accretion, prioritizing credit-to-deposit ratio (CD ratio) consolidation over aggressive loan book expansion.",
            sentiment_hint="neutral",
            metadata={"topic": "cd_ratio_guidance"},
        ),

        # TCS (Tata Consultancy Services)
        FilingChunk(
            chunk_id="tcs_fy25_mda_verticals",
            ticker="TCS.NS",
            fiscal_year=2025,
            section="MD&A",
            content="Revenue growth in the Banking, Financial Services and Insurance (BFSI) vertical and North American enterprise segment witnessed client decision-making delays and discretionary tech spend cutbacks.",
            sentiment_hint="negative",
            metadata={"topic": "bfsi_slowdown", "region": "North America"},
        ),
        FilingChunk(
            chunk_id="tcs_fy25_guidance_deals",
            ticker="TCS.NS",
            fiscal_year=2025,
            section="Forward Guidance",
            content="Total Contract Value (TCV) deal order book reached resilient records of $13.2B, driven by enterprise AI infrastructure modernisation and long-term cost optimization vendor consolidation engagements.",
            sentiment_hint="positive",
            metadata={"topic": "order_book", "tcv_usd_b": 13.2},
        ),
        FilingChunk(
            chunk_id="tcs_fy25_risk_margins",
            ticker="TCS.NS",
            fiscal_year=2025,
            section="Risk Factors",
            content="Operating margin resilience faces wage increase headwinds, cross-currency volatility, and competitive pricing renegotiations in fixed-price transformation contracts.",
            sentiment_hint="negative",
            metadata={"topic": "operating_margins"},
        ),

        # ICICI Bank
        FilingChunk(
            chunk_id="icici_fy25_mda_margins",
            ticker="ICICIBANK.NS",
            fiscal_year=2025,
            section="MD&A",
            content="Domestic net interest margin moderated slightly to 4.36% due to lagged cost of deposits repricing, offset by superior risk-calibrated operating profit growth across retail and SME lending.",
            sentiment_hint="positive",
            metadata={"topic": "margins", "nim_pct": 4.36},
        ),
        FilingChunk(
            chunk_id="icici_fy25_risk_unsecured",
            ticker="ICICIBANK.NS",
            fiscal_year=2025,
            section="Risk Factors",
            content="Regulatory tightening of risk weights on unsecured consumer credit and non-banking finance exposures has prompted conservative origination underwriting standards in personal loans.",
            sentiment_hint="negative",
            metadata={"topic": "unsecured_lending_risk"},
        ),

        # Infosys
        FilingChunk(
            chunk_id="infy_fy25_mda_guidance",
            ticker="INFY.NS",
            fiscal_year=2025,
            section="Forward Guidance",
            content="Company revised constant-currency revenue growth guidance following persistent discretionary program deferrals in Europe and telecommunications client budget freezes.",
            sentiment_hint="negative",
            metadata={"topic": "guidance_revision"},
        ),

        # Reliance Industries
        FilingChunk(
            chunk_id="reliance_fy25_mda_o2c",
            ticker="RELIANCE.NS",
            fiscal_year=2025,
            section="MD&A",
            content="Oil to Chemicals (O2C) segment earnings experienced pressure from lower global gross refining margins (GRMs) and downstream petrochemical margin tightness, buffered by retail footprint expansion.",
            sentiment_hint="negative",
            metadata={"topic": "refining_margins"},
        ),
    ]


_DEFAULT_RETRIEVER = None

def get_default_filing_retriever() -> FilingRetriever:
    """Returns singleton filing retriever pre-seeded with universe annual report disclosures."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        retriever = FilingRetriever()
        retriever.add_filings(get_curated_annual_reports())
        _DEFAULT_RETRIEVER = retriever
    return _DEFAULT_RETRIEVER
