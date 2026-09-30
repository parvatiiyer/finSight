"""
Graph-based monitoring loop for FinSight (using NetworkX).

Models relationships:
  (:Company)-[:HAD_MOVE]->(:MoveWindow)
  (:MoveWindow)-[:EXPLAINED_BY]->(:Evidence)
  (:MoveWindow)-[:CONSIDERED_AND_REJECTED]->(:Evidence)
  (:Company)-[:PEER_OF]->(:Company)
  (:Evidence)-[:CONTRADICTS]->(:Evidence)
  (:Evidence)-[:CORROBORATES]->(:Evidence)

Enables:
- Tracking explanation drift over time
- Discovering contradicted explanations ("earlier cited concern X, later invalidated by Y")
- Source reliability feedback loop
- Peer network traversal
"""

from datetime import date, datetime
from typing import Any
import networkx as nx

from config.universe import UNIVERSE, peers_of, sector_of
from src.agents.evidence_types import AdjudicationResult, Direction, Evidence


class MonitoringGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._init_companies()

    def _init_companies(self):
        """Populate known companies and their PEER_OF relationships."""
        for sector, stocks in UNIVERSE.items():
            for ticker, name in stocks:
                self.g.add_node(f"Company:{ticker}", type="Company", ticker=ticker, name=name, sector=sector)

        for sector, stocks in UNIVERSE.items():
            tickers = [t for t, _ in stocks]
            for t1 in tickers:
                for t2 in tickers:
                    if t1 != t2:
                        self.g.add_edge(f"Company:{t1}", f"Company:{t2}", key="PEER_OF", type="PEER_OF")

    def ingest_run(
        self,
        result: AdjudicationResult,
        run_id: str | int | None = None,
        narrative: str | None = None,
    ) -> str:
        """
        Ingests an AdjudicationResult into the graph:
        Creates MoveWindow, Evidence nodes, links them to Company,
        and computes CORROBORATES and CONTRADICTS edges.
        """
        m = result.move
        company_node = f"Company:{m.ticker}"
        if company_node not in self.g:
            self.g.add_node(company_node, type="Company", ticker=m.ticker, name=m.ticker, sector=sector_of(m.ticker) or "Unknown")

        move_node_id = f"MoveWindow:{m.ticker}:{m.start_date}:{m.end_date}:{run_id or id(result)}"
        self.g.add_node(
            move_node_id,
            type="MoveWindow",
            ticker=m.ticker,
            start_date=str(m.start_date),
            end_date=str(m.end_date),
            raw_return=m.raw_return,
            residual_zscore=m.residual_zscore,
            confidence=result.confidence,
            unexplained=result.unexplained,
            narrative=narrative or "",
        )
        self.g.add_edge(company_node, move_node_id, key="HAD_MOVE", type="HAD_MOVE")

        new_evidence_nodes = []
        for idx, v in enumerate(result.verdicts):
            ev = v.evidence
            ev_node_id = f"Evidence:{m.ticker}:{ev.source_agent}:{ev.event_date}:{idx}:{run_id or id(v)}"
            self.g.add_node(
                ev_node_id,
                type="Evidence",
                source_agent=ev.source_agent,
                event_date=str(ev.event_date),
                direction=ev.direction.value,
                magnitude_hint=ev.magnitude_hint,
                raw_text=ev.raw_text,
                contribution=v.contribution,
                accepted=v.accepted,
                rejection_reason=v.rejection_reason or "",
            )
            new_evidence_nodes.append((ev_node_id, v))

            if v.accepted:
                self.g.add_edge(move_node_id, ev_node_id, key="EXPLAINED_BY", type="EXPLAINED_BY", contribution=v.contribution)
            else:
                self.g.add_edge(move_node_id, ev_node_id, key="CONSIDERED_AND_REJECTED", type="CONSIDERED_AND_REJECTED", reason=v.rejection_reason)

        # 1. Intra-run Corroboration: accepted evidence from different agents in the same direction
        accepted_new = [(node_id, v) for node_id, v in new_evidence_nodes if v.accepted]
        for i in range(len(accepted_new)):
            for j in range(i + 1, len(accepted_new)):
                n1, v1 = accepted_new[i]
                n2, v2 = accepted_new[j]
                if v1.evidence.source_agent != v2.evidence.source_agent:
                    if v1.evidence.direction == v2.evidence.direction or v1.evidence.direction == Direction.NEUTRAL or v2.evidence.direction == Direction.NEUTRAL:
                        self.g.add_edge(n1, n2, key="CORROBORATES", type="CORROBORATES")
                        self.g.add_edge(n2, n1, key="CORROBORATES", type="CORROBORATES")

        # 2. Cross-run Contradiction Detection:
        # Check against existing accepted evidence for this company
        self._detect_contradictions(m.ticker, accepted_new)

        return move_node_id

    def _detect_contradictions(self, ticker: str, new_accepted_nodes: list[tuple[str, Any]]):
        """
        Detects if new evidence contradicts earlier accepted evidence for this stock.
        Contradiction occurs when:
        - Direction is opposing (POSITIVE vs NEGATIVE)
        - Event date of newer evidence is after or overlapping earlier evidence
        - Text or tag references similar metrics (e.g., margins, earnings, regulation)
        """
        company_node = f"Company:{ticker}"
        # Find all previous MoveWindows for this company
        for _, move_node, move_data in self.g.out_edges(company_node, data=True):
            if move_data.get("type") != "HAD_MOVE":
                continue
            for _, prior_ev_node, ev_edge in self.g.out_edges(move_node, data=True):
                if ev_edge.get("type") != "EXPLAINED_BY":
                    continue
                prior_ev = self.g.nodes[prior_ev_node]
                prior_dir = prior_ev.get("direction")
                prior_text = prior_ev.get("raw_text", "").lower()

                for new_node, new_verdict in new_accepted_nodes:
                    new_ev = new_verdict.evidence
                    new_dir = new_ev.direction.value
                    new_text = new_ev.raw_text.lower()

                    # Check for opposing directions
                    if (prior_dir == "positive" and new_dir == "negative") or (prior_dir == "negative" and new_dir == "positive"):
                        # Check topical overlap or subsequent temporal invalidation
                        topic_words = {"earnings", "revenue", "profit", "margin", "growth", "probe", "investigation", "npa", "debt", "rating", "regulat"}
                        has_topic_overlap = any(w in prior_text and w in new_text for w in topic_words)
                        if has_topic_overlap or str(new_ev.event_date) >= prior_ev.get("event_date", ""):
                            # Add directed CONTRADICTS edge from new_node to prior_ev_node
                            self.g.add_edge(
                                new_node,
                                prior_ev_node,
                                key="CONTRADICTS",
                                type="CONTRADICTS",
                                reason=f"New fact '{new_ev.raw_text}' opposes earlier explanation '{prior_ev['raw_text']}'",
                            )

    def get_contradicted_moves(self, ticker: str | None = None) -> list[dict[str, Any]]:
        """
        Finds moves whose accepted evidence was subsequently contradicted by new evidence.
        Graph query: (MoveWindow)-[:EXPLAINED_BY]->(PriorEvidence)<-[:CONTRADICTS]-(NewEvidence)
        """
        results = []
        move_nodes = [n for n, d in self.g.nodes(data=True) if d.get("type") == "MoveWindow"]
        if ticker:
            move_nodes = [n for n in move_nodes if self.g.nodes[n].get("ticker") == ticker]

        for m_node in move_nodes:
            m_data = self.g.nodes[m_node]
            # Follow EXPLAINED_BY
            for _, ev_node, edge_data in self.g.out_edges(m_node, data=True):
                if edge_data.get("type") != "EXPLAINED_BY":
                    continue
                # Check incoming CONTRADICTS edges to this evidence
                for contradictor_node, _, c_edge in self.g.in_edges(ev_node, data=True):
                    if c_edge.get("type") == "CONTRADICTS":
                        new_ev_data = self.g.nodes[contradictor_node]
                        prior_ev_data = self.g.nodes[ev_node]
                        results.append({
                            "move_ticker": m_data["ticker"],
                            "move_window": f"{m_data['start_date']} to {m_data['end_date']}",
                            "move_confidence": m_data["confidence"],
                            "invalidated_evidence": prior_ev_data["raw_text"],
                            "invalidated_date": prior_ev_data["event_date"],
                            "contradicting_evidence": new_ev_data["raw_text"],
                            "contradicting_date": new_ev_data["event_date"],
                            "reason": c_edge.get("reason"),
                        })
        return results

    def diff_explanations(self, prior_move_id: str, new_move_id: str) -> dict[str, Any]:
        """Compares two explanation runs and reports evolution of confidence and evidence."""
        if prior_move_id not in self.g or new_move_id not in self.g:
            raise KeyError("Both move IDs must exist in the monitoring graph")

        p_move = self.g.nodes[prior_move_id]
        n_move = self.g.nodes[new_move_id]

        p_ev = [self.g.nodes[v] for _, v, d in self.g.out_edges(prior_move_id, data=True) if d.get("type") == "EXPLAINED_BY"]
        n_ev = [self.g.nodes[v] for _, v, d in self.g.out_edges(new_move_id, data=True) if d.get("type") == "EXPLAINED_BY"]

        conf_diff = round(n_move["confidence"] - p_move["confidence"], 3)
        return {
            "prior_confidence": p_move["confidence"],
            "new_confidence": n_move["confidence"],
            "confidence_delta": conf_diff,
            "prior_accepted_count": len(p_ev),
            "new_accepted_count": len(n_ev),
            "prior_narrative": p_move.get("narrative"),
            "new_narrative": n_move.get("narrative"),
        }

    def get_source_reliability(self, source_agent: str) -> dict[str, Any]:
        """Calculates empirical corroboration and contradiction rates for an evidence source."""
        agent_nodes = [
            n for n, d in self.g.nodes(data=True)
            if d.get("type") == "Evidence" and d.get("source_agent") == source_agent
        ]
        total = len(agent_nodes)
        if total == 0:
            return {"source_agent": source_agent, "total_evidence": 0, "corroboration_rate": 0.0, "contradiction_rate": 0.0}

        corroborated = 0
        contradicted = 0
        for n in agent_nodes:
            if any(d.get("type") == "CORROBORATES" for _, _, d in self.g.out_edges(n, data=True)):
                corroborated += 1
            if any(d.get("type") == "CONTRADICTS" for _, _, d in self.g.in_edges(n, data=True)):
                contradicted += 1

        return {
            "source_agent": source_agent,
            "total_evidence": total,
            "corroboration_rate": round(corroborated / total, 3),
            "contradiction_rate": round(contradicted / total, 3),
        }

    def export_to_cypher(self) -> str:
        """
        Exports the in-memory NetworkX monitoring graph to standard Neo4j Cypher statements.
        Produces idempotent MERGE statements for all Company, MoveWindow, and Evidence nodes
        and their relationships (HAD_MOVE, EXPLAINED_BY, CONSIDERED_AND_REJECTED,
        PEER_OF, CORROBORATES, CONTRADICTS).
        """
        lines: list[str] = [
            "// FinSight Neo4j Graph Export",
            "// Constraints & Indices",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MoveWindow) REQUIRE m.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Evidence) REQUIRE e.id IS UNIQUE;",
            "",
        ]

        def _escape(val: Any) -> str:
            if val is None:
                return '""'
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, (int, float)):
                return str(val)
            s = str(val).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
            return f'"{s}"'

        # 1. Export Nodes
        lines.append("// 1. Nodes")
        for node_id, data in self.g.nodes(data=True):
            node_type = data.get("type", "Node")
            if node_type == "Company":
                ticker = _escape(data.get("ticker", ""))
                name = _escape(data.get("name", ""))
                sector = _escape(data.get("sector", ""))
                lines.append(
                    f'MERGE (c:Company {{ticker: {ticker}}}) '
                    f'ON CREATE SET c.name = {name}, c.sector = {sector}, c.id = {_escape(node_id)} '
                    f'ON MATCH SET c.name = {name}, c.sector = {sector};'
                )
            elif node_type == "MoveWindow":
                id_val = _escape(node_id)
                ticker = _escape(data.get("ticker", ""))
                start_date = _escape(data.get("start_date", ""))
                end_date = _escape(data.get("end_date", ""))
                raw_return = _escape(data.get("raw_return", 0.0))
                residual_zscore = _escape(data.get("residual_zscore", 0.0))
                confidence = _escape(data.get("confidence", 0.0))
                unexplained = _escape(data.get("unexplained", False))
                narrative = _escape(data.get("narrative", ""))
                lines.append(
                    f'MERGE (m:MoveWindow {{id: {id_val}}}) '
                    f'ON CREATE SET m.ticker = {ticker}, m.start_date = {start_date}, m.end_date = {end_date}, '
                    f'm.raw_return = {raw_return}, m.residual_zscore = {residual_zscore}, m.confidence = {confidence}, '
                    f'm.unexplained = {unexplained}, m.narrative = {narrative};'
                )
            elif node_type == "Evidence":
                id_val = _escape(node_id)
                src = _escape(data.get("source_agent", ""))
                ev_date = _escape(data.get("event_date", ""))
                direction = _escape(data.get("direction", ""))
                mag = _escape(data.get("magnitude_hint", 0.0))
                text = _escape(data.get("raw_text", ""))
                contrib = _escape(data.get("contribution", 0.0))
                accepted = _escape(data.get("accepted", False))
                reason = _escape(data.get("rejection_reason", ""))
                lines.append(
                    f'MERGE (e:Evidence {{id: {id_val}}}) '
                    f'ON CREATE SET e.source_agent = {src}, e.event_date = {ev_date}, e.direction = {direction}, '
                    f'e.magnitude_hint = {mag}, e.raw_text = {text}, e.contribution = {contrib}, '
                    f'e.accepted = {accepted}, e.rejection_reason = {reason};'
                )

        # 2. Export Edges
        lines.append("\n// 2. Relationships")
        for u, v, edge_data in self.g.edges(data=True):
            edge_type = edge_data.get("type", "RELATED_TO")
            u_node = self.g.nodes[u]
            v_node = self.g.nodes[v]
            u_label = u_node.get("type", "Node")
            v_label = v_node.get("type", "Node")

            u_key = "ticker" if u_label == "Company" else "id"
            v_key = "ticker" if v_label == "Company" else "id"
            u_val = _escape(u_node.get("ticker" if u_label == "Company" else "id", u))
            v_val = _escape(v_node.get("ticker" if v_label == "Company" else "id", v))

            # Edge props
            props = []
            if "contribution" in edge_data:
                props.append(f"contribution: {_escape(edge_data['contribution'])}")
            if "reason" in edge_data:
                props.append(f"reason: {_escape(edge_data['reason'])}")
            prop_str = f" {{{', '.join(props)}}}" if props else ""

            lines.append(
                f'MATCH (a:{u_label} {{{u_key}: {u_val}}}), (b:{v_label} {{{v_key}: {v_val}}}) '
                f'MERGE (a)-[r:{edge_type}{prop_str}]->(b);'
            )

        return "\n".join(lines)

    def save_cypher_export(self, filepath: str) -> None:
        """Writes the generated Cypher script to disk."""
        cypher = self.export_to_cypher()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cypher)

    def sync_to_neo4j(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "password"),
    ) -> dict[str, Any]:
        """
        Attempts direct sync into a running Neo4j instance.
        Gracefully returns error status if Neo4j is offline or unreachable.
        """
        try:
            from neo4j import GraphDatabase
            statements = self.export_to_cypher().split(";\n")
            clean_stmts = [s.strip() for s in statements if s.strip() and not s.strip().startswith("//")]
            with GraphDatabase.driver(uri, auth=auth) as driver:
                with driver.session() as session:
                    for stmt in clean_stmts:
                        session.run(stmt)
            return {"status": "success", "statements_executed": len(clean_stmts)}
        except Exception as exc:
            return {"status": "offline_or_error", "message": str(exc)}

