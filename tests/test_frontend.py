"""
Frontend DOM and static contract verification for frontend/dashboard.html.
Ensures UI controls, bundled offline fallback examples, and DOM IDs remain intact.
"""
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
import pytest

from src.api.schemas import ExplainResponse

DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "frontend" / "dashboard.html"


@pytest.fixture(scope="module")
def dashboard_soup():
    assert DASHBOARD_PATH.exists(), f"{DASHBOARD_PATH} does not exist"
    html_content = DASHBOARD_PATH.read_text(encoding="utf-8")
    return BeautifulSoup(html_content, "html.parser"), html_content


def test_dashboard_critical_dom_elements(dashboard_soup):
    soup, _ = dashboard_soup

    # Landing view elements
    landing = soup.find(id="landingView")
    assert landing is not None, "Missing #landingView"
    assert soup.find(id="tickerInput") is not None, "Missing #tickerInput"
    assert soup.find(id="explainBtn") is not None, "Missing #explainBtn"
    assert soup.find(id="startDate") is not None, "Missing #startDate"
    assert soup.find(id="endDate") is not None, "Missing #endDate"
    assert soup.find(id="apiStatus") is not None, "Missing #apiStatus"

    # Report view elements
    report_view = soup.find(id="reportView")
    assert report_view is not None, "Missing #reportView"
    assert soup.find(id="report") is not None, "Missing #report container"
    assert soup.find(id="tickerInput2") is not None, "Missing #tickerInput2"
    assert soup.find(id="explainBtn2") is not None, "Missing #explainBtn2"
    assert soup.find(id="newSearchBtn") is not None, "Missing #newSearchBtn"
    assert soup.find(id="reportStartDate") is not None, "Missing #reportStartDate"
    assert soup.find(id="reportEndDate") is not None, "Missing #reportEndDate"


def test_dashboard_bundled_examples_match_schema(dashboard_soup):
    _, html_content = dashboard_soup

    # Extract EXAMPLES = [...] from JS
    match = re.search(r"const EXAMPLES = (\[.*?\]);\s*\n\s*const SOURCE_COLORS", html_content, re.DOTALL)
    assert match is not None, "Could not locate EXAMPLES array in dashboard.html"

    raw_json = match.group(1)
    examples = json.loads(raw_json)
    assert len(examples) >= 3

    for ex in examples:
        # Every bundled example must conform to ExplainResponse schema
        validated = ExplainResponse(**ex)
        assert validated.ticker
        assert validated.start_date
        assert validated.end_date
        assert isinstance(validated.has_sector_factor, bool)
        assert 0.0 <= validated.confidence <= 1.0


def test_dashboard_script_functions(dashboard_soup):
    _, html_content = dashboard_soup

    # Ensure key JS handlers exist
    for fn in ["checkApi", "render", "renderDecompositionBar", "renderTimeline", "runExplain"]:
        assert f"function {fn}" in html_content or f"async function {fn}" in html_content
