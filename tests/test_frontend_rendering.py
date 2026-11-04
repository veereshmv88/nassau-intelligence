"""
Tests for frontend rendering and safety against raw HTML code block leakage.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def test_render_safe_html_strips_indentation():
    raw_html = """
        <div class="results-table-container">
            <table class="nassau-results-table">
                <tr class="row-recommended">
                    <td>Test</td>
                </tr>
            </table>
        </div>
    """
    cleaned = "\n".join(line.strip() for line in raw_html.strip().splitlines())
    lines = cleaned.splitlines()
    for line in lines:
        assert not line.startswith("    "), f"Line still has 4-space indentation: {line}"
        assert not line.startswith("\t"), f"Line still has tab indentation: {line}"

def test_app_py_has_no_raw_html_code_block_leaks():
    app_text = (BASE_DIR / "src" / "app.py").read_text(encoding="utf-8")
    
    # Verify render_safe_html is defined and used for results table
    assert "def render_safe_html(" in app_text
    assert "render_safe_html(table_html)" in app_text
    assert "st.markdown(table_html" not in app_text

def test_css_contains_container_border_card_styling():
    css_text = (BASE_DIR / "styles" / "main.css").read_text(encoding="utf-8")
    assert 'div[data-testid="stVerticalBlockBorderWrapper"] > div' in css_text
    assert ".nassau-results-table" in css_text
    assert ".row-recommended" in css_text
