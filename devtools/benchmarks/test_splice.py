import pytest
import splice as mod


def test_marker_is_replaced_by_the_named_block():
    assert mod.splice("a\n<!-- ONE -->\nb\n", {"ONE": "table"}) == "a\ntable\nb\n"


def test_an_unfilled_marker_is_an_error_rather_than_an_empty_section():
    # A table that failed to generate must not silently vanish from the report.
    with pytest.raises(SystemExit, match="no content for marker"):
        mod.splice("<!-- ONE -->\n<!-- TWO -->\n", {"ONE": "table"})


def test_a_block_with_no_marker_is_an_error_rather_than_dropped():
    with pytest.raises(SystemExit, match="absent marker"):
        mod.splice("<!-- ONE -->\n", {"ONE": "x", "TWO": "y"})


def test_the_same_marker_twice_is_filled_both_times():
    assert mod.splice("<!-- X -->\n<!-- X -->\n", {"X": "q"}) == "q\nq\n"


def test_marker_syntax_is_exact_so_prose_mentioning_one_is_not_substituted():
    for text in ("<!--ONE-->\n", "<!-- one -->\n", "prefix <!-- ONE -->\n"):
        assert mod.splice(text, {}) == text
