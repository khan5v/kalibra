"""Tests for KalibraDataWarning — data-loss signals during loading.

Silent data loss corrupts downstream metrics. Each loader path that
drops, coerces, or falls back must emit a KalibraDataWarning the user
can see.
"""

from __future__ import annotations

import warnings

import pytest

from kalibra import KalibraDataWarning
from kalibra.loaders._utils import _group_by_trace_id


class TestGroupByTraceIdWarnings:
    """_group_by_trace_id must warn when spans are dropped."""

    def test_no_warning_when_all_spans_have_trace_id(self):
        spans = [
            {"context": {"trace_id": "t1"}, "name": "a"},
            {"context": {"trace_id": "t1"}, "name": "b"},
            {"context": {"trace_id": "t2"}, "name": "c"},
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            groups = _group_by_trace_id(spans)
        assert len(groups) == 2
        assert not any(
            issubclass(w.category, KalibraDataWarning) for w in caught
        )

    def test_warns_on_missing_trace_id(self):
        spans = [
            {"context": {"trace_id": "t1"}, "name": "kept"},
            {"context": {"trace_id": ""}, "name": "dropped-empty"},
            {"context": {}, "name": "dropped-no-trace-id"},
        ]
        with pytest.warns(KalibraDataWarning, match="Dropped 2 span"):
            groups = _group_by_trace_id(spans)
        assert list(groups.keys()) == ["t1"]

    def test_warns_on_missing_context(self):
        spans = [
            {"context": {"trace_id": "t1"}, "name": "kept"},
            {"name": "no-context"},
            {"context": "not-a-dict", "name": "bad-context"},
        ]
        with pytest.warns(KalibraDataWarning, match="missing context object"):
            _group_by_trace_id(spans)

    def test_warns_on_non_dict_items(self):
        spans = [
            {"context": {"trace_id": "t1"}},
            "string-instead-of-span",
            42,
            None,
        ]
        with pytest.warns(KalibraDataWarning, match="3 not a dict"):
            _group_by_trace_id(spans)

    def test_warning_message_aggregates_all_categories(self):
        spans = [
            {"context": {"trace_id": "t1"}},
            "not-a-dict",
            {"name": "no-context"},
            {"context": {"trace_id": ""}},
        ]
        with pytest.warns(KalibraDataWarning) as caught:
            _group_by_trace_id(spans)
        msg = str(caught[0].message)
        assert "Dropped 3 span" in msg
        assert "missing context.trace_id" in msg
        assert "missing context object" in msg
        assert "not a dict" in msg
