"""Tests for where-clause filtering — Prometheus-style matchers on trace metadata."""

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from kalibra.cli import main
from kalibra.commands.compare import _resolve_where_field
from kalibra.config import Matcher, PopulationConfig, CompareConfig, parse_matcher
from kalibra.model import Span, Trace


# ── Matcher parsing ──────────────────────────────────────────────────────────

class TestParseMatchers:
    def test_equality(self):
        m = parse_matcher("variant = baseline")
        assert m.field == "variant"
        assert m.op == "="
        assert m.value == "baseline"

    def test_not_equal(self):
        m = parse_matcher("env != staging")
        assert m.field == "env"
        assert m.op == "!="
        assert m.value == "staging"

    def test_regex_match(self):
        m = parse_matcher("model =~ gpt-4.*")
        assert m.field == "model"
        assert m.op == "=~"
        assert m.value == "gpt-4.*"

    def test_regex_not_match(self):
        m = parse_matcher("team !~ test|staging")
        assert m.field == "team"
        assert m.op == "!~"
        assert m.value == "test|staging"

    def test_double_equals(self):
        m = parse_matcher("variant == baseline")
        assert m.field == "variant"
        assert m.op == "=="
        assert m.value == "baseline"

    def test_double_equals_matches(self):
        assert Matcher("f", "==", "a").matches("a")
        assert not Matcher("f", "==", "a").matches("b")

    def test_no_spaces(self):
        m = parse_matcher("variant=baseline")
        assert m.field == "variant"
        assert m.op == "="
        assert m.value == "baseline"

    def test_extra_spaces(self):
        m = parse_matcher("  variant  =  baseline  ")
        assert m.field == "variant"
        assert m.op == "="
        assert m.value == "baseline"

    def test_value_with_spaces(self):
        m = parse_matcher("name = hello world")
        assert m.field == "name"
        assert m.value == "hello world"

    def test_empty_value(self):
        m = parse_matcher("tag =")
        assert m.field == "tag"
        assert m.op == "="
        assert m.value == ""

    def test_no_operator_raises(self):
        with pytest.raises(ValueError, match="No operator found"):
            parse_matcher("just a string")

    def test_empty_field_raises(self):
        with pytest.raises(ValueError, match="Empty field name"):
            parse_matcher("= value")

    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            parse_matcher("field =~ [invalid")

    def test_invalid_regex_not_match_raises(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            parse_matcher("field !~ (unclosed")


# ── Matcher evaluation ───────────────────────────────────────────────────────

class TestMatcherEval:
    def test_eq_match(self):
        assert Matcher("f", "=", "a").matches("a")

    def test_eq_no_match(self):
        assert not Matcher("f", "=", "a").matches("b")

    def test_neq_match(self):
        assert Matcher("f", "!=", "a").matches("b")

    def test_neq_no_match(self):
        assert not Matcher("f", "!=", "a").matches("a")

    def test_regex_match(self):
        assert Matcher("f", "=~", "gpt-4.*").matches("gpt-4o")

    def test_regex_no_match(self):
        assert not Matcher("f", "=~", "^gpt-4$").matches("gpt-4o")

    def test_regex_not_match(self):
        assert Matcher("f", "!~", "staging").matches("production")

    def test_regex_not_no_match(self):
        """!~ is a full match — 'staging' doesn't fully match 'staging-us', so !~ passes."""
        assert Matcher("f", "!~", "staging").matches("staging-us")
        # To exclude 'staging-us', use a pattern that fully matches it.
        assert not Matcher("f", "!~", "staging.*").matches("staging-us")

    def test_none_never_matches(self):
        assert not Matcher("f", "=", "a").matches(None)
        assert not Matcher("f", "!=", "a").matches(None)
        assert not Matcher("f", "=~", "a").matches(None)
        assert not Matcher("f", "!~", "a").matches(None)

    def test_numeric_coercion(self):
        """Metadata values may be ints/floats — compared as strings."""
        assert Matcher("f", "=", "42").matches(42)
        assert Matcher("f", "=~", r"^4\d$").matches(42)


# ── Field resolution fallback ────────────────────────────────────────────────

class TestResolveWhereField:
    def test_metadata_found(self):
        t = Trace(metadata={"variant": "baseline"})
        assert _resolve_where_field(t, "variant") == "baseline"

    def test_root_span_attribute_fallback(self):
        """Fields skipped by OpenInference metadata extraction (e.g. llm.model_name)
        should still be reachable via root span attributes."""
        t = Trace(
            spans=[
                Span(span_id="root", parent_id=None, attributes={"llm.model_name": "haiku"}),
                Span(span_id="child", parent_id="root", attributes={"llm.model_name": "haiku"}),
            ],
        )
        assert _resolve_where_field(t, "llm.model_name") == "haiku"

    def test_metadata_wins_over_span_attributes(self):
        t = Trace(
            metadata={"variant": "from-metadata"},
            spans=[Span(span_id="root", parent_id=None, attributes={"variant": "from-span"})],
        )
        assert _resolve_where_field(t, "variant") == "from-metadata"

    def test_missing_field_returns_none(self):
        t = Trace(
            metadata={},
            spans=[Span(span_id="root", parent_id=None, attributes={})],
        )
        assert _resolve_where_field(t, "nonexistent") is None

    def test_no_spans_no_metadata(self):
        t = Trace()
        assert _resolve_where_field(t, "anything") is None


# ── PopulationConfig parsing ─────────────────────────────────────────────────

class TestPopulationWhere:
    def test_where_parsed(self):
        pop = PopulationConfig.from_dict({
            "path": "traces.jsonl",
            "where": ["variant = baseline", "env != staging"],
        })
        assert len(pop.where) == 2
        assert pop.where[0].field == "variant"
        assert pop.where[1].op == "!="

    def test_where_empty(self):
        pop = PopulationConfig.from_dict({"path": "traces.jsonl"})
        assert pop.where == []

    def test_where_in_named_source(self):
        data = {
            "sources": {
                "prod-baseline": {
                    "path": "traces.jsonl",
                    "where": ["variant = baseline"],
                },
            },
            "baseline": "prod-baseline",
            "current": {"path": "other.jsonl"},
        }
        cfg = CompareConfig.from_dict(data)
        assert len(cfg.baseline.where) == 1
        assert cfg.baseline.where[0].value == "baseline"
        assert cfg.current.where == []


# ── End-to-end filtering via CLI ─────────────────────────────────────────────

class TestWhereFilterIntegration:
    @pytest.fixture()
    def tagged_traces(self, tmp_path):
        """Single file with traces tagged as baseline or current."""
        path = tmp_path / "all.jsonl"
        traces = []
        for variant in ("baseline", "current"):
            for i in range(3):
                traces.append({
                    "trace_id": f"{variant}-{i}",
                    "variant": variant,
                    "outcome": "success",
                    "spans": [{
                        "span_id": f"s-{variant}-{i}",
                        "name": "step",
                        "cost": 0.01 if variant == "baseline" else 0.02,
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "start_ns": 0,
                        "end_ns": 1_000_000_000,
                    }],
                })
        path.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
        return str(path)

    def test_where_splits_single_file(self, tagged_traces, tmp_path):
        config = {
            "sources": {
                "baseline": {
                    "path": tagged_traces,
                    "where": ["variant = baseline"],
                },
                "current": {
                    "path": tagged_traces,
                    "where": ["variant = current"],
                },
            },
            "baseline": "baseline",
            "current": "current",
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code in (0, 1), result.output
        assert "Kalibra Compare" in result.output
        # Should show 3 traces per side, not 6.
        assert "3 traces" in result.output

    def test_where_with_regex(self, tagged_traces, tmp_path):
        config = {
            "sources": {
                "baseline": {
                    "path": tagged_traces,
                    "where": ["variant =~ base.*"],
                },
                "current": {
                    "path": tagged_traces,
                    "where": ["variant =~ curr.*"],
                },
            },
            "baseline": "baseline",
            "current": "current",
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        assert result.exit_code in (0, 1), result.output
        assert "3 traces" in result.output

    def test_where_via_cli_named_source(self, tagged_traces, tmp_path):
        """CLI --baseline/--current referencing named sources should use their where."""
        config = {
            "sources": {
                "base": {
                    "path": tagged_traces,
                    "where": ["variant = baseline"],
                },
                "curr": {
                    "path": tagged_traces,
                    "where": ["variant = current"],
                },
            },
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, [
                "compare", "--baseline", "base", "--current", "curr",
            ])
        assert result.exit_code in (0, 1), result.output
        assert "Kalibra Compare" in result.output
        assert "3 traces" in result.output

    def test_where_no_matches_shows_no_data(self, tagged_traces, tmp_path):
        config = {
            "baseline": {
                "path": tagged_traces,
                "where": ["variant = nonexistent"],
            },
            "current": {
                "path": tagged_traces,
                "where": ["variant = current"],
            },
        }
        config_file = tmp_path / "kalibra.yml"
        config_file.write_text(yaml.dump(config))

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["compare"])
        # Should handle 0-trace population gracefully.
        assert result.exit_code in (0, 1), result.output
