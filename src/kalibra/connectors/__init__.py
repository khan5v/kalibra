"""External trace store connectors — zero-migration ingestion."""

from __future__ import annotations


def get_connector(source: str):
    """Return a connector instance based on env vars."""
    import os

    if source == "langfuse":
        from kalibra.connectors.langfuse import LangfuseConnector
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
        if not pk or not sk:
            raise RuntimeError(
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )
        return LangfuseConnector(host=host, public_key=pk, secret_key=sk)

    if source == "langsmith":
        from kalibra.connectors.langsmith import LangSmithConnector
        api_key = os.environ.get("LANGSMITH_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set LANGSMITH_API_KEY environment variable.")
        api_url = os.environ.get("LANGSMITH_API_URL") or None
        return LangSmithConnector(api_key=api_key, api_url=api_url)

    raise ValueError(f"Unknown source: {source!r}. Supported: langfuse, langsmith")
