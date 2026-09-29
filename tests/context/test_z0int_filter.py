"""Adapter tests for CONTEXT_FILTER=z0int. z0int is optional; the lane must be
lazy-imported and must degrade to keyword rather than break a research run."""
import asyncio
import builtins
import sys

import pytest

from gpt_researcher.context.select import CONTEXT_FILTERS, resolve_context_filter
from gpt_researcher.context.z0int_filter import Z0intContextCompressor

PAGES = [
    {"url": "https://a.test/1", "title": "Paris", "source": "https://a.test/1",
     "raw_content": "Paris is the capital of France. " * 60},
    {"url": "https://b.test/2", "title": "Bananas", "source": "https://b.test/2",
     "raw_content": "Bananas are an edible fruit. " * 60},
]
QUERY = "capital of France"


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if sys.version_info < (3, 10) \
        else asyncio.run(coro)


def test_z0int_is_a_known_mode_and_does_not_hijack_auto():
    assert "z0int" in CONTEXT_FILTERS
    # existing modes unchanged
    assert set(("auto", "jev", "keyword", "embeddings", "none")) <= set(CONTEXT_FILTERS)
    # auto must not silently become z0int
    assert resolve_context_filter("auto") in ("jev", "keyword")
    assert resolve_context_filter("z0int") == "z0int"


def test_z0int_keyword_path_is_local_and_preserves_citations():
    out = run(Z0intContextCompressor(documents=PAGES, mode="keyword", remote=False)
              .async_get_context(QUERY, 10))
    assert out.strip(), "must return context"
    assert "Source:" in out and "Title:" in out and "Content:" in out
    assert "a.test" in out, "source URL must survive filtering"


def test_z0int_none_path_returns_everything():
    out = run(Z0intContextCompressor(documents=PAGES, mode="none", remote=False)
              .async_get_context(QUERY, 10))
    assert "a.test" in out and "b.test" in out


def test_z0int_unavailable_falls_back_to_keyword(monkeypatch):
    """Regression: z0int missing must warn and degrade, not raise."""
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name.startswith("z0int"):
            raise ImportError("z0int not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    out = run(Z0intContextCompressor(documents=PAGES, mode="keyword", remote=False)
              .async_get_context(QUERY, 10))
    assert out.strip(), "fallback must still produce context"
    assert "Source:" in out


def test_remote_authority_defaults_off():
    """Private/local sources must not be sent to a remote scorer by default."""
    c = Z0intContextCompressor(documents=PAGES, mode="jev")
    assert c.remote is False


def test_z0int_never_chunks_itself():
    """GPT Researcher owns chunk boundaries; z0int must receive pre_chunked units."""
    import inspect
    src = inspect.getsource(Z0intContextCompressor.async_get_context)
    assert "pre_chunked=True" in src
    # behavioural, not textual: the same pages must yield the same chunks
    from gpt_researcher.context.lexical import LexicalContextCompressor as L
    mine = [c.page_content for c in Z0intContextCompressor(documents=PAGES)._chunks()]
    theirs = [c.page_content for c in L(documents=PAGES)._chunks()]
    assert mine == theirs, "z0int lane must chunk exactly as the keyword lane does"
