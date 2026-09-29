"""z0int context filtering: GPT Researcher owns chunking, z0int owns ranking.

z0int's splitter is measurably NOT equivalent to LangChain's
``RecursiveCharacterTextSplitter`` (0/9 exact match on a pinned golden fixture
set), so this adapter does not let z0int chunk. It chunks exactly as the existing
keyword lane does -- 1000/100 -- and hands those chunks to z0int with
``pre_chunked=True``, so both implementations rank the SAME units and their
ranking can be compared like for like.

z0int is an OPTIONAL dependency: it is imported lazily inside the z0int lane only,
and any failure degrades to the existing keyword ranking with a clear warning
rather than breaking a research run.
"""

import logging
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..prompts import PromptFamily
from .lexical import LexicalContextCompressor
from .retriever import SearchAPIRetriever

logger = logging.getLogger(__name__)

#: Same bound the keyword lane uses, so the two are comparable.
Z0INT_KEYWORD_MAX_RESULTS = int(os.environ.get("KEYWORD_MAX_RESULTS", "25"))


class Z0intUnavailable(RuntimeError):
    """z0int could not be imported or the filter raised."""


class Z0intContextCompressor:
    """Selects context with z0int's ``context.filter`` primitive."""

    def __init__(
        self,
        documents,
        *,
        mode: str = "keyword",
        remote: bool = False,
        chunk_size: int = 1000,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        self.documents = documents
        self.mode = mode
        #: Remote (Jev) scoring is only ever authorized when the caller says the
        #: source is knowable public web content. Defaults to False so private or
        #: local sources are never sent to a remote scorer by default.
        self.remote = bool(remote)
        self.chunk_size = chunk_size
        self.prompt_family = prompt_family

    def _chunks(self) -> list[Document]:
        """Identical chunking to the keyword lane: GPT Researcher owns boundaries."""
        pages = SearchAPIRetriever(pages=self.documents).invoke("")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_size // 10
        )
        return [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    async def async_get_context(self, query: str, max_results: int = 10,
                                cost_callback=None) -> str:
        try:
            from z0int.context_filter import filter_context
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CONTEXT_FILTER=z0int but z0int is unavailable (%s); "
                "falling back to keyword ranking", exc)
            return await self._keyword(query, max_results, cost_callback)

        chunks = self._chunks()
        passages = [
            {
                "raw_content": c.page_content,
                "url": (c.metadata or {}).get("source") or (c.metadata or {}).get("url") or "",
                "title": (c.metadata or {}).get("title") or "",
            }
            for c in chunks
        ]
        try:
            result = filter_context(
                query,
                passages,
                mode=self.mode,
                max_results=max_results,
                keyword_max_results=Z0INT_KEYWORD_MAX_RESULTS,
                pre_chunked=True,
                allow_remote=self.remote and self.mode in ("jev", "auto"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("z0int context filter failed (%s); falling back to keyword", exc)
            return await self._keyword(query, max_results, cost_callback)

        if result.receipt.fallback and result.receipt.fallback_reason:
            logger.info("z0int context filter degraded: %s", result.receipt.fallback_reason)

        # Rebuild Documents so the writer sees GPT Researcher's own citation
        # contract (Source / Title / Content) rather than z0int's context() string.
        docs = [
            Document(
                page_content=s.text,
                metadata={"source": s.url, "title": s.title},
            )
            for s in result.selected
        ]
        return self.prompt_family.pretty_print_docs(docs, max_results)

    async def _keyword(self, query: str, max_results: int, cost_callback=None) -> str:
        return await LexicalContextCompressor(
            documents=self.documents,
            relative_threshold=float(os.environ.get("KEYWORD_RELATIVE_THRESHOLD", "0.5")),
            prompt_family=self.prompt_family,
        ).async_get_context(query, Z0INT_KEYWORD_MAX_RESULTS, cost_callback)
