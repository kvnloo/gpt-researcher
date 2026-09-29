"""Phase A: does z0int rank the SAME chunks the same way as GPT Researcher?

Splitter parity is known false (0/9) and is NOT what this measures. Both sides
receive identical chunks, so only ranking/filtering/formatting are compared.
"""
import hashlib, json, os, sys

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, "/home/kvn/tmp/gptr")
# z0int is an optional dependency; point PYTHONPATH at the z0intelligence checkout
sys.path.insert(0, os.environ.get("Z0INT_SRC", "/home/kvn/tmp/z0-live/src"))
from gpt_researcher.context.lexical import LexicalContextCompressor, bm25_scores
from gpt_researcher.prompts import PromptFamily
from gpt_researcher.context.retriever import SearchAPIRetriever
from z0int.context_filter import filter_context

ENC = tiktoken.get_encoding("cl100k_base")
H = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]

PAGES = [
    {"url": "https://a.test/1", "title": "Paris overview",
     "raw_content": ("Paris is the capital and most populous city of France. " * 30) + ("filler text " * 120)},
    {"url": "https://b.test/2", "title": "Banana facts",
     "raw_content": ("Bananas are an edible fruit produced by herbaceous plants. " * 30) + ("more filler " * 120)},
    {"url": "https://c.test/3", "title": "Seine river",
     "raw_content": ("The Seine flows through Paris and is a major commercial river. " * 25) + ("padding words " * 120)},
]
QUERY = "What is the capital of France?"


def chunks_for(pages, size=1000):
    p = SearchAPIRetriever(pages=pages).invoke("")
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 10)
    return [c for c in sp.split_documents(p) if c.page_content.strip()]


def main():
    pf = PromptFamily
    chunks = chunks_for(PAGES)
    print(f"identical chunk set: {len(chunks)} chunks")

    # ---------------- upstream keyword ----------------
    up = LexicalContextCompressor(documents=PAGES, relative_threshold=0.5, prompt_family=pf)
    up_docs = up.select(QUERY, 25)
    up_ctx = pf.pretty_print_docs(up_docs, 25)

    # ---------------- z0int keyword (pre_chunked: same units) -------------
    passages = [{"raw_content": c.page_content,
                 "url": (c.metadata or {}).get("source") or "",
                 "title": (c.metadata or {}).get("title") or ""} for c in chunks]
    zr = filter_context(QUERY, passages, mode="keyword", pre_chunked=True,
                        max_results=25, keyword_max_results=25, relative_threshold=0.5)
    z_docs = [Document(page_content=s.text, metadata={"source": s.url, "title": s.title})
              for s in zr.selected]
    z_ctx = pf.pretty_print_docs(z_docs, 25)

    # ---------------- upstream none ----------------
    up_none_docs = SearchAPIRetriever(pages=PAGES).invoke("")
    up_none = pf.pretty_print_docs(up_none_docs)
    zn = filter_context(QUERY, PAGES, mode="none", pre_chunked=True)
    z_none_docs = [Document(page_content=s.text, metadata={"source": s.url, "title": s.title})
                   for s in zn.selected]
    z_none = pf.pretty_print_docs(z_none_docs)

    def row(label, a_docs, a_ctx, b_docs, b_ctx):
        ah = [H(d.page_content) for d in a_docs]
        bh = [H(d.page_content) for d in b_docs]
        return {
            "comparison": label,
            "upstream_chunks": len(a_docs), "z0int_chunks": len(b_docs),
            "hashes_identical": ah == bh,
            "order_identical": ah == bh,
            "source_identical": [d.metadata.get("source") for d in a_docs] ==
                                [d.metadata.get("source") for d in b_docs],
            "title_identical": [d.metadata.get("title") for d in a_docs] ==
                               [d.metadata.get("title") for d in b_docs],
            "context_identical": a_ctx == b_ctx,
            "chars_upstream": len(a_ctx), "chars_z0int": len(b_ctx),
            "tokens_upstream": len(ENC.encode(a_ctx)),
            "tokens_z0int": len(ENC.encode(b_ctx)),
        }

    rows = [
        row("upstream keyword vs z0int keyword", up_docs, up_ctx, z_docs, z_ctx),
        row("upstream none vs z0int none", up_none_docs, up_none, z_none_docs, z_none),
    ]
    print("\n=== PHASE A ===")
    for r in rows:
        ok = all(r[k] for k in ("hashes_identical", "source_identical",
                                "title_identical", "context_identical"))
        print(f"  {r['comparison']}")
        print(f"    chunks {r['upstream_chunks']}/{r['z0int_chunks']} | hashes={r['hashes_identical']} "
              f"source={r['source_identical']} title={r['title_identical']} "
              f"context={r['context_identical']} | chars {r['chars_upstream']}/{r['chars_z0int']} "
              f"| tokens {r['tokens_upstream']}/{r['tokens_z0int']} | {'PASS' if ok else 'FAIL'}")
    out = {"query": QUERY, "rows": rows,
           "determinism": None}
    # determinism: rerun both keyword sides
    z2 = filter_context(QUERY, passages, mode="keyword", pre_chunked=True, max_results=25,
                        keyword_max_results=25, relative_threshold=0.5)
    out["determinism"] = [H(s.text) for s in z2.selected] == [H(s.text) for s in zr.selected]
    print(f"  z0int determinism: {out['determinism']}")
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase-a-results.json"), "w").write(json.dumps(out, indent=2))
    print("  wrote phase-a-results.json")


main()
