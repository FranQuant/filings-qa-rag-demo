import argparse
from pathlib import Path

from .acquisition import run_fetch
from .parsing import run_parse
from .indexing import run_index
from .embeddings import run_embed
from .retrieval import retrieve
from .answering import synthesize_answer

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenAI filings workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch filings.")
    fetch_parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    fetch_parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")

    parse_parser = subparsers.add_parser("parse", help="Parse filings.")
    parse_parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    parse_parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
    index_parser = subparsers.add_parser("index", help="Index filings.")
    index_parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    index_parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
    embed_parser = subparsers.add_parser("embed", help="Embed filings.")
    embed_parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    embed_parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
    embed_parser.add_argument("--model", required=True, help="Embedding model name.")
    embed_parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    embed_parser.add_argument("--force", action="store_true", help="Re-embed all chunks.")
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve filings.")
    retrieve_parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    retrieve_parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
    retrieve_parser.add_argument("--query", required=True, help="Query string.")
    retrieve_parser.add_argument("--k", type=int, default=5, help="Top-K results.")
    retrieve_parser.add_argument("--doc-type", dest="doc_type", help="Optional doc type filter.")
    answer_parser = subparsers.add_parser("answer", help="Answer using retrieved filings.")
    answer_parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    answer_parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
    answer_parser.add_argument("--query", required=True, help="Query string.")
    answer_parser.add_argument("--k", type=int, default=5, help="Top-K results.")
    answer_parser.add_argument("--model", required=True, help="Model name.")
    answer_parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens.")
    answer_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    subparsers.add_parser("ask", help="Ask questions over filings (placeholder).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        output_root = Path("data/raw/filings")
        results = run_fetch(args.issuer, args.period, output_root)
        print("Downloaded filings:")
        for name, entry in results.items():
            print(f"- {name}: {entry['url']}")
        return

    if args.command == "parse":
        input_root = Path("data/raw/filings")
        output_root = Path("data/processed/filings")
        output_path = run_parse(args.issuer, args.period, input_root, output_root)
        print(f"Parsed sections saved to: {output_path}")
        return

    if args.command == "index":
        input_root = Path("data/processed/filings")
        output_root = Path("data/processed/filings")
        output_path = run_index(args.issuer, args.period, input_root, output_root)
        print(f"Chunked sections saved to: {output_path}")
        return

    if args.command == "embed":
        input_root = Path("data/processed/filings")
        output_root = Path("data/processed/filings")
        embeddings_path, manifest_path = run_embed(
            args.issuer,
            args.period,
            input_root,
            output_root,
            model=args.model,
            batch_size=args.batch_size,
            force=args.force,
        )
        print(f"Embeddings saved to: {embeddings_path}")
        print(f"Embedding manifest saved to: {manifest_path}")
        return

    if args.command == "retrieve":
        results = retrieve(
            args.query,
            args.issuer,
            args.period,
            k=args.k,
            doc_type=args.doc_type,
        )
        for item in results:
            print(
                f"{item['rank']}. "
                f"score={item['score']:.6f} "
                f"uid={item['chunk_uid']} "
                f"doc={item['doc_type']} "
                f"file={item['source_file']} "
                f"chunk={item['chunk_id']}"
            )
            print(item["text_preview"])
        return

    if args.command == "answer":
        result = synthesize_answer(
            query=args.query,
            issuer=args.issuer,
            period=args.period,
            k=args.k,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result["answer_markdown"])
        return

    placeholders = {
        "ask": "Ask placeholder: not implemented.",
    }
    print(placeholders[args.command])


if __name__ == "__main__":
    main()
