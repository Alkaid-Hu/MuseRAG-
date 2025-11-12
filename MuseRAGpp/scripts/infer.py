import argparse
from museragpp.search_generate import search, generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="用户问题")
    ap.add_argument("--topk", type=int, default=None)
    args = ap.parse_args()

    hits = search(args.query, topk=args.topk)
    if not hits:
        print("未检索到相关证据。")
    else:
        out = generate(args.query, hits)
        print("【回答】\n", out["answer"])
        print("\n【引用覆盖率】", out["coverage"])
        print("\n【来源】")
        print("\n".join(out["sources"]))