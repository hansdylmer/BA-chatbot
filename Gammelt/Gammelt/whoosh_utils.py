from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser

def whoosh_search(query, top_k=5):
    ix = open_dir("whoosh_index")
    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "heading", "content"], schema=ix.schema)
        parsed_query = parser.parse(query)
        results = searcher.search(parsed_query, limit=top_k)
        return [{"id": r["id"], "heading": r["heading"], "title": r["title"], "content": r["content"]} for r in results]

results = whoosh_search("2025")
for r in results:
    print(f"{r['id']} | {r['heading']}\n{r['content'][:150]}...\n")