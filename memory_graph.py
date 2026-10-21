"""
Knowledge graph of long-term memory, for the Bee agent only.

memory/persistent.json is turned into memory/graph.json: one node per stored
fact, the people, places and things those facts mention, and how they relate.
The file uses graphify's node-link JSON format, but it is built here with
networkx and the agent's own model, so memory never goes through graphify's
Claude-based extraction. persistent.json stays the source of truth; the graph
is an index rebuilt from it at the end of every session (see
MemoryStore.end_session) and read by the lookup_memory tool.
"""
import hashlib
import json
import re
import sys
from collections import deque
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph
from pydantic import BaseModel, Field, field_validator
from strands import tool

from memory import CATEGORIES, MEMORY_DIR, Memory, _content_words, _read_json, _structured, _write_json

GRAPH_FILE = "graph.json"
EXTRACT_CHUNK = 8  # facts per extraction call
LOOKUP_CHARS = 3000  # max characters one lookup returns
MAX_START = 3  # best-matching nodes a lookup starts from
SOURCE = "memory/persistent.json"


class Entity(BaseModel):
    name: str = Field(description='As written in the fact, e.g. "Priya", "Pune", "morning run".')
    kind: str = Field(description="One of person, place, organization, activity, thing, time.")


class Relation(BaseModel):
    source: str = Field(description='Entity name, or "user" for the user.')
    relation: str = Field(description='Short snake_case verb phrase, e.g. "sister_of", "lives_in", "goes_on".')
    target: str = Field(description='Entity name, or "user" for the user.')
    fact: int = Field(description="Number of the fact this relation comes from.")


class Extraction(BaseModel):
    """People, places and things in the facts, and how they relate."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @field_validator("entities", "relations", mode="before")
    @classmethod
    def _none_is_empty(cls, value):
        return [] if value is None else value  # Nova sends null for an empty list


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _fact_id(text: str) -> str:
    return "fact_" + hashlib.sha1(text.lower().encode()).hexdigest()[:10]


def _in_text(name: str, text: str) -> bool:
    """True if ``name`` is written in ``text``: every content word of it, or the
    whole name for short ones like "Al". Keeps the model from adding entities
    the fact doesn't mention."""
    words = _content_words(name)
    if words:
        return words <= _content_words(text)
    return bool(name.strip()) and re.search(rf"\b{re.escape(name.strip().lower())}\b", text.lower()) is not None


def _facts(mem: Memory) -> list[tuple[str, str]]:
    """(category, text) for every list item in memory. Profile values are added
    to the graph directly, without the model."""
    return [(c, item) for c in CATEGORIES[1:] for item in getattr(mem, c)]


def _source_hash(mem: Memory) -> str:
    return hashlib.sha256(mem.model_dump_json().encode()).hexdigest()


def _extract(model, facts: list[str]) -> Extraction:
    listing = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))
    prompt = f"""These are facts a personal assistant remembers about its user.

FACTS:
{listing}

List the entities they mention (people, places, organizations, activities,
things, times) and how they relate. Rules:
- Use names exactly as written in the facts. Refer to the user as "user".
- Every relation gives the number of the fact it comes from.
- Only relations a fact states; do not guess or connect different facts.
- Skip vague words like "things", "stuff" or "answers"."""
    return _structured(model, Extraction, prompt)


def build_graph(model, memory_dir: Path = MEMORY_DIR, force: bool = False) -> nx.DiGraph | None:
    """Rebuild memory_dir/graph.json from persistent.json. Returns the graph, or
    None if persistent memory hasn't changed since the last build."""
    path = memory_dir / GRAPH_FILE
    data = _read_json(memory_dir / "persistent.json", {})
    data.pop("updated_at", None)
    mem = Memory(**data)
    digest = _source_hash(mem)
    if not force and _read_json(path, {}).get("graph", {}).get("source_hash") == digest:
        return None

    G = nx.DiGraph(source_hash=digest)
    user_name = mem.profile.get("name", "")
    G.add_node("user", label=user_name or "User", kind="user", file_type="document", source_file=SOURCE)
    aliases = {"user", "the user", "me", "i"} | ({user_name.lower()} if user_name else set())

    def entity(name: str, kind: str) -> str:
        if name.strip().lower() in aliases:
            return "user"
        nid = _slug(name)
        if nid not in G:
            G.add_node(nid, label=name.strip(), kind=kind, file_type="document", source_file=SOURCE)
        return nid

    def edge(src: str, rel: str, tgt: str, confidence: str, score: float, **extra) -> None:
        G.add_edge(src, tgt, relation=rel, confidence=confidence, confidence_score=score,
                   source_file=SOURCE, weight=1.0, **extra)

    for key, value in mem.profile.items():
        if key != "name":
            edge("user", _slug(key), entity(value, "profile"), "EXTRACTED", 1.0)

    facts = _facts(mem)
    for category, text in facts:
        G.add_node(_fact_id(text), label=text, kind="fact", category=category, file_type="document",
                   source_file=SOURCE, source_location=category)
        edge("user", category, _fact_id(text), "EXTRACTED", 1.0)

    for start in range(0, len(facts), EXTRACT_CHUNK):
        chunk = [text for _, text in facts[start:start + EXTRACT_CHUNK]]
        try:
            found = _extract(model, chunk)
        except Exception as exc:  # the facts are still in the graph, just without links
            print(f"(memory graph: extraction failed for {len(chunk)} facts: {exc})")
            continue
        found.entities = [e for e in found.entities if _slug(e.name)]
        kinds = {e.name.strip().lower(): e.kind.strip().lower() or "thing" for e in found.entities}
        # Entities are only kept where a fact actually names them.
        for e in found.entities:
            for text in chunk:
                if e.name.strip().lower() not in aliases and _in_text(e.name, text):
                    edge(entity(e.name, kinds[e.name.strip().lower()]), "mentioned_in", _fact_id(text), "EXTRACTED", 1.0)
        for r in found.relations:
            if not 1 <= r.fact <= len(chunk) or not _slug(r.relation) or not (_slug(r.source) and _slug(r.target)):
                continue
            text = chunk[r.fact - 1]
            ends = [n for n in (r.source, r.target) if n.strip().lower() not in aliases]
            if not all(_in_text(n, text) for n in ends) or r.source.strip().lower() == r.target.strip().lower():
                continue
            src = entity(r.source, kinds.get(r.source.strip().lower(), "thing"))
            tgt = entity(r.target, kinds.get(r.target.strip().lower(), "thing"))
            edge(src, _slug(r.relation), tgt, "INFERRED", 0.8, fact=_fact_id(text))
            for nid in (src, tgt):  # Nova often leaves entities empty and only lists relations
                if nid != "user":
                    edge(nid, "mentioned_in", _fact_id(text), "EXTRACTED", 1.0)

    _write_json(path, json_graph.node_link_data(G, edges="links"))
    return G


_cache: dict[Path, tuple[float, nx.DiGraph]] = {}


def load_graph(memory_dir: Path = MEMORY_DIR) -> nx.DiGraph | None:
    """graph.json as a graph, reloaded only when the file changes."""
    path = memory_dir / GRAPH_FILE
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    G = json_graph.node_link_graph(_read_json(path, {}), edges="links")
    _cache[path] = (mtime, G)
    return G


def lookup(question: str, memory_dir: Path = MEMORY_DIR, max_chars: int = LOOKUP_CHARS) -> str:
    """The stored facts and links around whatever ``question`` names."""
    G = load_graph(memory_dir)
    if G is None or G.number_of_nodes() <= 1:
        return "Long-term memory graph is empty."
    terms = _content_words(question)
    scores = {n: len(terms & _content_words(d["label"])) for n, d in G.nodes(data=True)}
    starts = sorted((n for n, s in scores.items() if s), key=lambda n: (-scores[n], -G.degree(n)))[:MAX_START]
    if not starts and re.search(r"\b(me|my|myself|i)\b", question.lower()):
        starts = ["user"]  # "what do you know about me": everything, cut to max_chars
    if not starts:
        return f"Nothing in long-term memory matches '{question}'."

    # Two hops out, without passing through the user node, which links to every fact.
    U = G.to_undirected(as_view=True)
    seen, queue = set(starts), deque((n, 0) for n in starts)
    while queue:
        node, depth = queue.popleft()
        if depth == 2 or (node == "user" and node not in starts):
            continue
        for nb in U.neighbors(node):
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, depth + 1))

    def rank(n):
        return (-scores[n], -G.degree(n))

    facts = [n for n in sorted(seen, key=rank) if G.nodes[n]["kind"] == "fact"]
    lines = ["Facts:"] + [f"- {G.nodes[n]['label']}" for n in facts]
    links = [(u, v, d) for u, v, d in G.edges(data=True)
             if u in seen and v in seen and d["relation"] != "mentioned_in"
             and G.nodes[v]["kind"] != "fact" and G.nodes[u]["kind"] != "fact"]
    if links:
        lines.append("Links:")
        lines += [f"- {G.nodes[u]['label']} --{d['relation']}--> {G.nodes[v]['label']}" for u, v, d in links]

    out, size = [], 0
    for line in lines:
        if size + len(line) + 1 > max_chars:
            out.append("... (more in memory; ask about something narrower)")
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


def lookup_tool(memory_dir: Path = MEMORY_DIR):
    @tool
    def lookup_memory(question: str) -> str:
        """Search the user's long-term memory graph: what is stored about a person,
        place, activity or thing, and how they connect to the user and each other.
        Use it for questions about the user's life that the memory at the start of
        the conversation doesn't answer directly. It only reads; use remember to save.

        Args:
            question: What to look up, naming the people, places or things, e.g.
                "Priya" or "where does my sister live".
        """
        return lookup(question, memory_dir)

    return lookup_memory


if __name__ == "__main__":
    # python memory_graph.py [memory_dir]: rebuild the graph now, e.g. after
    # editing persistent.json by hand.
    from bee import bedrock_model

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else MEMORY_DIR
    G = build_graph(bedrock_model, target, force=True)
    print(f"{target / GRAPH_FILE}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
