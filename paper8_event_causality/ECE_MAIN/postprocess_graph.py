# 论文八：将预测转为可审核表，导出原因→结果的有向图及边表。
# 中文注释为提交整理时补充，实验算法沿用本机已有版本。
"""Convert ECE predictions into a directed event graph.

The thesis's original domain corpus and manual decisions are unavailable.
Use --accept-all to visualize model output directly, without human review.
"""

import argparse
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARGUMENTS = ("product", "region", "industry")
FIELDS = ["text_id", "source_text", "source", "reason_type", "reason_product",
          "reason_region", "reason_industry", "result_type", "result_product",
          "result_region", "result_industry", "reason_label", "result_label", "decision"]


def parse_record(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def event_label(item, prefix):
    parts = [str(item[f"{prefix}_type"]).strip()]
    for argument in ARGUMENTS:
        value = str(item.get(f"{prefix}_{argument}", "")).strip()
        if value:
            parts.append(value)
    return "｜".join(parts)


def pairwise_char_tfidf_cosine(left, right):
    """Cosine of two character TF-IDF vectors fitted on this pair only.

    This matches sklearn's smoothed IDF and L2 normalization for
    TfidfVectorizer(analyzer="char", ngram_range=(1, 1)).fit_transform([a, b]).
    """
    shared = left.keys() & right.keys()
    unique_idf = 1 + math.log(3 / 2)
    numerator = sum(left[term] * right[term] for term in shared)
    left_norm = math.sqrt(sum(count * count * (1 if term in right else unique_idf) ** 2
                              for term, count in left.items()))
    right_norm = math.sqrt(sum(count * count * (1 if term in left else unique_idf) ** 2
                               for term, count in right.items()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def argument_counter(row, prefix):
    """Character tokens keep the argument field and causal side separate."""
    return Counter((argument, char)
                   for argument in ARGUMENTS
                   for char in row.get(f"{prefix}_{argument}", "").strip().lower())


def argument_cosine(left, right):
    if not left and not right:
        return 1.0
    return pairwise_char_tfidf_cosine(left, right)


# 把预测与原文 ID 对齐，生成供人工核查的 CSV。
def prepare(args):
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    raw = {}
    for line in args.raw_data.read_text(encoding="utf-8").splitlines():
        item = parse_record(line)
        raw[str(item["text_id"])] = item["text"]
    rows = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction = parse_record(line)
        text_id = str(prediction["text_id"])
        if text_id not in raw:
            raise ValueError(f"Prediction ID missing from raw data: {text_id}")
        for pair in prediction["result"]:
            row = {key: pair.get(key, "") for key in FIELDS}
            row.update(text_id=text_id, source_text=raw[text_id], source=args.source,
                       reason_label=event_label(pair, "reason"),
                       result_label=event_label(pair, "result"), decision="")
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predicted event pairs to {args.output}")
    print("Use export --accept-all to visualize every prediction directly.")


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def component_layout(graph, nx):
    """Pack disconnected causal chains into readable regions of the canvas."""
    if graph.number_of_nodes() == 0:
        return {}
    components = sorted(nx.connected_components(graph.to_undirected()),
                        key=lambda items: (-len(items), min(items)))
    columns = max(1, math.ceil(math.sqrt(len(components) * 1000 / 700)))
    rows = math.ceil(len(components) / columns)
    cell_w, cell_h = 1000 / columns, 700 / rows
    positions = {}
    for index, component in enumerate(components):
        center_x = (index % columns + 0.5) * cell_w
        center_y = (index // columns + 0.5) * cell_h
        subgraph = graph.subgraph(component)
        if len(component) == 1:
            positions[next(iter(component))] = (center_x, center_y)
        elif len(component) == 2:
            source, target = next(iter(subgraph.edges()))
            positions[source] = (center_x - 25, center_y - min(34, cell_h * 0.23))
            positions[target] = (center_x - 25, center_y + min(34, cell_h * 0.23))
        else:
            local = nx.spring_layout(subgraph, seed=42)
            for node, xy in local.items():
                positions[node] = (center_x + xy[0] * cell_w * 0.32,
                                   center_y + xy[1] * cell_h * 0.32)
    return positions


# 按审核决定或 accept-all 选择边；可选相似去重和静态/交互图输出。
def export(args):
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    with args.reviewed.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.accept_all:
        kept = rows
    else:
        invalid = [row for row in rows if row.get("decision", "").strip().lower() not in ("keep", "reject")]
        if invalid:
            raise SystemExit(f"{len(invalid)} row(s) lack a keep/reject decision")
        kept = [row for row in rows if row["decision"].strip().lower() == "keep"]
    for row in kept:
        if not row.get("reason_label") or not row.get("result_label"):
            raise SystemExit("Kept pair has an empty event label")

    representative_indices = []
    groups = []
    assignments = []
    if args.tfidf_threshold is not None and kept:
        # The thesis gives TF-IDF and 0.40, but no executable Chinese tokenizer.
        # Character features are explicit here and should be reported as such.
        texts = [row["reason_label"] + " → " + row["result_label"] for row in kept]
        if args.structure_aware:
            arguments = [{prefix: argument_counter(row, prefix) for prefix in ("reason", "result")}
                         for row in kept]
        elif args.tfidf_fit_scope == "pair":
            counters = [Counter(text.lower()) for text in texts]
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectors = TfidfVectorizer(analyzer="char", ngram_range=(1, 1)).fit_transform(texts)
    for index, row in enumerate(kept):
        if args.tfidf_threshold is None:
            match = next((j for j, rep in enumerate(representative_indices)
                          if row["reason_label"] == kept[rep]["reason_label"]
                          and row["result_label"] == kept[rep]["result_label"]), None)
        else:
            if args.structure_aware:
                similarities = [
                    min(argument_cosine(arguments[index]["reason"], arguments[rep]["reason"]),
                        argument_cosine(arguments[index]["result"], arguments[rep]["result"]))
                    if (row["reason_type"] == kept[rep]["reason_type"]
                        and row["result_type"] == kept[rep]["result_type"]) else 0.0
                    for rep in representative_indices]
            elif args.tfidf_fit_scope == "pair":
                similarities = [pairwise_char_tfidf_cosine(counters[index], counters[rep])
                                for rep in representative_indices]
            else:
                similarities = ((vectors[index] @ vectors[representative_indices].T).toarray()[0]
                                if representative_indices else [])
            match = next((j for j, score in enumerate(similarities)
                          if score >= args.tfidf_threshold), None)
        if match is None:
            representative_indices.append(index)
            groups.append([row])
            representative = index
            score = None
        else:
            groups[match].append(row)
            representative = representative_indices[match]
            score = float(similarities[match]) if args.tfidf_threshold is not None else 1.0
        if args.tfidf_threshold is not None:
            assignments.append((index, representative, score))

    node_ids = {}
    nodes = []
    edges = []
    pairs = []
    for representative, group in zip(representative_indices, groups):
        row = kept[representative]
        for label in (row["reason_label"], row["result_label"]):
            if label not in node_ids:
                identifier = str(len(node_ids) + 1)
                node_ids[label] = identifier
                nodes.append({"Id": identifier, "Label": label})
        ids = ";".join(item["text_id"] for item in group)
        edges.append({"Source": node_ids[row["reason_label"]],
                      "Target": node_ids[row["result_label"]],
                      "Type": "Directed", "Weight": len(group), "EvidenceIds": ids})
        pairs.append({"reason_label": row["reason_label"],
                      "result_label": row["result_label"],
                      "first_text_id": row["text_id"], "evidence_ids": ids,
                      "merged_count": len(group)})

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "nodes.csv", ["Id", "Label"], nodes)
    write_csv(args.output_dir / "edges.csv", ["Source", "Target", "Type", "Weight", "EvidenceIds"], edges)
    write_csv(args.output_dir / "event_pairs.csv",
              ["reason_label", "result_label", "first_text_id", "evidence_ids", "merged_count"], pairs)
    type_mismatched_merges = 0
    if args.tfidf_threshold is not None:
        audit = []
        for index, representative, score in assignments:
            row, first = kept[index], kept[representative]
            same_reason_type = row["reason_type"] == first["reason_type"]
            same_result_type = row["result_type"] == first["result_type"]
            if index != representative and not (same_reason_type and same_result_type):
                type_mismatched_merges += 1
            audit.append({"row_index": index, "text_id": row["text_id"],
                          "reason_label": row["reason_label"], "result_label": row["result_label"],
                          "representative_row_index": representative,
                          "representative_reason_label": first["reason_label"],
                          "representative_result_label": first["result_label"],
                          "cosine_to_representative": "" if score is None else f"{score:.6f}",
                          "same_reason_type": same_reason_type,
                          "same_result_type": same_result_type})
        write_csv(args.output_dir / "merge_audit.csv",
                  ["row_index", "text_id", "reason_label", "result_label",
                   "representative_row_index", "representative_reason_label",
                   "representative_result_label", "cosine_to_representative",
                   "same_reason_type", "same_result_type"], audit)

    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from((node["Id"], {"label": node["Label"]}) for node in nodes)
    graph.add_edges_from((edge["Source"], edge["Target"],
                          {"weight": edge["Weight"], "evidence_ids": edge["EvidenceIds"]})
                         for edge in edges)
    nx.write_gexf(graph, args.output_dir / "graph.gexf")
    positions = component_layout(graph, nx)
    sources = {row["source"] for row in kept}
    provenance = ("公开金标准流程演示" if sources == {"gold_demo"}
                  else "模型直接预测" if args.accept_all else "已审核模型结果")
    payload = {
        "provenance": provenance,
        "nodes": [{"id": node["Id"], "label": node["Label"],
                   "x": round(positions[node["Id"]][0], 2),
                   "y": round(positions[node["Id"]][1], 2)} for node in nodes],
        "edges": [{"source": edge["Source"], "target": edge["Target"],
                   "weight": edge["Weight"], "evidence": edge["EvidenceIds"]} for edge in edges],
    }
    safe_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "graph_template.html").read_text(encoding="utf-8")
    (args.output_dir / "graph.html").write_text(
        template.replace("__GRAPH_DATA__", safe_json), encoding="utf-8")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(18, 14))
        labels = {node["Id"]: node["Label"][:12] for node in nodes}
        if graph.number_of_nodes():
            nx.draw_networkx(graph, positions, labels=labels, ax=ax, node_size=750,
                             font_size=8, arrows=True, arrowsize=12, width=0.8,
                             node_color="#b8d9ef", edge_color="#617a8b")
        else:
            ax.text(0.5, 0.5, "No predicted causal pairs", ha="center", va="center",
                    transform=ax.transAxes, fontsize=18)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.output_dir / "graph.png", dpi=180)
        plt.close(fig)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "prediction_rows": len(rows), "kept_rows": len(kept), "merged_pairs": len(pairs),
        "nodes": len(nodes), "edges": len(edges),
        "accept_all": args.accept_all,
        "tfidf_threshold": args.tfidf_threshold,
        "tfidf_features": ("Argument-field character unigrams" if args.structure_aware else
                           "Chinese character unigrams" if args.tfidf_threshold is not None else None),
        "tfidf_fit_scope": args.tfidf_fit_scope if args.tfidf_threshold is not None else None,
        "structure_aware": args.structure_aware,
        "type_mismatched_merges": type_mismatched_merges if args.tfidf_threshold is not None else None,
        "note": ("All predicted pairs accepted without review. " if args.accept_all else "Decisions read from reviewed CSV. ")
                + "Event labels derive from ECE type and arguments unless edited. "
                + ("Character tokenization is an explicit approximation; the thesis does not specify it."
                   if args.tfidf_threshold is not None else "")
                + (" Exact reason/result event types and separate argument similarities are extra safeguards, not thesis code."
                   if args.structure_aware else "")
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(nodes)} nodes and {len(edges)} directed edges to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="create event-pair CSV from model predictions")
    prep.add_argument("--predictions", type=Path, required=True)
    prep.add_argument("--raw-data", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--source", default="model", help="provenance label recorded in review CSV")
    prep.set_defaults(func=prepare)
    exp = sub.add_parser("export", help="deduplicate pairs and export a directed graph")
    exp.add_argument("--reviewed", type=Path, required=True)
    exp.add_argument("--output-dir", type=Path, required=True)
    exp.add_argument("--accept-all", action="store_true", help="use all model predictions without manual review")
    exp.add_argument("--tfidf-threshold", type=float, default=None,
                     help="optional thesis-inspired character TF-IDF cosine threshold, e.g. 0.4")
    exp.add_argument("--tfidf-fit-scope", choices=("corpus", "pair"), default="corpus",
                     help="fit TF-IDF on all pairs or separately on each compared pair as in thesis 5.3")
    exp.add_argument("--structure-aware", action="store_true",
                     help="extra safeguard: exact cause/effect types and separate argument similarities")
    exp.add_argument("--plot", action="store_true", help="also create a static PNG preview")
    exp.set_defaults(func=export)
    args = parser.parse_args()
    if args.command == "export" and args.tfidf_threshold is not None and not 0 <= args.tfidf_threshold <= 1:
        parser.error("tfidf-threshold must be between 0 and 1")
    if args.command == "export" and args.structure_aware and (args.tfidf_threshold is None or args.tfidf_fit_scope != "pair"):
        parser.error("--structure-aware requires --tfidf-threshold and --tfidf-fit-scope pair")
    args.func(args)


if __name__ == "__main__":
    main()
