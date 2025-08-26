
"""
eval_utils.py — Lightweight recommender evaluation helpers.

Functions:
- segment_users_by_interactions(user_interactions, bins=None)
- evaluate_model(preds, truth, K_LIST, catalog=None, item_categories=None, item_popularity=None)
- evaluate_by_segment(preds, truth, segments, K_LIST, catalog=None, item_categories=None, item_popularity=None)

Inputs:
- preds: dict[user_id] -> ranked list of item_ids (longer than max K)
- truth: dict[user_id] -> set/list of held-out relevant item_ids
- K_LIST: list of cutoffs, e.g., [5, 10, 20]
- catalog: set/list of all item_ids (for coverage@K). Optional but recommended.
- item_categories: dict[item_id] -> category_id (for ILD@K). Optional.
- item_popularity: dict[item_id] -> interaction_count (for novelty@K). Optional.

All functions are pure-Python + NumPy/Pandas and should run in most notebooks.
"""

from __future__ import annotations
from typing import Dict, List, Iterable, Optional, Tuple, Any
import math
import numpy as np
import pandas as pd
from collections import defaultdict


# ----------------------------
# Utilities
# ----------------------------

def _to_set(x: Iterable[Any]) -> set:
    """Convert labels to a set safely."""
    if x is None:
        return set()
    if isinstance(x, set):
        return x
    return set(x)


def _precision_recall_hit(pred_k: List[Any], rel_set: set) -> Tuple[float, float, int]:
    """Return (precision@k, recall@k, hit@k) for a single user."""
    if not pred_k:
        return 0.0, 0.0, 0
    if not rel_set:
        # No relevant items: precision defined, recall = 0 by convention
        inter = 0
    else:
        inter = sum(1 for i in pred_k if i in rel_set)
    precision = inter / len(pred_k)
    recall = inter / max(1, len(rel_set))
    hit = 1 if inter > 0 else 0
    return precision, recall, hit


def _apk(pred: List[Any], rel_set: set, k: int) -> float:
    """Average Precision @ K for one user."""
    if k <= 0:
        return 0.0
    score = 0.0
    hits = 0
    for i, p in enumerate(pred[:k], start=1):
        if p in rel_set:
            hits += 1
            score += hits / i
    denom = min(k, len(rel_set)) if len(rel_set) > 0 else k
    return score / max(1, denom)


def _ndcg(pred: List[Any], rel_set: set, k: int) -> float:
    """nDCG@K with binary relevance."""
    dcg = 0.0
    for i, p in enumerate(pred[:k], start=1):
        if p in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(rel_set))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def _coverage_at_k(all_preds: Dict[Any, List[Any]], k: int, catalog: Optional[Iterable[Any]]) -> float:
    """Proportion of catalog that appears at least once in any user's top-k recommendations."""
    if catalog is None:
        return float('nan')
    seen = set()
    for u, items in all_preds.items():
        seen.update(items[:k])
    return len(seen) / max(1, len(set(catalog)))


def _ild_at_k(items: List[Any], item_categories: Optional[Dict[Any, Any]], k: int) -> float:
    """Intra-list diversity @ K using category dissimilarity (1 if categories differ else 0)."""
    if item_categories is None or k <= 1:
        return float('nan')
    items_k = items[:k]
    cats = [item_categories.get(x) for x in items_k]
    n = len(cats)
    if n <= 1:
        return float('nan')
    pairs = 0
    diverse = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if cats[i] is None or cats[j] is None:
                # skip unknowns rather than assuming diverse/similar
                continue
            if cats[i] != cats[j]:
                diverse += 1
    return diverse / pairs if pairs else float('nan')


def _novelty_at_k(items: List[Any], item_popularity: Optional[Dict[Any, int]], total_pop: Optional[int], k: int) -> float:
    """Novelty @ K = average -log2(popularity / total_pop) of recommended items. Higher = more novel.
    If popularity missing, returns NaN.
    """
    if item_popularity is None or not item_popularity:
        return float('nan')
    if total_pop is None or total_pop <= 0:
        total_pop = sum(max(1, c) for c in item_popularity.values())
    vals = []
    for it in items[:k]:
        p = item_popularity.get(it, None)
        if p is None or p <= 0:
            # Skip unknowns to avoid distorting novelty
            continue
        vals.append(-math.log2(p / total_pop))
    return float(np.mean(vals)) if vals else float('nan')


# ----------------------------
# Public API
# ----------------------------

def segment_users_by_interactions(user_interactions: Dict[Any, int],
                                  bins: Optional[Dict[str, Tuple[int, Optional[int]]]] = None
                                 ) -> Dict[Any, str]:
    """
    Segment users by their historical interaction counts.

    bins: mapping of segment_name -> (min_inclusive, max_inclusive or None for open-ended)
          default = {'cold': (0, 5), 'warm': (6, 20), 'hot': (21, None)}

    Returns: dict[user_id] -> segment_name
    """
    if bins is None:
        bins = {'cold': (0, 5), 'warm': (6, 20), 'hot': (21, None)}
    result = {}
    for u, cnt in user_interactions.items():
        seg = None
        for name, (lo, hi) in bins.items():
            if cnt >= lo and (hi is None or cnt <= hi):
                seg = name
                break
        result[u] = seg if seg is not None else 'unsegmented'
    return result


def evaluate_model(preds: Dict[Any, List[Any]],
                   truth: Dict[Any, Iterable[Any]],
                   K_LIST: List[int],
                   catalog: Optional[Iterable[Any]] = None,
                   item_categories: Optional[Dict[Any, Any]] = None,
                   item_popularity: Optional[Dict[Any, int]] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute standard metrics at each K in K_LIST.
    Returns: (overall_df, per_user_df)

    overall_df columns (per K): precision@K, recall@K, map@K, ndcg@K, hit_rate@K, coverage@K, ild@K, novelty@K
    per_user_df columns: user_id, K, precision, recall, ap, ndcg, hit, ild, novelty
    """
    users = sorted(set(preds.keys()) & set(truth.keys()))
    if not users:
        raise ValueError("No overlap between preds and truth users.")

    maxK = max(K_LIST)
    rel = {u: _to_set(truth.get(u, [])) for u in users}

    per_user_rows = []
    overall_rows = []

    # Precompute totals for coverage/novelty
    total_pop = sum(max(1, c) for c in item_popularity.values()) if item_popularity else None

    for K in K_LIST:
        precs, recs, aps, ndcgs, hits, ilds, novs = [], [], [], [], [], [], []

        for u in users:
            pred_u = preds.get(u, [])[:maxK]  # ensure we have enough
            rel_u = rel[u]

            p, r, h = _precision_recall_hit(pred_u[:K], rel_u)
            ap = _apk(pred_u, rel_u, K)
            nd = _ndcg(pred_u, rel_u, K)
            ild = _ild_at_k(pred_u, item_categories, K)
            nov = _novelty_at_k(pred_u, item_popularity, total_pop, K)

            precs.append(p); recs.append(r); aps.append(ap); ndcgs.append(nd); hits.append(h)
            ilds.append(ild); novs.append(nov)

            per_user_rows.append({
                "user_id": u, "K": K,
                "precision": p, "recall": r, "ap": ap, "ndcg": nd, "hit": h,
                "ild": ild, "novelty": nov
            })

        coverage = _coverage_at_k(preds, K, catalog)
        overall_rows.append({
            "K": K,
            f"precision@{K}": float(np.mean(precs)) if precs else float('nan'),
            f"recall@{K}": float(np.mean(recs)) if recs else float('nan'),
            f"map@{K}": float(np.mean(aps)) if aps else float('nan'),
            f"ndcg@{K}": float(np.mean(ndcgs)) if ndcgs else float('nan'),
            f"hit_rate@{K}": float(np.mean(hits)) if hits else float('nan'),
            f"coverage@{K}": coverage,
            f"ild@{K}": float(np.nanmean(ilds)) if ilds else float('nan'),
            f"novelty@{K}": float(np.nanmean(novs)) if novs else float('nan'),
        })

    overall_df = pd.DataFrame(overall_rows).set_index("K")
    per_user_df = pd.DataFrame(per_user_rows)
    return overall_df, per_user_df


def evaluate_by_segment(preds: Dict[Any, List[Any]],
                        truth: Dict[Any, Iterable[Any]],
                        segments: Dict[Any, str],
                        K_LIST: List[int],
                        catalog: Optional[Iterable[Any]] = None,
                        item_categories: Optional[Dict[Any, Any]] = None,
                        item_popularity: Optional[Dict[Any, int]] = None
                       ) -> pd.DataFrame:
    """
    Evaluate metrics per segment. Returns a wide DataFrame:
    index: segment, columns: metrics per K (precision@5, recall@5, ..., novelty@K).
    """
    # group users by segment
    seg_to_users = defaultdict(list)
    for u, seg in segments.items():
        if u in preds and u in truth:
            seg_to_users[seg].append(u)

    rows = []
    for seg, seg_users in seg_to_users.items():
        if not seg_users:
            continue
        # Slice dicts for the segment
        preds_seg = {u: preds[u] for u in seg_users}
        truth_seg = {u: truth[u] for u in seg_users}

        overall_df, _ = evaluate_model(preds_seg, truth_seg, K_LIST, catalog, item_categories, item_popularity)
        row = {"segment": seg}
        for K in K_LIST:
            row.update({
                f"precision@{K}": overall_df.loc[K, f"precision@{K}"],
                f"recall@{K}": overall_df.loc[K, f"recall@{K}"],
                f"map@{K}": overall_df.loc[K, f"map@{K}"],
                f"ndcg@{K}": overall_df.loc[K, f"ndcg@{K}"],
                f"hit_rate@{K}": overall_df.loc[K, f"hit_rate@{K}"],
                f"coverage@{K}": overall_df.loc[K, f"coverage@{K}"],
                f"ild@{K}": overall_df.loc[K, f"ild@{K}"],
                f"novelty@{K}": overall_df.loc[K, f"novelty@{K}"],
            })
        rows.append(row)

    df = pd.DataFrame(rows).set_index("segment").sort_index()
    return df
