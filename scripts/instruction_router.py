# --*-- coding: utf-8 -*-
# @Function: 开放指令意图路由器 v2（方案①系统级方向控制）。
#   两级解析：①词法规则层（模态关键词族，高精度，优先）；②CLIP 锚点层
#   （多锚点 max 相似度，处理无关键词的语义意图，margin 不足时弃权）。
#   生成核心不动——路由层负责"指令→方向"，ID 机制继续保底。
#
# 运行（服务器或任意有 torch+clip 的机器）:
#   1) 内嵌测试集评测（显式 28 条 + 语义意图 27 条，共 55 条）:
#      python scripts/instruction_router.py
#   2) 追加评测 3,000 条正式指令池的方向一致率:
#      python scripts/instruction_router.py --eval-pool dataloaders/instructions_a2a_v2.json
#   3) 单句演示: python scripts/instruction_router.py --demo "I want to see this in infrared"
#   4) 消融词法层（纯 CLIP）: python scripts/instruction_router.py --no-lexical
#
# 输出: results/instruction_router/routing_eval.json（准确率/混淆矩阵/分层统计/弃权率）
#       results/instruction_router/pool_direction_agreement.json（池方向一致率）

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip

MODALITY_NAMES = {0: "rgb", 1: "nir", 2: "dem", 3: "spot"}
MODALITY_DISPLAY = {0: "RGB", 1: "NIR", 2: "DEM", 3: "SPOT"}

# ============ 第 1 级：词法规则层（高精度关键词族，正则词边界） ============
# 短 token 用 \b 词边界；多词短语用普通子串。命中计数唯一最大者直接判定。
LEXICAL_KEYWORDS = {
    0: [r"\brgb\b", "true color", "natural color", "visible spectrum", "real color",
        "human eye", "eye would see", "appearance", "looks like", "photograph",
        "tourism", "brochure", "photorealistic"],
    1: [r"\bnir\b", "infrared", "chlorophyll", "vegetation", r"\bcrop", r"\bcrops\b",
        "canopy", "plant health", "biomass", "coniferous", "deciduous", "vigor",
        "drought", "photosynthetic"],
    2: [r"\bdem\b", "elevation", "height map", r"\bheights?\b", "topograph",
        "surface model", r"\bslope\b", "lidar", "earthwork", "drainage",
        "line of sight", "flood", "3d structure", "terrain model"],
    3: ["satellite", "spaceborne", r"\bspot\b", "orbit", "from space", "space photo",
        "space view", "regional context"],
}
_LEXICAL_RES = {mid: [re.compile(p, re.I) for p in pats] for mid, pats in LEXICAL_KEYWORDS.items()}


def lexical_route(text):
    """词法层：返回 (modality_id or None, 命中详情)。唯一最大命中族才判定。"""
    hits = {}
    for mid, res in _LEXICAL_RES.items():
        matched = [r.pattern for r in res if r.search(text)]
        if matched:
            hits[mid] = matched
    if not hits:
        return None, {}
    best = max(hits.values(), key=len)
    winners = [mid for mid, m in hits.items() if len(m) == len(best)]
    if len(winners) != 1:
        return None, hits  # 平票 → 交给 CLIP 层
    return winners[0], hits


# ============ 第 2 级：CLIP 锚点层（语义意图） ============
# 锚点按"模态的本质属性"设计：SPOT=平台（卫星/轨道），NIR=光谱（红外/植被生理），
# DEM=几何（高程/地形），RGB=外观（自然色/真实观感）——避免语义族互相渗透。
MODALITY_ANCHORS = {
    0: [
        "a true-color aerial photograph in natural visible colors",
        "an RGB aerial photo showing the scene as the human eye sees it",
        "a natural color image of the landscape",
        "a realistic photo of the actual appearance of buildings and fields",
        "a photorealistic visible-light picture of the ground",
        "a picture showing how the scene looks to the eye",
    ],
    1: [
        "a near-infrared aerial image",
        "an infrared picture where vegetation appears bright",
        "a grayscale near-infrared reflectance map of plants",
        "a vegetation vigor map showing crop and canopy health",
        "an infrared rendering highlighting chlorophyll reflectance",
        "a plant stress map derived from infrared reflectance",
    ],
    2: [
        "a digital elevation model of the terrain",
        "a topographic height map showing ground elevation",
        "a grayscale surface model where brightness encodes height",
        "a terrain slope and elevation map",
        "a height model showing how tall structures and the ground are",
        "a geospatial elevation product from height measurements",
    ],
    3: [
        "a satellite image of the region captured from space",
        "a picture taken by an Earth observation satellite",
        "a spaceborne view of the landscape from orbit",
        "a wide-coverage satellite picture of the territory",
        "the same area seen from a satellite in space",
    ],
}

# 内嵌路由测试集：(指令, 期望模态ID, 类型 explicit|semantic)
ROUTING_TEST_SET = [
    # ---- RGB: 显式 ----
    ("give me the true color version of this image", 0, "explicit"),
    ("show this area in natural colors", 0, "explicit"),
    ("make it look like a normal aerial photo", 0, "explicit"),
    ("i want the RGB picture", 0, "explicit"),
    ("render the scene as the human eye would see it", 0, "explicit"),
    ("convert this to a natural color photograph", 0, "explicit"),
    ("show me the visible spectrum image", 0, "explicit"),
    ("produce a true color aerial view", 0, "explicit"),
    # ---- RGB: 语义意图 ----
    ("i need to check the actual appearance of the rooftops for a site report", 0, "semantic"),
    ("what does this place look like to a visitor", 0, "semantic"),
    ("i want to verify the painted markings on the sports field", 0, "semantic"),
    ("show the scene for a visual inspection of the buildings", 0, "semantic"),
    ("i need a picture for the tourism brochure of this village", 0, "semantic"),
    ("let me see the real colors of the fields and roads", 0, "semantic"),
    # ---- NIR: 显式 ----
    ("make this an infrared picture", 1, "explicit"),
    ("show me the near-infrared version", 1, "explicit"),
    ("i want to see this in infrared", 1, "explicit"),
    ("give me the NIR image", 1, "explicit"),
    ("convert this to a near-infrared rendering", 1, "explicit"),
    ("show the scene in the infrared band", 1, "explicit"),
    ("produce the near-infrared counterpart of this photo", 1, "explicit"),
    # ---- NIR: 语义意图 ----
    ("i am assessing crop vigor across these parcels", 1, "semantic"),
    ("map the vegetation stress after the dry weeks", 1, "semantic"),
    ("highlight where the forest canopy is densest", 1, "semantic"),
    ("i am monitoring plant health in this agricultural zone", 1, "semantic"),
    ("show which fields have the strongest biomass", 1, "semantic"),
    ("i need to discriminate coniferous from deciduous stands", 1, "semantic"),
    ("estimate the drought impact on the vegetation here", 1, "semantic"),
    ("reveal the chlorophyll patterns of the crops", 1, "semantic"),
    # ---- DEM: 显式 ----
    ("show me the elevation model of this area", 2, "explicit"),
    ("i need the digital surface model", 2, "explicit"),
    ("give me the terrain height map", 2, "explicit"),
    ("convert this to a DEM", 2, "explicit"),
    ("show the topography and ground elevation", 2, "explicit"),
    ("produce a height map of the buildings and terrain", 2, "explicit"),
    ("render the elevation surface", 2, "explicit"),
    # ---- DEM: 语义意图 ----
    ("i am checking building heights for a construction permit", 2, "semantic"),
    ("assess the slope stability of this hillside", 2, "semantic"),
    ("i need to model flood propagation in this valley", 2, "semantic"),
    ("plan the drainage network for this new district", 2, "semantic"),
    ("compute line of sight from the radio tower on the hill", 2, "semantic"),
    ("estimate the earthwork volume for this construction site", 2, "semantic"),
    ("i am studying landslide susceptibility along the slope", 2, "semantic"),
    ("show me the 3D structure of the city blocks", 2, "semantic"),
    # ---- SPOT: 显式 ----
    ("give me the satellite version of this scene", 3, "explicit"),
    ("show this area as seen from a satellite", 3, "explicit"),
    ("i want the spaceborne image", 3, "explicit"),
    ("simulate the SPOT satellite view", 3, "explicit"),
    ("convert this to a satellite picture", 3, "explicit"),
    ("show me what the observation satellite captures here", 3, "explicit"),
    # ---- SPOT: 语义意图 ----
    ("i need regional context beyond this single aerial tile", 3, "semantic"),
    ("cross-check this area against the spaceborne archive", 3, "semantic"),
    ("show the wider territory around this site from orbit", 3, "semantic"),
    ("i am building a large-area mosaic from satellite coverage", 3, "semantic"),
    ("let me see the coarser spaceborne view of the whole district", 3, "semantic"),
]


def _load_clip(model_name, device):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    model, _ = clip.load(model_name, device=device)
    model.eval()
    return model


class InstructionRouter:
    """开放指令 → 目标模态 ID。两级：词法规则（优先）→ CLIP 锚点（兜底，可弃权）。"""

    def __init__(self, clip_model_name="ViT-L/14", device=None,
                 margin_tau=0.008, min_abs_sim=0.15, use_lexical=True):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.model = _load_clip(clip_model_name, self.device)
        self.margin_tau = margin_tau
        self.min_abs_sim = min_abs_sim
        self.use_lexical = use_lexical
        with torch.no_grad():
            self.anchor_embs = {}
            for mid, anchors in MODALITY_ANCHORS.items():
                toks = clip.tokenize(anchors, truncate=True).to(self.device)
                embs = self.model.encode_text(toks)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                self.anchor_embs[mid] = embs

    def _clip_scores(self, emb):
        return {mid: float((emb @ a.T).max()) for mid, a in self.anchor_embs.items()}

    @staticmethod
    def _decide_clip(sims, margin_tau, min_abs_sim):
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_sim = ranked[0]
        margin = top_sim - ranked[1][1]
        abstain = margin < margin_tau or top_sim < min_abs_sim
        return (-1 if abstain else top_id), margin, abstain

    def route(self, text):
        """返回 (modality_id or -1 表示弃权, info dict)。"""
        t = self.torch
        info = {"layer": "clip", "sims": None, "margin": None, "abstain": False}
        if self.use_lexical:
            lid, hits = lexical_route(text)
            if lid is not None:
                info.update({"layer": "lexical", "hits": hits})
                return lid, info
        with t.no_grad():
            tok = clip.tokenize([text], truncate=True).to(self.device)
            emb = self.model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = self._clip_scores(emb)
        pid, margin, abstain = self._decide_clip(sims, self.margin_tau, self.min_abs_sim)
        info.update({"sims": sims, "margin": margin, "abstain": bool(abstain)})
        return pid, info

    def route_batch(self, texts):
        t = self.torch
        results = [None] * len(texts)
        need_clip = []
        for i, text in enumerate(texts):
            if self.use_lexical:
                lid, hits = lexical_route(text)
                if lid is not None:
                    results[i] = (lid, {"layer": "lexical", "hits": hits,
                                        "sims": None, "margin": None, "abstain": False})
                    continue
            need_clip.append(i)
        if need_clip:
            with t.no_grad():
                toks = clip.tokenize([texts[i] for i in need_clip], truncate=True).to(self.device)
                embs = self.model.encode_text(toks)
                embs = embs / embs.norm(dim=-1, keepdim=True)
            for j, i in enumerate(need_clip):
                sims = self._clip_scores(embs[j])
                pid, margin, abstain = self._decide_clip(sims, self.margin_tau, self.min_abs_sim)
                results[i] = (pid, {"layer": "clip", "sims": sims, "margin": margin,
                                    "abstain": bool(abstain)})
        return results


def run_eval(router, out_dir):
    texts = [t for t, _, _ in ROUTING_TEST_SET]
    preds = router.route_batch(texts)
    per_type = {"explicit": [0, 0], "semantic": [0, 0]}  # [correct, total]
    per_layer = {"lexical": [0, 0], "clip": [0, 0]}
    per_modality = {m: [0, 0] for m in MODALITY_NAMES.values()}
    confusion = {m: {n: 0 for n in MODALITY_NAMES.values()} for m in MODALITY_NAMES.values()}
    abstain_count, records = 0, []
    for (text, gold, typ), (pid, info) in zip(ROUTING_TEST_SET, preds):
        correct = (pid == gold)
        abstain_count += int(info["abstain"])
        layer = info["layer"]
        per_type[typ][1] += 1
        per_type[typ][0] += int(correct)
        per_layer[layer][1] += 1
        per_layer[layer][0] += int(correct)
        gname = MODALITY_NAMES[gold]
        per_modality[gname][1] += 1
        per_modality[gname][0] += int(correct)
        if not info["abstain"]:
            confusion[gname][MODALITY_NAMES[pid]] += 1
        rec = {"text": text, "gold": MODALITY_DISPLAY[gold], "type": typ, "layer": layer,
               "pred": MODALITY_DISPLAY.get(pid, "ABSTAIN"),
               "margin": round(info["margin"], 4) if info["margin"] is not None else None,
               "correct": bool(correct)}
        if not correct and info["sims"]:
            rec["sims"] = {MODALITY_DISPLAY[k]: round(v, 4) for k, v in info["sims"].items()}
        records.append(rec)
    total = len(ROUTING_TEST_SET)
    total_correct = sum(c for c, _ in per_type.values())
    summary = {
        "overall_accuracy": round(total_correct / total, 4),
        "explicit_accuracy": round(per_type["explicit"][0] / per_type["explicit"][1], 4),
        "semantic_accuracy": round(per_type["semantic"][0] / per_type["semantic"][1], 4),
        "abstain_rate": round(abstain_count / total, 4),
        "layer_usage": {k: {"acc": round(c / n, 4) if n else None, "n": n}
                        for k, (c, n) in per_layer.items()},
        "per_modality": {m: {"acc": round(c / n, 4) if n else None, "n": n}
                         for m, (c, n) in per_modality.items()},
        "confusion": confusion,
        "margin_tau": router.margin_tau, "min_abs_sim": router.min_abs_sim,
        "use_lexical": router.use_lexical, "n_items": total,
    }
    print("=" * 64)
    print(f"Routing accuracy: overall {summary['overall_accuracy']:.1%} | "
          f"explicit {summary['explicit_accuracy']:.1%} | "
          f"semantic {summary['semantic_accuracy']:.1%} | "
          f"abstain {summary['abstain_rate']:.1%}")
    print(f"  layer usage: lexical n={per_layer['lexical'][0]}/{per_layer['lexical'][1]}, "
          f"clip n={per_layer['clip'][0]}/{per_layer['clip'][1]}")
    for m, d in summary["per_modality"].items():
        print(f"  {m:>5}: {d['acc'] if d['acc'] is not None else 'n/a'} (n={d['n']})")
    print("  错例（含 CLIP 层相似度，用于改进锚点）:")
    for r in records:
        if not r["correct"]:
            print(f"    [{r['type']}/{r['layer']}] {r['text']!r} gold={r['gold']} "
                  f"pred={r['pred']} sims={r.get('sims')}")
    print("=" * 64)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "routing_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out}")
    return summary


def run_pool_eval(router, pool_path, out_dir):
    """正式指令池的方向一致率。池结构：{direction: [{"instruction":..., "category":...}, ...]}"""
    with open(pool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    report, all_texts, meta = {}, [], []
    for key, items in pool.items():
        m = re.match(r'^(rgb|nir|dem|spot)_to_(rgb|nir|dem|spot)$', key) \
            or re.match(r'^(rgb|nir|dem|spot)2(rgb|nir|dem|spot)$', key)
        if not isinstance(items, list) or not m:
            continue
        tgt_id = {"rgb": 0, "nir": 1, "dem": 2, "spot": 3}[m.group(2)]
        for it in items:
            text = it.get("instruction") if isinstance(it, dict) else it
            if isinstance(text, str) and text.strip():
                all_texts.append(text)
                meta.append((key, tgt_id))
    if not all_texts:
        print(f"pool {pool_path} 未解析出指令，跳过池评测")
        return
    preds = router.route_batch(all_texts)
    per_dir = {}
    for (key, tgt_id), (pid, info) in zip(meta, preds):
        c, n = per_dir.get(key, [0, 0])
        per_dir[key] = [c + int(pid == tgt_id), n + 1]
    for key, (c, n) in sorted(per_dir.items()):
        report[key] = {"agree_rate": round(c / n, 4), "n": n}
        print(f"  {key:>12}: agree {report[key]['agree_rate']:.1%} (n={n})")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pool_direction_agreement.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-model", type=str, default="ViT-L/14",
                    help="与条件编码器同款 CLIP（论文叙事更干净）；求快可用 ViT-B/32")
    ap.add_argument("--demo", type=str, default=None, help="路由单句演示")
    ap.add_argument("--eval-pool", type=str, default=None,
                    help="附加评测正式指令池的方向一致率（如 dataloaders/instructions_a2a_v2.json）")
    ap.add_argument("--margin-tau", type=float, default=0.008)
    ap.add_argument("--min-abs-sim", type=float, default=0.15)
    ap.add_argument("--no-lexical", action="store_true", help="消融词法层（纯 CLIP 锚点）")
    ap.add_argument("--out-dir", type=str, default="results/instruction_router")
    args = ap.parse_args()

    router = InstructionRouter(args.clip_model, margin_tau=args.margin_tau,
                               min_abs_sim=args.min_abs_sim, use_lexical=not args.no_lexical)
    if args.demo:
        pid, info = router.route(args.demo)
        name = MODALITY_DISPLAY.get(pid, "ABSTAIN（置信度不足，回退询问/默认模态）")
        print(f"{args.demo!r} -> {name}  layer={info['layer']}  "
              f"sims={ {k: round(v, 3) for k, v in (info['sims'] or {}).items()} }")
        return
    run_eval(router, args.out_dir)
    if args.eval_pool:
        run_pool_eval(router, args.eval_pool, args.out_dir)


if __name__ == "__main__":
    main()
