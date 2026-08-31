"""
render_airs.py — the dumb formatter.

Reads one `run` dict (matching run.schema.json / run_example.json) and returns a
complete HTML string by replacing the {{TOKENS}} in airs_template.html.

Design rule: the UI computes nothing. Every string, number, and bar width comes
from the run dict. All pipeline text is HTML-escaped through esc() so a stray
`<` or `&` in an LLM-generated idea can't break the page.

Run standalone to preview with the example data:
    python render_airs.py                 # renders run_example.json
    python render_airs.py path/to/run.json
-> writes rendered_preview.html next to this file.
"""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(HERE, "airs_template.html")

CHEV = "&#9656;"          # ▸  (rotates 90deg on open via CSS)
ARROW_UP = "&#9650;"      # ▲
ARROW_DOWN = "&#9660;"    # ▼
DOC_ICON = "&#128196;"    # 📄
ARM_COLORS = {"A": "var(--armA)", "B": "var(--armB)", "C": "var(--armC)"}
ARM_LABELS = {"A": "Arm A", "B": "Arm B", "C": "Arm C"}
JUDGE_METRICS = [
    ("novelty", "Novelty"),
    ("diversity", "Diversity"),
    ("usefulness", "Usefulness"),
    ("assumption_challenge", "Assump. challenge"),
]


def esc(value):
    """HTML-escape any value; None -> empty string."""
    return html.escape(str(value if value is not None else ""))


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value, digits=1):
    """Format a number for display; blank for None/non-numeric."""
    if value is None or value == "":
        return "&mdash;"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


def _pct(value):
    if value is None or value == "":
        return "&mdash;"
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return esc(value)


# ---------------------------------------------------------------- fragments

def _idea_eval_folds(ev):
    """Per-idea judge + literature details -> two nested <details> folds
    that live inside a feature card's expand body (layer 2)."""
    if not ev:
        return ""
    folds = []

    judge = ev.get("judge") or {}
    if judge:
        cells = "".join(
            f'<div class="jm"><span>{label}</span><b>{_fmt(judge.get(key))}</b></div>'
            for key, label in JUDGE_METRICS
        )
        rationale = judge.get("rationale", "")
        rat_html = f'<div class="jrat">{esc(rationale)}</div>' if rationale else ""
        folds.append(
            f'<details class="subfold"><summary>Judge scores</summary>'
            f'<div class="jgrid">{cells}</div>{rat_html}</details>'
        )

    lit = ev.get("literature") or {}
    if lit:
        cells = (
            f'<div class="jm"><span>Novelty</span><b>{_fmt(lit.get("novelty"), 2)}</b></div>'
            f'<div class="jm"><span>Evidence</span><b>{esc(lit.get("evidence_count", 0))}</b></div>'
            f'<div class="jm"><span>Ev. ratio</span><b>{_pct(lit.get("evidence_ratio"))}</b></div>'
        )
        closest = lit.get("closest_paper") or {}
        if closest and closest.get("title"):
            url = closest.get("url", "")
            title = esc(closest.get("title"))
            sim = _fmt(closest.get("similarity"), 2)
            link = (
                f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{title}</a>'
                if url else f"<span>{title}</span>"
            )
            closest_html = f'<div class="closest">Closest paper: {link} <span class="sim">sim {sim}</span></div>'
        else:
            closest_html = '<div class="closest none">No grounding paper above threshold &middot; speculative.</div>'
        folds.append(
            f'<details class="subfold"><summary>Literature grounding</summary>'
            f'<div class="jgrid lit">{cells}</div>{closest_html}</details>'
        )

    if not folds:
        return ""
    return f'<div class="idfolds">{"".join(folds)}</div>'


def _idea_tags(ev):
    """Top-3 badges shown on the card itself (debate / pairwise usefulness)."""
    if not ev:
        return ""
    tags = []
    if ev.get("debated"):
        tags.append('<span class="itag debate">Debated</span>')
    rank = ev.get("usefulness_rank")
    if rank in (1, 2, 3):
        tags.append(f'<span class="itag useful">#{rank} usefulness</span>')
    return "".join(tags)


def _feature_cards(directions):
    """Arm C directions -> the featured lens cards."""
    out = []
    for d in directions:
        evidence = d.get("evidence") or {}
        is_spec = evidence.get("type") == "speculative"
        lens_cls = "lens spec" if is_spec else "lens"

        # evidence row: speculative tag OR a citation link
        if is_spec:
            cite = '<span class="spectag">Speculative extension</span>'
        else:
            url = evidence.get("url", "")
            citation = esc(evidence.get("citation", "Source"))
            if url:
                cite = (
                    f'<span>{DOC_ICON}</span>'
                    f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{citation}</a>'
                )
            else:
                cite = f'<span>{DOC_ICON}</span><span>{citation}</span>'

        breaks = d.get("breaks", "")
        breaks_html = (
            f'<div class="breaks"><b>Breaks:</b> {esc(breaks)}</div>' if breaks else ""
        )

        lens = d.get("lens", "")
        lens_html = f'<span class="{lens_cls}">{esc(lens)}</span>' if lens else ""
        tags_html = _idea_tags(d.get("eval"))
        chips_html = (
            f'<div class="chips">{lens_html}{tags_html}</div>'
            if (lens_html or tags_html) else ""
        )
        oneline = d.get("oneline", "")
        oneline_html = f'<div class="oneline">{esc(oneline)}</div>' if oneline else ""

        folds_html = _idea_eval_folds(d.get("eval"))

        # Expandable when the detail adds something beyond the one-liner OR when
        # there are per-idea evaluation folds to show.
        detail = d.get("detail", "")
        detail_shows = bool(detail.strip()) and detail.strip() != oneline.strip()
        expandable = detail_shows or bool(folds_html)
        card_cls = "lcard" if expandable else "lcard static"
        chev_html = f'<span class="chev">{CHEV}</span>' if expandable else ""
        detail_html = f'<div class="dtl">{esc(detail)}</div>' if detail_shows else ""
        body_html = (
            f'<div class="body"><div class="inner">{detail_html}{folds_html}</div></div>'
            if expandable else ""
        )

        out.append(
            f'''        <div class="{card_cls}">
          <div class="lctop">
            <h4>{esc(d.get("title"))}</h4>
            {chev_html}
          </div>
          {chips_html}
          {oneline_html}
          {body_html}
          {breaks_html}
          <div class="cite">{cite}</div>
        </div>'''
        )
    return "\n".join(out)


def _baseline_cards(ideas):
    """Arm A / Arm B ideas -> flat title + detail cards."""
    out = []
    for it in ideas:
        title = it.get("title", "")
        detail = it.get("detail", "")
        # Expandable only when the detail says more than the title already does.
        expandable = bool(detail.strip()) and detail.strip() != title.strip()
        card_cls = "scard" if expandable else "scard static"
        chev_html = f'<span class="chev">{CHEV}</span>' if expandable else ""
        body_html = (
            f'<div class="body"><div class="inner">{esc(detail)}</div></div>'
            if expandable else ""
        )
        out.append(
            f'''            <div class="{card_cls}">
              <div class="st">
                <h4>{esc(title)}</h4>
                {chev_html}
              </div>
              {body_html}
            </div>'''
        )
    return "\n".join(out)


def _assumptions(items):
    """assumptions[] -> the Classroom 'assumptions the lenses broke' panel."""
    out = []
    for a in items:
        out.append(
            f'''        <div class="assump">
          <div class="was">Assumed</div>
          <div class="claim">{esc(a.get("assumed"))}</div>
          <div class="arrow"><span class="a">&#8594;</span><span class="broke">{esc(a.get("reframe"))}</span></div>
          <div class="who">{esc(a.get("lens", ""))}</div>
        </div>'''
        )
    return "\n".join(out)


def _vendi_bars(vendi):
    """metrics.vendi -> the three bars, widths scaled to the max value in this run."""
    rows = [("A", "Arm A", "fillA", ""),
            ("B", "Arm B", "fillB", ""),
            ("C", "Arm C", "fillC", "winrow")]
    values = [_num(vendi.get(k)) for k, *_ in rows]
    max_val = max(values) or 1.0

    out = []
    for key, label, fill, row_cls in rows:
        value = vendi.get(key, 0)
        width = round(_num(value) / max_val * 96)
        out.append(
            f'''        <div class="bar {row_cls}">
          <span class="bl">{label}</span>
          <div class="track"><div class="fill {fill}" data-w="{width}"></div></div>
          <span class="bv">{esc(value)}</span>
        </div>'''
        )
    return "\n".join(out)


def _crosschecks(items):
    """metrics.crosschecks -> the MPD / Distinct-2 / Self-BLEU lines."""
    out = []
    for c in items:
        better = c.get("better", "higher")
        arrow = ARROW_DOWN if better == "lower" else ARROW_UP
        out.append(
            f'''        <div class="cc">{esc(c.get("name"))} &middot; '''
            f'''C <b>{esc(c.get("c"))}</b> vs B <b>{esc(c.get("b"))}</b> '''
            f'''<span class="up">{arrow} {esc(better)} = better</span></div>'''
        )
    return "\n".join(out)


def _eval_score_table(summary, metrics, digits=1, as_pct=False):
    """Per-arm score table. `summary` = {A:{...},B:{...},C:{...}};
    `metrics` = list of (key, label)."""
    if not summary:
        return ""
    head = "".join(f"<th>{label}</th>" for _, label in metrics)
    rows = []
    for key in ("A", "B", "C"):
        arm = summary.get(key) or {}
        cells = []
        for mkey, _ in metrics:
            v = arm.get(mkey)
            cells.append(f"<td>{_pct(v) if as_pct else _fmt(v, digits)}</td>")
        is_c = " isc" if key == "C" else ""
        rows.append(
            f'<tr class="{is_c.strip()}"><th class="arm"><span class="sw sw{key}"></span>'
            f'{ARM_LABELS[key]}</th>{"".join(cells)}</tr>'
        )
    return (
        f'<table class="evt"><thead><tr><th class="arm"></th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _debate_block(debate):
    """AC debate audit -> expandable rows for the debated (top-3) ideas."""
    items = (debate or {}).get("top", []) or []
    if not items:
        return ""
    rows = []
    for it in items:
        # debate_decision is freeform LLM text; colour by its first word when it
        # matches a known verdict, otherwise stay neutral. Always shown verbatim.
        raw = str(it.get("decision") or "").strip()
        first = raw.split()[0].lower().strip(".,:;") if raw else ""
        if first in ("raise", "upgrade", "uphold", "keep", "support", "strengthen"):
            dec_cls = "dec-raise"
        elif first in ("lower", "downgrade", "revise", "reject", "weaken"):
            dec_cls = "dec-lower"
        elif first in ("hold", "split", "mixed", "uncertain", "unchanged"):
            dec_cls = "dec-hold"
        else:
            dec_cls = "dec-neutral"
        label = raw if len(raw) <= 28 else raw[:27].rstrip() + "…"
        conf = it.get("confidence")
        conf_html = f'<span class="conf">confidence {_fmt(conf, 2)}</span>' if conf is not None else ""
        score_html = (
            f'<span class="delta">{_fmt(it.get("llm_score"))} '
            f'&rarr; <b>{_fmt(it.get("final_score"))}</b></span>'
        )
        rows.append(
            f'''<details class="drow">
  <summary>
    <span class="dtitle">{esc(it.get("idea_title"))}</span>
    <span class="ddec {dec_cls}">{esc(label) or "reviewed"}</span>
    {score_html}
  </summary>
  <div class="dbody">
    <div class="darg adv"><span class="dlabel">Advocate</span>{esc(it.get("advocate_argument"))}</div>
    <div class="darg skp"><span class="dlabel">Skeptic</span>{esc(it.get("skeptic_argument"))}</div>
    <div class="dmeta">{conf_html}</div>
  </div>
</details>'''
        )
    return (
        '<div class="detblock"><div class="detbh">Assumption-challenge debate</div>'
        f'{"".join(rows)}</div>'
    )


def _pairwise_block(pairwise):
    """Pairwise usefulness tournament -> top-3 ranking + sample comparisons."""
    pw = pairwise or {}
    top = pw.get("top", []) or []
    comps = pw.get("comparisons", []) or []
    if not top and not comps:
        return ""

    rank_rows = []
    for it in top:
        wl = f'{esc(it.get("wins", 0))}&ndash;{esc(it.get("losses", 0))}'
        if it.get("ties"):
            wl += f'&ndash;{esc(it.get("ties"))}'
        rank_rows.append(
            f'<div class="pwrow"><span class="pwrank">#{esc(it.get("rank", ""))}</span>'
            f'<span class="pwtitle">{esc(it.get("idea_title"))}</span>'
            f'<span class="pwwr">{_pct(it.get("win_rate"))} win rate</span>'
            f'<span class="pwrec">{wl}</span></div>'
        )
    rank_html = f'<div class="pwrank-list">{"".join(rank_rows)}</div>' if rank_rows else ""

    comp_rows = []
    for c in comps:
        comp_rows.append(
            f'<div class="pwcomp"><b>{esc(c.get("a"))}</b> '
            f'<span class="beat">beat</span> {esc(c.get("b"))}'
            f'<div class="pwreason">{esc(c.get("reason"))}</div></div>'
        )
    comp_html = (
        f'<details class="subfold wide"><summary>Sample head-to-head comparisons</summary>'
        f'<div class="pwcomps">{"".join(comp_rows)}</div></details>'
        if comp_rows else ""
    )

    return (
        '<div class="detblock"><div class="detbh">Pairwise usefulness tournament</div>'
        f'{rank_html}{comp_html}</div>'
    )


def _gen_context(gc, run):
    """generation_context + run-level metadata -> the bottom info panel."""
    gc = gc or {}
    params = gc.get("params") or {}
    lit = gc.get("literature") or {}

    def row(label, value):
        return f'<div class="gcrow"><span class="gck">{label}</span><span class="gcv">{value}</span></div>'

    rows = [
        row("Topic", esc(run.get("topic", ""))),
        row("Generated", esc(run.get("generated_at", "")) or "&mdash;"),
    ]
    model = gc.get("model")
    backend = gc.get("backend")
    if model or backend:
        mv = esc(model or "")
        if backend:
            mv += f' <span class="gcsub">({esc(backend)})</span>'
        rows.append(row("Model", mv))
    if params:
        parts = []
        if "paper_limit" in params:
            parts.append(f'paper limit {esc(params.get("paper_limit"))}')
        if "use_llm_queries" in params:
            parts.append(f'LLM queries {"on" if params.get("use_llm_queries") else "off"}')
        if "run_debate" in params:
            parts.append(f'debate {"on" if params.get("run_debate") else "off"}')
        rows.append(row("Run parameters", esc(" &middot; ".join(parts)).replace("&amp;middot;", "&middot;")))
    if lit:
        qsrc = esc(lit.get("query_source", ""))
        pc = esc(lit.get("paper_count", ""))
        rows.append(row("Literature", f'{pc} papers &middot; queries: {qsrc}'))
        queries = lit.get("queries") or []
        if queries:
            qhtml = "".join(f'<span class="gcq">{esc(q)}</span>' for q in queries)
            rows.append(row("Queries", f'<div class="gcqs">{qhtml}</div>'))
    counts = gc.get("idea_counts") or {}
    if counts:
        cparts = " &middot; ".join(f'{k}: {esc(v)}' for k, v in counts.items())
        rows.append(row("Ideas per arm", cparts))

    arms = gc.get("arms") or []
    arm_html = ""
    if arms:
        arm_rows = "".join(
            f'<div class="gcarm"><span class="sw sw{esc(a.get("key"))}"></span>'
            f'<b>{esc(a.get("label"))}</b>'
            + (f'<span class="gctemp">temp {esc(a.get("temperature"))}</span>' if a.get("temperature") is not None else "")
            + '</div>'
            for a in arms
        )
        arm_html = f'<div class="gcarms">{arm_rows}</div>'

    return f'<div class="gcgrid">{"".join(rows)}</div>{arm_html}'


def _human_rows(ratings, vendi):
    """human_eval.ratings + metrics.vendi -> the humans-vs-metric table."""
    vendi_values = [_num(vendi.get(k)) for k in ("A", "B", "C")]
    vendi_max = max(vendi_values) or 1.0

    out = []
    for key in ("A", "B", "C"):
        rating = ratings.get(key, 0)
        vendi_val = vendi.get(key, 0)
        rating_w = round(_num(rating) / 5.0 * 100)          # class ratings are 1-5
        vendi_w = round(_num(vendi_val) / vendi_max * 100)
        color = ARM_COLORS[key]
        out.append(
            f'''        <div class="hv-row">
          <span class="bl">Arm {key}</span>
          <div style="display:flex;align-items:center;gap:8px">
            <div class="hv-bar" style="flex:1"><i style="width:{rating_w}%;background:{color}"></i></div>
            <span style="font-size:11px;color:var(--sub);font-variant-numeric:tabular-nums">{esc(rating)}</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <div class="hv-bar" style="flex:1"><i style="width:{vendi_w}%;background:{color}"></i></div>
            <span style="font-size:11px;color:var(--sub);font-variant-numeric:tabular-nums">{esc(vendi_val)}</span>
          </div>
        </div>'''
        )
    return "\n".join(out)


# ---------------------------------------------------------------- render

def render(run):
    """Return the full HTML page for a run dict."""
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    arms = run.get("arms", {}) or {}
    arm_a = arms.get("A", {}) or {}
    arm_b = arms.get("B", {}) or {}
    arm_c = arms.get("C", {}) or {}
    directions = arm_c.get("directions", []) or []

    metrics = run.get("metrics", {}) or {}
    vendi = metrics.get("vendi", {}) or {}
    human = run.get("human_eval", {}) or {}

    evaluation = run.get("evaluation", {}) or {}
    judge_summary = evaluation.get("judge_summary", {}) or {}
    literature_summary = evaluation.get("literature_summary", {}) or {}
    lit_metrics = [
        ("avg_novelty", "Avg novelty"),
        ("avg_evidence_count", "Avg evidence"),
        ("avg_evidence_ratio", "Evidence ratio"),
    ]

    rationale = run.get("discussion_rationale", "")
    rationale_html = (
        f'<details class="disc-why"><summary>Why this question</summary>'
        f'<p class="why">{esc(rationale)}</p></details>'
        if rationale else ""
    )

    replacements = {
        "{{TOPIC}}": esc(run.get("topic", "")),
        "{{C_LABEL}}": esc(arm_c.get("label", "Divergent directions")),
        "{{C_SUBLABEL}}": esc(arm_c.get("sublabel", "")),
        "{{C_META}}": f"<b>{len(directions)}</b>divergent directions",
        "{{FEATURE_CARDS}}": _feature_cards(directions),
        "{{DISCUSSION}}": esc(run.get("discussion_prompt", "")),
        "{{DISCUSSION_RATIONALE}}": rationale_html,
        "{{ASSUMPTIONS}}": _assumptions(run.get("assumptions", []) or []),
        "{{A_LABEL}}": esc(arm_a.get("label", "Naive baseline")),
        "{{A_SUB}}": esc(arm_a.get("sublabel", "Arm A")),
        "{{BASELINE_A}}": _baseline_cards(arm_a.get("ideas", []) or []),
        "{{B_LABEL}}": esc(arm_b.get("label", "Strong baseline")),
        "{{B_SUB}}": esc(arm_b.get("sublabel", "Arm B")),
        "{{BASELINE_B}}": _baseline_cards(arm_b.get("ideas", []) or []),
        "{{VENDI_BARS}}": _vendi_bars(vendi),
        "{{CROSSCHECKS}}": _crosschecks(metrics.get("crosschecks", []) or []),
        "{{JUDGE_TABLE}}": _eval_score_table(judge_summary, JUDGE_METRICS, 1),
        "{{LIT_TABLE}}": _eval_score_table(literature_summary, lit_metrics, 2),
        "{{DEBATE_BLOCK}}": _debate_block(evaluation.get("debate")),
        "{{PAIRWISE_BLOCK}}": _pairwise_block(evaluation.get("pairwise")),
        "{{GEN_CONTEXT}}": _gen_context(run.get("generation_context"), run),
        "{{HUMAN_ROWS}}": _human_rows(human.get("ratings", {}) or {}, vendi),
        "{{HUMAN_NOTE}}": esc(human.get("note", "")),
    }

    for token, fragment in replacements.items():
        template = template.replace(token, fragment)
    return template


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "run_example.json")
    with open(src, encoding="utf-8") as f:
        run_data = json.load(f)

    output_html = render(run_data)
    dest = os.path.join(HERE, "rendered_preview.html")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(output_html)
    print(f"wrote {dest}")
