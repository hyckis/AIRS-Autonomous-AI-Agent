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


def esc(value):
    """HTML-escape any value; None -> empty string."""
    return html.escape(str(value if value is not None else ""))


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- fragments

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
        oneline = d.get("oneline", "")
        oneline_html = f'<div class="oneline">{esc(oneline)}</div>' if oneline else ""

        # Only make the card expandable when the detail adds something beyond the
        # always-visible one-liner; otherwise drop the chevron and expand body.
        detail = d.get("detail", "")
        expandable = bool(detail.strip()) and detail.strip() != oneline.strip()
        card_cls = "lcard" if expandable else "lcard static"
        chev_html = f'<span class="chev">{CHEV}</span>' if expandable else ""
        body_html = (
            f'<div class="body"><div class="inner">{esc(detail)}</div></div>'
            if expandable else ""
        )

        out.append(
            f'''        <div class="{card_cls}">
          <div class="lctop">
            <h4>{esc(d.get("title"))}</h4>
            {chev_html}
          </div>
          {lens_html}
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

    replacements = {
        "{{TOPIC}}": esc(run.get("topic", "")),
        "{{C_LABEL}}": esc(arm_c.get("label", "Divergent directions")),
        "{{C_SUBLABEL}}": esc(arm_c.get("sublabel", "")),
        "{{C_META}}": f"<b>{len(directions)}</b>divergent directions",
        "{{FEATURE_CARDS}}": _feature_cards(directions),
        "{{DISCUSSION}}": esc(run.get("discussion_prompt", "")),
        "{{ASSUMPTIONS}}": _assumptions(run.get("assumptions", []) or []),
        "{{A_LABEL}}": esc(arm_a.get("label", "Naive baseline")),
        "{{A_SUB}}": esc(arm_a.get("sublabel", "Arm A")),
        "{{BASELINE_A}}": _baseline_cards(arm_a.get("ideas", []) or []),
        "{{B_LABEL}}": esc(arm_b.get("label", "Strong baseline")),
        "{{B_SUB}}": esc(arm_b.get("sublabel", "Arm B")),
        "{{BASELINE_B}}": _baseline_cards(arm_b.get("ideas", []) or []),
        "{{VENDI_BARS}}": _vendi_bars(vendi),
        "{{CROSSCHECKS}}": _crosschecks(metrics.get("crosschecks", []) or []),
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
