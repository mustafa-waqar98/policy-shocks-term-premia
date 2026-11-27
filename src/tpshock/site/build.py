"""Builds the static dashboard in site/ from data/processed/. Deployed with GitHub Pages.

The page is organised as an argument, not a catalogue. Each section's title is its takeaway, and
its main chart is the evidence for that one claim:
  1. Guidance surprises turn into term premium, but only after the meeting (and only since 2008)
  2. It lands in real rates, far out the curve
  3. It survives the obvious objections (forest plot)
  4. At a single meeting it's a tilt, not a sure thing (hit rates, live scorecard)
  5. Is it Treasury supply? (refunding announcements, supply context)
then the tools: meeting explorer, today's curve, every response, scenarios, validation.

Chart conventions (see theme.py for colours): the story series is blue, a second identity aqua,
context grey; 2px lines, 90% bands as light washes, solid hairline grids, direct labels on the
series that matter, a legend whenever there are two or more series, and a table view under
every headline chart.
"""

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

from tpshock import config
from tpshock.site import theme

P = config.PROCESSED
REF = config.ROOT / "data" / "reference"
TEMPLATES = Path(__file__).parent / "templates"
T = theme.LIGHT
Z90 = 1.645

LABELS = {"y24": "2y yield", "y120": "10y yield", "rn120": "10y expected rates", "tp120": "10y term premium",
          "DFII10": "10y real yield", "T10YIE": "10y breakeven", "BAA10Y": "Baa credit spread",
          "DTWEXBGS": "Broad dollar", "fwd_nominal": "5y5y forward", "fwd_expected": "5y5y: expected rates",
          "fwd_tp": "5y5y: term premium", "fwd_real": "5y5y real forward (TIPS)",
          "fwd_breakeven": "5y5y breakeven"}
UNITS = {"DTWEXBGS": "%"}
SHOCK_NAMES = {"MP_median": "Monetary policy", "CBI_median": "Information", "target": "Target", "path": "Path"}
EVENTS = [("1979-10-06", "Volcker shock"), ("2008-11-25", "QE1"), ("2013-05-22", "Taper tantrum"),
          ("2020-03-15", "Covid: rates to zero"), ("2022-03-16", "Hikes start"),
          ("2026-06-17", "Warsh's first meeting")]
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "zoomIn2d", "zoomOut2d"]}
ROBUST_HEADLINE = "Guidance surprise: 10y term premium after 20 days"


# ---- styling helpers ---------------------------------------------------------------------------

def _style(fig, height=360, hover="x unified", legend=True, **kw):
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=56, r=24, t=40, b=44), font=dict(family=theme.FONT, size=12, color=T["ink2"]),
        showlegend=legend, hovermode=hover, hoverdistance=40, spikedistance=-1,
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left", font=dict(size=12, color=T["ink2"]),
                    itemwidth=30, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=T["surface"], bordercolor=T["grid"],
                        font=dict(family=theme.FONT, size=12, color=T["ink"])),
        **kw)
    fig.update_xaxes(showgrid=False, linecolor=T["axis"], linewidth=1, ticks="outside", tickcolor=T["axis"],
                     ticklen=4, zeroline=False, tickfont=dict(color=T["muted"]),
                     showspikes=hover == "x unified", spikemode="across", spikethickness=1, spikecolor=T["axis"],
                     spikedash="solid", spikesnap="cursor")
    fig.update_yaxes(gridcolor=T["grid"], gridwidth=1, zeroline=False, showline=False, ticks="",
                     tickfont=dict(color=T["muted"]))
    return fig


def _html(fig, div_id=None):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG, div_id=div_id)


def _zero(fig, **rc):
    fig.add_hline(y=0, line=dict(color=T["axis"], width=1), layer="below", **rc)


def _band(fig, x, lo, hi, colour, **rc):
    fig.add_scatter(x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", mode="lines",
                    fillcolor=colour, line=dict(width=0), showlegend=False, hoverinfo="skip", **rc)


def _label(fig, x, y, text, colour=None, xshift=8, size=12, bold=False, **kw):
    t = f"<b>{text}</b>" if bold else text
    fig.add_annotation(x=x, y=y, text=t, showarrow=False, xanchor="left", xshift=xshift,
                       font=dict(size=size, color=colour or T["ink2"]), **kw)


def _chg(s, k):
    return (s.iloc[-1] - s.iloc[-1 - k]) * 100


def _bp(x, digits=1):
    return f"{x:+.{digits}f}bp"


def _table(header, rows, cls=""):
    """Small HTML table for the 'table view' under a chart. Cells are escaped."""
    th = "".join(f'<th class="{"l" if i == 0 else ""}">{escape(str(h))}</th>' for i, h in enumerate(header))
    body = "".join("<tr>" + "".join(f'<td class="{"l" if i == 0 else ""}">{escape(str(c))}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table class="{cls}"><tr>{th}</tr>{body}</table>'


# ---- words ---------------------------------------------------------------------------------------

def todays_read(daily, fomc_days):
    y, rn, tp = daily.y120, daily.rn120, daily.tp120
    d_y, d_rn, d_tp = _chg(y, 21), _chg(rn, 21), _chg(tp, 21)
    driver = "the expected path of policy rates" if abs(d_rn) > abs(d_tp) else "the term premium"
    since_1990 = tp.loc["1990":]
    pct = int(round((since_1990 < tp.iloc[-1]).mean() * 100))
    pos = daily.index.searchsorted(fomc_days.index[-1].normalize())
    f_y = (y.iloc[-1] - y.iloc[pos - 1]) * 100
    f_tp = (tp.iloc[-1] - tp.iloc[pos - 1]) * 100
    return [
        f"The 10-year yield is {y.iloc[-1]:.2f}%: {rn.iloc[-1]:.2f}% is the average short rate the market expects "
        f"over the next ten years and {tp.iloc[-1]:.2f}% is term premium.",
        f"Over the past month the 10y moved {_bp(d_y)}. Expected rates contributed {_bp(d_rn)} and the term premium "
        f"{_bp(d_tp)}, so the move came mainly from {driver}.",
        f"The 10y term premium is higher than on {pct}% of days since 1990.",
        f"Since the last FOMC meeting in the shock data ({fomc_days.index[-1]:%d %b %Y}) the 10y is {_bp(f_y)}, "
        f"of which {_bp(f_tp)} is term premium.",
    ]


def _irf(eras, era, shock, outcome):
    d = eras[(eras.era == era) & (eras.shock == shock) & (eras.outcome == outcome)].sort_values("h")
    return d.set_index("h")


def headline(eras):
    post = _irf(eras, "2008-present", "path", "tp120")
    pre = _irf(eras, "1990-2007", "path", "tp120")
    peak_h = int(post.coef.loc[:30].idxmax())
    return {"h20": post.coef.loc[20], "h20_t": post.coef.loc[20] / post.se.loc[20], "h0": post.coef.loc[0],
            "pre20": pre.coef.loc[20], "h60": post.coef.loc[60], "peak_h": peak_h,
            "n_post": int(post.n.loc[20]), "n_pre": int(pre.n.loc[20])}


# ---- sparkline (inline SVG, no JS) --------------------------------------------------------------

def sparkline(s, w=120, h=28):
    s = s.iloc[-252:].iloc[::5]
    v = s.to_numpy()
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo or 1
    xs = np.linspace(2, w - 4, len(v))
    ys = h - 3 - (v - lo) / span * (h - 6)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
            f'<polyline points="{pts}" fill="none" stroke="var(--muted)" stroke-width="1.5" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3" fill="var(--accent)" stroke="var(--surface)" '
            f'stroke-width="1.5"/></svg>')


# ---- 1. the headline: guidance surprises build into term premium, since 2008 -------------------

def fig_hero(eras, hl):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[f"Before 2008  ·  {hl['n_pre']} meetings",
                                        f"Since 2008  ·  {hl['n_post']} meetings"])
    for c, era in enumerate(["1990-2007", "2008-present"], start=1):
        tp, rn = _irf(eras, era, "path", "tp120"), _irf(eras, era, "path", "rn120")
        _band(fig, tp.index, tp.coef - Z90 * tp.se, tp.coef + Z90 * tp.se, T["accent_band"], row=1, col=c)
        fig.add_scatter(x=rn.index, y=rn.coef, name="Expected rates", legendgroup="rn", showlegend=c == 1,
                        line=dict(color=T["muted"], width=2), hovertemplate="%{y:+.1f}bp", row=1, col=c)
        fig.add_scatter(x=tp.index, y=tp.coef, name="Term premium", legendgroup="tp", showlegend=c == 1,
                        line=dict(color=T["accent"], width=2.5), hovertemplate="%{y:+.1f}bp", row=1, col=c)
        _zero(fig, row=1, col=c)
        fig.add_vline(x=20, line=dict(color=T["grid"], width=1), layer="below", row=1, col=c)
        if era == "2008-present":
            fig.add_scatter(x=[20], y=[tp.coef.loc[20]], mode="markers", showlegend=False, hoverinfo="skip",
                            marker=dict(size=10, color=T["accent"], line=dict(color=T["surface"], width=2)),
                            row=1, col=c)
            fig.add_annotation(x=20, y=tp.coef.loc[20], xref="x2", yref="y2", ax=46, ay=-30, showarrow=True,
                               arrowhead=0, arrowwidth=1, arrowcolor=T["muted"], xanchor="left",
                               text=f"<b>{tp.coef.loc[20]:+.0f}bp</b> after 20 days",
                               font=dict(color=T["ink"], size=12))
            fig.add_annotation(x=0, y=tp.coef.loc[0], xref="x2", yref="y2", ax=26, ay=34, showarrow=True,
                               arrowhead=0, arrowwidth=1, arrowcolor=T["muted"], xanchor="left",
                               text=f"{tp.coef.loc[0]:+.1f}bp on the day", font=dict(color=T["ink2"], size=11))
        else:
            fig.add_annotation(x=30, y=tp.coef.loc[20], xref="x", yref="y", yshift=34, showarrow=False,
                               text="No build-up", font=dict(color=T["muted"], size=12))
    fig = _style(fig, 400)
    fig.update_annotations(selector=dict(text=f"Before 2008  ·  {hl['n_pre']} meetings"),
                           font=dict(size=13, color=T["ink"]), xanchor="left", x=0, yshift=16)
    fig.update_annotations(selector=dict(text=f"Since 2008  ·  {hl['n_post']} meetings"),
                           font=dict(size=13, color=T["ink"]), xanchor="left", x=0.525, yshift=16)
    fig.update_xaxes(tickvals=[0, 10, 20, 30, 40, 50, 60], range=[-1, 61])
    fig.update_xaxes(title=dict(text="trading days after the meeting", font=dict(size=11, color=T["muted"])))
    fig.update_yaxes(ticksuffix="bp", row=1, col=1)
    fig.update_layout(margin=dict(t=84), legend=dict(y=1.2))
    rows = []
    for era in ["1990-2007", "2008-present"]:
        tp, rn = _irf(eras, era, "path", "tp120"), _irf(eras, era, "path", "rn120")
        for h in [0, 5, 10, 20, 40, 60]:
            rows.append([era, h, f"{tp.coef.loc[h]:+.1f}", f"{tp.coef.loc[h] - Z90 * tp.se.loc[h]:+.1f} to "
                         f"{tp.coef.loc[h] + Z90 * tp.se.loc[h]:+.1f}", f"{rn.coef.loc[h]:+.1f}"])
    table = _table(["Period", "Trading days after", "Term premium (bp)", "90% interval", "Expected rates (bp)"], rows)
    return _html(fig, "fig-hero"), table


# ---- 2. real or inflation, far out the curve ---------------------------------------------------

def fig_real(eras):
    real, be = _irf(eras, "2008-present", "path", "fwd_real"), _irf(eras, "2008-present", "path", "fwd_breakeven")
    if real.empty:
        return None, None
    fig = go.Figure()
    _band(fig, real.index, real.coef - Z90 * real.se, real.coef + Z90 * real.se, T["accent_band"])
    _band(fig, be.index, be.coef - Z90 * be.se, be.coef + Z90 * be.se, T["second_band"])
    fig.add_scatter(x=be.index, y=be.coef, name="Breakeven inflation", line=dict(color=T["second"], width=2),
                    hovertemplate="%{y:+.1f}bp")
    fig.add_scatter(x=real.index, y=real.coef, name="Real rate (TIPS)", line=dict(color=T["accent"], width=2.5),
                    hovertemplate="%{y:+.1f}bp")
    _zero(fig)
    fig.add_vline(x=20, line=dict(color=T["grid"], width=1), layer="below")
    for s, name, col in [(real, "Real rate", T["ink"]), (be, "Breakeven", T["ink2"])]:
        fig.add_scatter(x=[20], y=[s.coef.loc[20]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=9, color=T["accent"] if s is real else T["second"],
                                    line=dict(color=T["surface"], width=2)))
        fig.add_annotation(x=20, y=s.coef.loc[20], ax=40, ay=-26 if s is real else 22, showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor=T["muted"], xanchor="left", font=dict(color=col, size=12),
                           text=f"<b>{name} {s.coef.loc[20]:+.0f}bp</b>" if s is real else
                           f"{name} {s.coef.loc[20]:+.0f}bp")
    fig = _style(fig, 340)
    fig.update_xaxes(tickvals=[0, 10, 20, 30, 40, 50, 60], range=[-1, 61],
                     title=dict(text="trading days after the meeting", font=dict(size=11, color=T["muted"])))
    fig.update_yaxes(ticksuffix="bp")
    rows = [[h, f"{real.coef.loc[h]:+.1f}", f"{real.coef.loc[h] / real.se.loc[h]:.1f}", f"{be.coef.loc[h]:+.1f}",
             f"{be.coef.loc[h] / be.se.loc[h]:.1f}"] for h in [0, 5, 10, 20, 40, 60]]
    table = _table(["Trading days after", "Real 5y5y (bp)", "t", "Breakeven 5y5y (bp)", "t"], rows)
    return _html(fig, "fig-real"), table


def forward_split(eras):
    """Numbers for the sentence under the real-rate chart: the model's split of the 5y5y forward."""
    out = {}
    for k in ["fwd_expected", "fwd_tp", "fwd_real", "fwd_breakeven", "fwd_nominal"]:
        d = _irf(eras, "2008-present", "path", k)
        if not d.empty:
            out[k] = {"h0": d.coef.loc[0], "h20": d.coef.loc[20], "t20": d.coef.loc[20] / d.se.loc[20]}
    return out


# ---- 3. forest plot of robustness checks --------------------------------------------------------

FOREST_ROWS = [
    ("Since 2008, every meeting", ROBUST_HEADLINE, "2008-present", "head"),
    ("Without the 35 QE announcement meetings", ROBUST_HEADLINE, "2008-present, excl. balance sheet announcements", ""),
    ("Without 2020", ROBUST_HEADLINE, "2008-present, excl. 2020", ""),
    ("Scheduled meetings only", ROBUST_HEADLINE, "2008-present, scheduled only", ""),
    ("Only surprises where stocks fell", ROBUST_HEADLINE, "2008-present, guidance where stocks fell", ""),
    ("Bias-corrected term premium model", ROBUST_HEADLINE + " (bias-corrected model)", "2008-present", ""),
    ("Before 2008, for contrast", ROBUST_HEADLINE, "1990-2007", "contrast"),
]


def fig_forest(rb):
    cell = {(r.result, r.check): r for r in rb.itertuples()}
    rows = [(lab, cell[(res, chk)], kind) for lab, res, chk, kind in FOREST_ROWS if (res, chk) in cell]
    if not rows:
        return None, None
    labels = [lab for lab, _, _ in rows]
    head = next(r for _, r, k in rows if k == "head")
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=T["axis"], width=1), layer="below")
    fig.add_vline(x=head.coef, line=dict(color=T["accent_soft"], width=1), layer="below")
    for lab, r, kind in rows:
        colour = T["muted"] if kind == "contrast" else T["accent"]
        fig.add_scatter(x=[r.coef], y=[lab], mode="markers", showlegend=False,
                        error_x=dict(type="data", array=[Z90 * r.se], color=colour, thickness=2, width=0),
                        marker=dict(size=11 if kind == "head" else 9, color=colour,
                                    line=dict(color=T["surface"], width=2)),
                        hovertemplate=f"{escape(lab)}<br>%{{x:+.1f}}bp, 90% interval "
                                      f"{r.coef - Z90 * r.se:+.1f} to {r.coef + Z90 * r.se:+.1f}<br>"
                                      f"t = {r.t:.1f}, {r.meetings} meetings<extra></extra>")
        fig.add_annotation(x=1.0, xref="paper", y=lab, xanchor="left", xshift=14, showarrow=False, align="left",
                           text=(f"<b>{r.coef:+.1f}</b>" if kind == "head" else f"{r.coef:+.1f}") +
                           f"  <span style='color:{T['muted']}'>t {r.t:.1f} · {r.meetings} mtgs</span>",
                           font=dict(size=12, color=T["ink"] if kind != "contrast" else T["ink2"]))
    fig = _style(fig, 46 * len(rows) + 70, hover="closest", legend=False)
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=T["ink"], size=12))
    fig.update_xaxes(ticksuffix="bp", showgrid=True, gridcolor=T["grid"], zeroline=False,
                     title=dict(text="10y term premium, 20 trading days after a 10bp guidance surprise",
                                font=dict(size=11, color=T["muted"])))
    fig.update_layout(margin=dict(l=250, r=190, t=16, b=52))
    table = _table(["Check", "bp per 10bp surprise", "90% interval", "t", "Meetings"],
                   [[lab, f"{r.coef:+.1f}", f"{r.coef - Z90 * r.se:+.1f} to {r.coef + Z90 * r.se:+.1f}",
                     f"{r.t:.1f}", r.meetings] for lab, r, _ in rows])
    return _html(fig, "fig-forest"), table


# ---- 4. single meetings: hit rates against a no-meeting base rate ------------------------------

HIT_ROWS = [
    ("Real 5y5y moves with the day-0 2y (any nonzero move)", "Any 2y move", ""),
    ("Real 5y5y moves with the day-0 2y (2y moved at least 2bp)", "2y moved 2bp or more", ""),
    ("Real 5y5y moves with the day-0 2y (2y moved at least 3bp)", "3bp or more: the call", "head"),
    ("Real 5y5y moves with the day-0 2y (2y moved at least 4bp)", "4bp or more", ""),
    ("Real 5y5y moves with the day-0 2y (2y moved at least 5bp)", "5bp or more", ""),
    ("10y term premium moves with the day-0 2y (2y moved at least 3bp)", "10y term premium instead (3bp+)", "contrast"),
    ("Real 5y5y moves more than breakeven 5y5y", "Real beats breakeven: a free roll", "contrast"),
]


def fig_hits(single):
    by = {r["test"]: r for r in single}
    rows = [(lab, by[t], kind) for t, lab, kind in HIT_ROWS if t in by]
    labels = [lab for lab, _, _ in rows]
    fig = go.Figure()
    fig.add_vline(x=50, line=dict(color=T["axis"], width=1), layer="below")
    for lab, r, _kind in rows:
        fig.add_scatter(x=[100 * r["control_rate"], 100 * r["fomc_rate"]], y=[lab, lab], mode="lines",
                        line=dict(color=T["context_soft"], width=4), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[100 * r["control_rate"] for _, r, _ in rows], y=labels, mode="markers",
                    name="20-day windows with no meeting", marker=dict(size=11, color=T["muted"],
                                                                         line=dict(color=T["surface"], width=2)),
                    hovertemplate="No-meeting windows: %{x:.0f}%<extra></extra>")
    fig.add_scatter(x=[100 * r["fomc_rate"] for _, r, _ in rows], y=labels, mode="markers",
                    name="After FOMC meetings", marker=dict(size=11, color=T["accent"],
                                                             line=dict(color=T["surface"], width=2)),
                    hovertemplate="After meetings: %{x:.0f}%<extra></extra>")
    for lab, r, kind in rows:
        p = r["p_two_sided"]
        rates = f"{100 * r['fomc_rate']:.0f}%"
        rates = (f"<b>{rates}</b>" if kind == "head" else rates) + f" vs {100 * r['control_rate']:.0f}%"
        pval = "p<0.01" if p < 0.01 else f"p {p:.2f}"
        fig.add_annotation(x=1.0, xref="paper", y=lab, xanchor="left", xshift=14, showarrow=False,
                           text=f"{rates}  <span style='color:{T['muted']}'>{pval} · {r['meetings']} mtgs</span>",
                           font=dict(size=12, color=T["ink"]))
    fig = _style(fig, 46 * len(rows) + 90, hover="closest")
    fig.update_yaxes(categoryorder="array", categoryarray=labels[::-1], gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=T["ink"], size=12))
    fig.update_xaxes(ticksuffix="%", range=[38, 82], showgrid=True, gridcolor=T["grid"],
                     title=dict(text="share of 20-day windows where the call would have been right",
                                font=dict(size=11, color=T["muted"])))
    fig.update_layout(margin=dict(l=250, r=215, t=40, b=52), legend=dict(y=1.02))
    table = _table(["Test", "Meetings", "After FOMC", "No-meeting windows", "p-value (two-sided)"],
                   [[r["test"], r["meetings"], f"{100 * r['fomc_rate']:.0f}%", f"{100 * r['control_rate']:.0f}%",
                     f"{r['p_two_sided']:.3f}"] for r in single])
    return _html(fig, "fig-hits"), table


# ---- 5. supply ---------------------------------------------------------------------------------

def fig_supply(sup):
    s = sup.dropna(subset=["tenyr_eq_private_gdp"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09, subplot_titles=[
        "Duration the private market holds  ·  10-year equivalents, % of GDP",
        "Share of that duration the Fed holds"])
    fig.add_scatter(x=s.index, y=100 * s.tenyr_eq_private_gdp, line=dict(color=T["accent"], width=2),
                    name="Private", showlegend=False, hovertemplate="%{y:.1f}% of GDP", row=1, col=1)
    fig.add_scatter(x=s.index, y=100 * s.fed_share_of_duration, line=dict(color=T["accent"], width=2),
                    name="Fed share", showlegend=False, hovertemplate="%{y:.0f}%", row=1, col=2)
    for c, v, fmt in [(1, 100 * s.tenyr_eq_private_gdp.iloc[-1], "{:.0f}%"), (2, 100 * s.fed_share_of_duration.iloc[-1],
                                                                             "{:.0f}%")]:
        fig.add_scatter(x=[s.index[-1]], y=[v], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=8, color=T["accent"], line=dict(color=T["surface"], width=2)), row=1, col=c)
        fig.add_annotation(x=s.index[-1], y=v, xref=f"x{'' if c == 1 else c}", yref=f"y{'' if c == 1 else c}",
                           text=fmt.format(v), showarrow=False, xanchor="left", xshift=8,
                           font=dict(size=12, color=T["ink"]))
    fig = _style(fig, 300)
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=12, color=T["ink"]), xanchor="left", yshift=14)
    fig.layout.annotations[0].x = 0
    fig.layout.annotations[1].x = 0.545
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(margin=dict(t=64, r=48))
    last = s.iloc[-1]
    table = _table(["Month", "Private duration, % GDP", "Fed share of duration", "Private WAM (years)"],
                   [[d.strftime("%b %Y"), f"{100 * r.tenyr_eq_private_gdp:.1f}%",
                     f"{100 * r.fed_share_of_duration:.0f}%", f"{r.wam_years:.1f}"]
                    for d, r in s[s.index.month == 12].iterrows()] +
                   [[s.index[-1].strftime("%b %Y"), f"{100 * last.tenyr_eq_private_gdp:.1f}%",
                     f"{100 * last.fed_share_of_duration:.0f}%", f"{last.wam_years:.1f}"]])
    return _html(fig, "fig-supply"), table


def fig_refunding(ev):
    """Scatter: announced change in coupon supply vs the 10-year yield on the statement day."""
    e = ev.dropna(subset=["d_tenyr_eq_bn", "d_y"])
    clean, overlap = e[~e.fomc_overlap], e[e.fomc_overlap]
    fig = go.Figure()
    _zero(fig)
    fig.add_vline(x=0, line=dict(color=T["axis"], width=1), layer="below")
    if len(clean) > 2:
        b = np.polyfit(clean.d_tenyr_eq_bn, clean.d_y, 1)
        xs = np.array([clean.d_tenyr_eq_bn.min(), clean.d_tenyr_eq_bn.max()])
        fig.add_scatter(x=xs, y=np.polyval(b, xs), mode="lines", line=dict(color=T["accent_soft"], width=2),
                        showlegend=False, hoverinfo="skip")
    hover = ("%{customdata}<br>coupon supply %{x:+.1f}bn (10-year equivalents)<br>10-year %{y:+.1f}bp"
             "<extra></extra>")
    fig.add_scatter(x=overlap.d_tenyr_eq_bn, y=overlap.d_y, mode="markers", name="FOMC within a day (left out)",
                    customdata=overlap.index.strftime("%d %b %Y"), hovertemplate=hover,
                    marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=T["muted"], width=1.5)))
    fig.add_scatter(x=clean.d_tenyr_eq_bn, y=clean.d_y, mode="markers", name="Refunding statement",
                    customdata=clean.index.strftime("%d %b %Y"), hovertemplate=hover,
                    marker=dict(size=10, color=T["accent"], line=dict(color=T["surface"], width=2)))
    big = clean[clean.d_tenyr_eq_bn.abs() >= 9]
    for d, r in big.iterrows():
        fig.add_annotation(x=r.d_tenyr_eq_bn, y=r.d_y, text=d.strftime("%b %Y"), showarrow=False, xanchor="left",
                           xshift=9, font=dict(size=11, color=T["ink2"]))
    fig = _style(fig, 380, hover="closest")
    fig.update_xaxes(ticksuffix="bn", showgrid=True, gridcolor=T["grid"],
                     title=dict(text="change in announced 3-, 10- and 30-year sizes vs the last refunding, "
                                     "10-year equivalents", font=dict(size=11, color=T["muted"])))
    fig.update_yaxes(ticksuffix="bp", title=dict(text="10-year yield on the day", font=dict(size=11, color=T["muted"])))
    return _html(fig, "fig-refunding")


def refunding_text(ev, res, hl):
    d, w, x = res["day"], res["three_day"], res["day_ex_largest"]
    lo, hi = sorted([res["median_change_bn"] / 10 * d["d_y"]["coef"], res["median_change_bn"] / 10 * x["coef"]])
    takeaway = (
        f"On the statement day the 10-year rises <b>{d['d_y']['coef']:+.1f}bp per $10bn</b> of extra "
        f"10-year-equivalent "
        f"supply (t {d['d_y']['t']:.1f}, {d['n']} statements with no FOMC meeting within a day); "
        f"{x['coef']:+.1f}bp (t {x['t']:.1f}) without {x['dropped']}, the largest change. Counting Monday's borrowing "
        f"estimate as well gives {w['d_y']['coef']:+.1f}bp (t {w['d_y']['t']:.1f}). The right direction, "
        f"but small: the "
        f"median change is ${res['median_change_bn']:.0f}bn, worth about {lo:.1f} to {hi:.1f}bp, against "
        f"<b>{hl['h20']:+.0f}bp</b> of term premium after a 10bp guidance surprise. Over a single day the model can't "
        f"say how much of it is term premium ({d['d_tp']['coef']:+.1f}bp, t {d['d_tp']['t']:.1f}).")
    rows = [[i.strftime("%d %b %Y"), f"{r.s3:.0f} / {r.s10:.0f} / {r.s30:.0f}",
             "" if pd.isna(r.d_tenyr_eq_bn) else f"{r.d_tenyr_eq_bn:+.1f}", f"{r.d_y:+.1f}",
             "yes" if r.fomc_overlap else ""] for i, r in ev.iloc[::-1].iterrows()]
    table = _table(["Statement", "3y / 10y / 30y sizes ($bn)", "Change, 10y eq. ($bn)", "10-year on the day (bp)",
                    "FOMC within a day"], rows)
    return {"headline": "Supply news nudges the long end the right way, but it's small next to guidance",
            "takeaway": takeaway, "table": table}


# ---- reference charts ----------------------------------------------------------------------------

def fig_history(daily):
    fig = go.Figure()
    for col, name, colour, w in [("y120", "10y yield", T["ink"], 1.5), ("rn120", "Expected short rates", T["muted"], 2),
                                 ("tp120", "Term premium", T["accent"], 2)]:
        fig.add_scatter(x=daily.index, y=daily[col], name=name, line=dict(color=colour, width=w),
                        hovertemplate="%{y:.2f}%")
    _zero(fig)
    recent = daily.index[-1] - pd.DateOffset(years=2)
    for k, (d, label) in enumerate(EVENTS):
        late = pd.Timestamp(d) > recent
        fig.add_vline(x=pd.Timestamp(d).value / 1e6, line=dict(color=T["grid"], width=1), layer="below")
        fig.add_annotation(x=d, y=1 if k % 2 == 0 else 0.94, yref="paper", text=label, showarrow=False,
                           xanchor="right" if late else "left", yanchor="top", xshift=-4 if late else 4,
                           font=dict(size=10, color=T["muted"]))
    start = daily.index[-1] - pd.DateOffset(years=20)
    vis = daily.loc[start:, ["y120", "rn120", "tp120"]]
    pad = 0.08 * (vis.max().max() - vis.min().min())
    fig = _style(fig, 420)
    fig.update_yaxes(ticksuffix="%", range=[vis.min().min() - pad, vis.max().max() + pad])
    fig.update_xaxes(range=[start, daily.index[-1]],
                     rangeselector=dict(x=1, xanchor="right", y=1.0, yanchor="bottom", bgcolor=T["page"],
                                        activecolor=T["grid"], bordercolor=T["grid"], borderwidth=1,
                                        font=dict(size=11, color=T["ink2"]), buttons=[
                                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                                            dict(count=20, label="20Y", step="year", stepmode="backward"),
                                            dict(step="all", label="Since 1961")]))
    return fig


def fig_curve_today(daily, lag=63):
    mats = [m for m in config.REPORT_MATURITIES if f"y{m}" in daily]
    yrs = [m / 12 for m in mats]
    now, then = daily.iloc[-1], daily.iloc[-1 - lag]
    fig = go.Figure()
    fig.add_scatter(x=yrs, y=[then[f"y{m}"] for m in mats], name="Yield 3 months ago", mode="lines",
                    line=dict(color=T["axis"], width=2), hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=yrs, y=[now[f"rn{m}"] for m in mats], name="Expected short rates", mode="lines+markers",
                    line=dict(color=T["muted"], width=2), marker=dict(size=8, line=dict(color=T["surface"], width=2)),
                    hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=yrs, y=[now[f"y{m}"] for m in mats], name="Yield today", mode="lines+markers",
                    line=dict(color=T["ink"], width=2), marker=dict(size=8, line=dict(color=T["surface"], width=2)),
                    fill="tonexty", fillcolor=T["accent_band"], hovertemplate="%{y:.2f}%")
    tp10 = now["y120"] - now["rn120"]
    fig.add_annotation(x=yrs[-1], y=(now["y120"] + now["rn120"]) / 2, text=f"term premium<br><b>{tp10:.2f}%</b>",
                       showarrow=False, xanchor="right", xshift=-10, align="right",
                       font=dict(size=11, color=T["accent"]))
    fig = _style(fig, 340)
    fig.update_xaxes(title=dict(text="maturity", font=dict(size=11, color=T["muted"])), tickvals=yrs,
                     ticktext=[f"{y:g}y" for y in yrs])
    fig.update_yaxes(ticksuffix="%")
    return fig


def fig_shocks(shocks):
    s = shocks.loc["1994":]
    x = s.index.normalize()
    width = 1000 * 3600 * 24 * 30
    fig = go.Figure()
    for col, name, colour in [("MP_median", "Monetary policy shock", T["accent"]),
                              ("CBI_median", "Information shock", T["second"])]:
        fig.add_bar(x=x, y=s[col] * 100, name=name, marker=dict(color=colour, line=dict(width=0)), width=width,
                    customdata=[d.strftime("%Y-%m-%d") for d in s.index],
                    hovertemplate="%{x|%d %b %Y}<br>" + name + ": %{y:+.1f}bp<extra></extra>")
    fig = _style(fig, 280, hover="closest", barmode="relative")
    fig.update_yaxes(ticksuffix="bp")
    _zero(fig)
    return fig


def fig_fomc_split(fomc_days, n=24):
    t = fomc_days.dropna(subset=["d_yield"]).tail(n)
    x = t.index.strftime("%b %y")
    fig = go.Figure()
    fig.add_bar(x=x, y=t.d_expectations, name="Expected rates",
                marker=dict(color=T["context_soft"], line=dict(width=0)),
                hovertemplate="Expected rates %{y:+.1f}bp<extra></extra>")
    fig.add_bar(x=x, y=t.d_term_premium, name="Term premium", marker=dict(color=T["accent"], line=dict(width=0)),
                hovertemplate="Term premium %{y:+.1f}bp<extra></extra>")
    fig.add_scatter(x=x, y=t.d_yield, name="10y change", mode="markers",
                    marker=dict(color=T["ink"], size=8, line=dict(color=T["surface"], width=2)),
                    hovertemplate="10y %{y:+.1f}bp<extra></extra>")
    fig = _style(fig, 320, hover="x", barmode="relative", bargap=0.55, barcornerradius=3)
    _zero(fig)
    fig.update_yaxes(ticksuffix="bp")
    fig.update_xaxes(type="category", tickangle=0, nticks=8)
    return fig


def fig_validation(path):
    if not path.exists():
        return None
    j = pd.read_csv(path, index_col=0)
    j.index = pd.PeriodIndex(j.index, freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_scatter(x=j.index, y=j.ACMTP10, name="NY Fed published", line=dict(color=T["context_soft"], width=4),
                    hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=j.index, y=j.tp120, name="This project", line=dict(color=T["accent"], width=1.5),
                    hovertemplate="%{y:.2f}%")
    _zero(fig)
    fig = _style(fig, 300)
    fig.update_yaxes(ticksuffix="%")
    return _html(fig)


def explore_data(eras):
    """Responses for the 'every response' explorer: era -> shock -> outcome -> h, coef, se."""
    hs = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]
    out = {}
    for (era, shock, outcome), g in eras[eras.h.isin(hs)].groupby(["era", "shock", "outcome"]):
        g = g.sort_values("h")
        out.setdefault(era, {}).setdefault(shock, {})[outcome] = {
            "h": g.h.tolist(), "c": g.coef.round(2).tolist(), "s": g.se.round(2).tolist()}
    return out


# ---- page ----------------------------------------------------------------------------------------

def build(out_dir: Path = config.SITE_OUT):
    daily = pd.read_csv(P / "acm_daily.csv", index_col=0, parse_dates=True)
    shocks = pd.read_csv(P / "shocks.csv", index_col=0, parse_dates=True)
    fomc_days = pd.read_csv(P / "fomc_days.csv", index_col=0, parse_dates=True)
    irf = pd.read_csv(P / "irf.csv")
    events = pd.read_csv(P / "event_regressions.csv")
    eras = pd.read_csv(P / "irf_eras.csv")

    def read_json(name, d=P):
        f = d / name
        return json.loads(f.read_text()) if f.exists() else None

    hl = headline(eras)
    tiles = []
    for key, label, tip in [
        ("y120", "10y yield", "Fitted 10-year zero-coupon Treasury yield."),
        ("rn120", "Expected short rates", "Average short-term rate the market expects over the next 10 years."),
        ("tp120", "Term premium", "Extra yield investors demand for holding a 10y bond instead of rolling bills."),
    ]:
        s = daily[key]
        tiles.append({"label": label, "tip": tip, "value": f"{s.iloc[-1]:.2f}%", "d1": _chg(s, 1), "d21": _chg(s, 21),
                      "spark": sparkline(s), "key": key})

    pos = daily.index.searchsorted(fomc_days.index[-1].normalize())
    curve = []
    for m in config.REPORT_MATURITIES:
        if f"y{m}" not in daily:
            continue
        row = {"tenor": f"{m // 12}y"}
        for part in ("y", "rn", "tp"):
            s = daily[f"{part}{m}"]
            row[part] = s.iloc[-1]
            row[part + "_d21"] = _chg(s, 21)
            row[part + "_fomc"] = (s.iloc[-1] - s.iloc[max(pos - 1, 0)]) * 100
        curve.append(row)

    ev = events[events.outcome.isin(["d_yield", "d_expectations", "d_term_premium"])]
    regime_rows = []
    for reg in ["all", "1990-2007", "2008-present"]:
        for s in SHOCK_NAMES:
            sub = ev[(ev.regime == reg) & (ev.shock == s)].set_index("outcome")
            if sub.empty:
                continue
            regime_rows.append({"regime": "Full sample" if reg == "all" else reg, "shock": SHOCK_NAMES[s],
                                **{o: (sub.loc[o, "bp_per_10bp"], sub.loc[o, "se"])
                                   for o in ["d_yield", "d_expectations", "d_term_premium"]}})

    scen = {s: {o: irf[(irf.shock == s) & (irf.outcome == o)].sort_values("h")[["h", "coef"]].to_dict("list")
                for o in irf.outcome.unique()} for s in irf.shock.unique()}

    robust, forest, forest_table = [], None, None
    if (P / "robustness.csv").exists():
        rb = pd.read_csv(P / "robustness.csv")
        forest, forest_table = fig_forest(rb)
        cell = {(r.result, r.check): r for r in rb.itertuples()}
        for title, pick in [("Full sample and by period", lambda c: not c.startswith("2008-present,")),
                            ("Inside the post-2008 sample",
                             lambda c: c == "2008-present" or c.startswith("2008-present,"))]:
            checks = [c for c in dict.fromkeys(rb.check) if pick(c)]
            results = [r for r in dict.fromkeys(rb.result) if any((r, c) in cell for c in checks)]
            robust.append({"title": title,
                           "checks": [c.replace("2008-present, ", "") if c != "2008-present" else "All 2008+"
                                      for c in checks],
                           "rows": [{"result": res, "cells": [cell.get((res, c)) for c in checks]} for res in results]})

    single, hits, hits_table = None, None, None
    if (P / "single_meeting.csv").exists():
        single = pd.read_csv(P / "single_meeting.csv").to_dict("records")
        hits, hits_table = fig_hits(single)

    card, card_tally, card_ref = None, None, None
    if (P / "scorecard.csv").exists():
        card = pd.read_csv(P / "scorecard.csv", keep_default_na=False).to_dict("records")
        card_tally = read_json("scorecard_meta.json")
        if single:
            r3 = {r["test"]: r for r in single}.get("Real 5y5y moves with the day-0 2y (2y moved at least 3bp)")
            if r3:
                card_ref = {"fomc": r3["fomc_rate"], "control": r3["control_rate"], "n": r3["meetings"]}

    hero, hero_table = fig_hero(eras, hl)
    real, real_table = fig_real(eras)
    supply, supply_table = (fig_supply(pd.read_csv(REF / "supply_monthly.csv", index_col=0, parse_dates=True))
                            if (REF / "supply_monthly.csv").exists() else (None, None))
    refunding, refunding_stats = None, None
    if (REF / "refunding_events.csv").exists():
        rev = pd.read_csv(REF / "refunding_events.csv", index_col=0, parse_dates=True)
        res = read_json("refunding_results.json", REF)
        refunding = fig_refunding(rev)
        refunding_stats = refunding_text(rev, res, hl) if res else None

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    html = env.get_template("index.html").render(
        updated=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        last_date=daily.index[-1].strftime("%d %b %Y"), n_meetings=len(fomc_days), hl=hl,
        tiles=tiles, tp_share=100 * daily.tp120.iloc[-1] / daily.y120.iloc[-1], curve=curve,
        read=todays_read(daily, fomc_days),
        fig_hero=hero, hero_table=hero_table, fig_real=real, real_table=real_table, fwd=forward_split(eras),
        fig_forest=forest, forest_table=forest_table, fig_hits=hits, hits_table=hits_table,
        fig_supply=supply, supply_table=supply_table, fig_refunding=refunding, refunding=refunding_stats,
        fig_history=_html(fig_history(daily), "fig-history"), fig_curve=_html(fig_curve_today(daily)),
        fig_shocks=_html(fig_shocks(shocks), "fig-shocks"), fig_split=_html(fig_fomc_split(fomc_days)),
        fig_valid=fig_validation(P / "acm_vs_official.csv"),
        regime_rows=regime_rows, robust=robust, single=single, card=card, card_tally=card_tally,
        card_ref=card_ref, acm_valid=read_json("acm_validation.json"), shock_valid=read_json("shocks_validation.json"),
        labels=[(k, v) for k, v in LABELS.items() if k in set(irf.outcome)], explore_labels=list(LABELS.items()),
        units=UNITS, scen_json=json.dumps(scen), explore_json=json.dumps(explore_data(eras)),
        meetings_json=json.dumps(meeting_explorer(daily, fomc_days)), repo_url=config.REPO_URL,
        light=theme.LIGHT, dark=theme.DARK, swap=theme.SWAP, font=theme.FONT,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    (out_dir / ".nojekyll").write_text("")
    return out_dir / "index.html"


def meeting_explorer(daily, fomc_days, window=(-5, 20)):
    cols = ["y24", "y120", "rn120", "tp120"]
    idx = daily.index
    big = fomc_days.pc1.abs()
    out = []
    for when, r in fomc_days.iterrows():
        pos = idx.searchsorted(when.normalize())
        if pos < 1 - window[0] or pos >= len(idx):
            continue
        lo, hi = pos + window[0], min(pos + window[1], len(idx) - 1)
        base = daily.iloc[pos - 1][cols]
        seg = (daily.iloc[lo:hi + 1][cols] - base) * 100
        mp, info = r.MP_median, r.CBI_median
        rank = float((big < abs(r.pc1)).mean())
        out.append({
            "date": when.strftime("%Y-%m-%d"), "label": when.strftime("%d %b %Y"),
            "pc1": r.pc1 * 100, "mp": mp * 100, "info": info * 100, "target": r.target * 100,
            "path": r.path * 100, "sp500": r.SP500,
            "kind": "Monetary policy" if abs(mp) >= abs(info) else "Information",
            "dir": "hawkish" if r.pc1 > 0 else "dovish", "size_pct": round(rank * 100),
            "h": list(range(lo - pos, hi - pos + 1)),
            **{c: [round(v, 2) for v in seg[c]] for c in cols},
        })
    return out
