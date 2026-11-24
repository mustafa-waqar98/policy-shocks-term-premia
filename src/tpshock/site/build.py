"""Builds the static dashboard in site/ from data/processed/. Deployed with GitHub Pages."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

from tpshock import config

P = config.PROCESSED
TEMPLATES = Path(__file__).parent / "templates"

# Colour roles. NAVY and SKY are the two categorical slots (validated for colour-blind
# separation); CHARCOAL is the neutral used for totals such as the yield itself.
NAVY, SKY, CHARCOAL = "#184f95", "#3d9be9", "#2b3340"
INK2, MUTED, GRID = "#4b5563", "#8b95a3", "#e8ecf1"
NAVY_BAND, SKY_BAND = "rgba(24,79,149,0.13)", "rgba(61,155,233,0.16)"
FONT = "Figtree, Helvetica, Arial, sans-serif"

LABELS = {"y24": "2y yield", "y120": "10y yield", "rn120": "10y expected rates",
          "tp120": "10y term premium", "DFII10": "10y real yield", "T10YIE": "10y breakeven",
          "BAA10Y": "Baa credit spread", "DTWEXBGS": "Broad dollar"}
UNITS = {"DTWEXBGS": "%"}
SHOCK_NAMES = {"MP_median": "Monetary policy", "CBI_median": "Information",
               "target": "Target", "path": "Path"}
EVENTS = [("1979-10-06", "Volcker shock"), ("2008-11-25", "QE1"), ("2013-05-22", "Taper tantrum"),
          ("2020-03-15", "Covid: rates to zero"), ("2022-03-16", "Hiking cycle starts"),
          ("2023-10-19", "10y touches 5%")]
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _style(fig, height=360, **kw):
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=52, r=16, t=36, b=36), font=dict(family=FONT, size=12, color=INK2),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12, color=INK2)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=CHARCOAL)),
        **kw)
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", ticks="")
    return fig


def _html(fig, div_id=None):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG, div_id=div_id)


def _chg(s, k):
    return (s.iloc[-1] - s.iloc[-1 - k]) * 100


def _bp(x, digits=1):
    return f"{x:+.{digits}f}bp"


# ---- words: the plain-English read of today's data and the main results ----------------------

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
        f"Since the last FOMC meeting ({fomc_days.index[-1]:%d %b %Y}) the 10y is {_bp(f_y)}, of which {_bp(f_tp)} "
        f"is term premium.",
    ]


def key_findings(irf):
    def get(shock, outcome, h):
        r = irf[(irf.shock == shock) & (irf.outcome == outcome) & (irf.h == h)]
        return (float(r.coef.iloc[0]), float(r.se.iloc[0])) if len(r) else (np.nan, np.nan)

    def band(c, se):
        return f"90% interval {c - 1.645 * se:+.1f} to {c + 1.645 * se:+.1f}"

    out = []
    c, se = get("path", "tp120", 20)
    c0, _ = get("path", "tp120", 0)
    out.append(("Guidance builds into the term premium",
                f"A 10bp path (forward guidance) surprise moves the 10y term premium {c0:+.1f}bp on the day and "
                f"{c:+.1f}bp after 20 trading days ({band(c, se)})."))
    c, se = get("target", "tp120", 0)
    r0, _ = get("target", "rn120", 0)
    out.append(("Rate surprises compress it",
                f"A 10bp surprise in today's rate decision raises 10y expected rates {r0:+.1f}bp but moves the term "
                f"premium {c:+.1f}bp on the day ({band(c, se)})."))
    c, se = get("CBI_median", "y24", 60)
    out.append(("Information shocks keep pushing yields up",
                f"When rates and stocks rise together, the 2y yield is {c:+.1f}bp higher 60 days later "
                f"({band(c, se)}), "
                "consistent with markets revising up their view of the economy."))
    c, se = get("MP_median", "DFII10", 20)
    b, bse = get("MP_median", "T10YIE", 20)
    if not np.isnan(c):
        out.append(("Tightening works through real yields",
                    f"After a 10bp monetary policy shock, the 10y real yield is {c:+.1f}bp higher at 20 days while "
                    f"breakevens move {b:+.1f}bp."))
    return out


# ---- data for the interactive meeting explorer ----------------------------------------------

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


# ---- charts ---------------------------------------------------------------------------------

def fig_history(daily):
    fig = go.Figure()
    for col, name, colour, w in [("y120", "10y yield", CHARCOAL, 1.4), ("rn120", "Expected short rates", NAVY, 1.8),
                                 ("tp120", "Term premium", SKY, 1.8)]:
        fig.add_scatter(x=daily.index, y=daily[col], name=name, line=dict(color=colour, width=w),
                        hovertemplate="%{y:.2f}%")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    for k, (d, label) in enumerate(EVENTS):
        fig.add_vline(x=pd.Timestamp(d).value / 1e6, line=dict(color="#c3cad3", width=1, dash="dot"))
        fig.add_annotation(x=d, y=1 if k % 2 == 0 else 0.95, yref="paper", text=label, showarrow=False,
                           xanchor="left", yanchor="top", xshift=3, font=dict(size=10, color=MUTED))
    start = daily.index[-1] - pd.DateOffset(years=20)
    vis = daily.loc[start:, ["y120", "rn120", "tp120"]]
    pad = 0.08 * (vis.max().max() - vis.min().min())
    fig.update_yaxes(ticksuffix="%", range=[vis.min().min() - pad, vis.max().max() + pad])
    fig.update_xaxes(range=[start, daily.index[-1]],
                     rangeselector=dict(x=1, xanchor="right", y=1.02, yanchor="bottom", bgcolor="#eef2f7",
                                        activecolor="#d3dce8", font=dict(size=11, color=INK2), buttons=[
                                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                                            dict(count=20, label="20Y", step="year", stepmode="backward"),
                                            dict(step="all", label="Since 1961")]))
    return _style(fig, 420)


def fig_curve_today(daily, lag=63):
    mats = [m for m in config.REPORT_MATURITIES if f"y{m}" in daily]
    yrs = [m / 12 for m in mats]
    now, then = daily.iloc[-1], daily.iloc[-1 - lag]
    fig = go.Figure()
    fig.add_scatter(x=yrs, y=[now[f"rn{m}"] for m in mats], name="Expected short rates", mode="lines+markers",
                    line=dict(color=NAVY, width=2), marker=dict(size=8), hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=yrs, y=[now[f"y{m}"] for m in mats], name="Yield (gap = term premium)", mode="lines+markers",
                    line=dict(color=CHARCOAL, width=2), marker=dict(size=8), fill="tonexty", fillcolor=SKY_BAND,
                    hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=yrs, y=[then[f"y{m}"] for m in mats], name="Yield 3 months ago", mode="lines",
                    line=dict(color=MUTED, width=1.4, dash="dot"), hovertemplate="%{y:.2f}%")
    fig.update_xaxes(title=dict(text="maturity (years)", font=dict(size=11)), tickvals=yrs,
                     ticktext=[f"{y:g}y" for y in yrs])
    fig.update_yaxes(ticksuffix="%")
    return _style(fig, 340)


def fig_shocks(shocks):
    s = shocks.loc["1994":]
    x = s.index.normalize()
    width = 1000 * 3600 * 24 * 40
    fig = go.Figure()
    for col, name, colour in [("MP_median", "Monetary policy shock", NAVY), ("CBI_median", "Information shock", SKY)]:
        fig.add_bar(x=x, y=s[col] * 100, name=name, marker=dict(color=colour, line=dict(width=0)), width=width,
                    customdata=[d.strftime("%Y-%m-%d") for d in s.index], hovertemplate="%{y:+.1f}bp")
    fig.update_yaxes(ticksuffix="bp")
    return _style(fig, 280, barmode="relative", hovermode="x")


def fig_fomc_split(fomc_days, n=24):
    t = fomc_days.dropna(subset=["d_yield"]).tail(n)
    x = t.index.strftime("%b %y")
    fig = go.Figure()
    fig.add_bar(x=x, y=t.d_expectations, name="Expected rates", marker=dict(color=NAVY, line=dict(width=0)),
                hovertemplate="%{y:+.1f}bp")
    fig.add_bar(x=x, y=t.d_term_premium, name="Term premium", marker=dict(color=SKY, line=dict(width=0)),
                hovertemplate="%{y:+.1f}bp")
    fig.add_scatter(x=x, y=t.d_yield, name="10y change", mode="markers",
                    marker=dict(color=CHARCOAL, size=8, line=dict(color="white", width=2)),
                    hovertemplate="%{y:+.1f}bp")
    fig.update_yaxes(ticksuffix="bp")
    fig.update_xaxes(type="category", tickangle=0, nticks=8)
    return _style(fig, 320, barmode="relative", bargap=0.35)


def fig_irf(irf, shocks):
    outcomes = [o for o in LABELS if o in set(irf.outcome)]
    ncols = 4
    nrows = -(-len(outcomes) // ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[LABELS[o] for o in outcomes],
                        vertical_spacing=0.16, horizontal_spacing=0.06)
    for (s, colour, band) in zip(shocks, [NAVY, SKY], [NAVY_BAND, SKY_BAND], strict=True):
        for k, o in enumerate(outcomes):
            r, c = k // ncols + 1, k % ncols + 1
            d = irf[(irf.shock == s) & (irf.outcome == o)].sort_values("h")
            lo, hi = d.coef - 1.645 * d.se, d.coef + 1.645 * d.se
            fig.add_scatter(x=list(d.h) + list(d.h[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                            mode="lines", fillcolor=band, line=dict(width=0), showlegend=False,
                            hoverinfo="skip", row=r, col=c)
            unit = UNITS.get(o, "bp")
            fig.add_scatter(x=d.h, y=d.coef, mode="lines+markers", name=f"{SHOCK_NAMES[s]} shock", legendgroup=s,
                            showlegend=(k == 0), line=dict(color=colour, width=2), marker=dict(size=5, color=colour),
                            hovertemplate=f"{SHOCK_NAMES[s]}: %{{y:+.2f}}{unit}<extra>day %{{x}}</extra>",
                            row=r, col=c)
    for k in range(len(outcomes)):
        fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=k // ncols + 1, col=k % ncols + 1)
    fig.update_annotations(font=dict(size=12, color=CHARCOAL, family=FONT))
    fig = _style(fig, 300 * nrows, hovermode="closest")
    fig.update_xaxes(tickvals=[0, 20, 40, 60])
    fig.update_layout(legend=dict(y=1.06))
    return fig


FWD_ORDER = ["fwd_nominal", "fwd_expected", "fwd_tp", "fwd_real", "fwd_breakeven"]


def fig_forwards(path):
    """Response of the 5y5y forward and its pieces, on the day and after 20 trading days."""
    if not path.exists():
        return None
    irf = pd.read_csv(path)
    names = config.FORWARD_OUTCOMES
    present = [o for o in FWD_ORDER if o in set(irf.outcome)]
    panels = [("target", "Target: surprise in today's rate"), ("path", "Path: guidance surprise"),
              ("MP_median", "Monetary policy shock"), ("CBI_median", "Information shock")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[t for _, t in panels], vertical_spacing=0.2,
                        horizontal_spacing=0.08, shared_yaxes=False)
    for k, (shock, _) in enumerate(panels):
        r, c = k // 2 + 1, k % 2 + 1
        for h, colour, label in [(0, NAVY, "On the day"), (20, SKY, "After 20 trading days")]:
            d = irf[(irf.shock == shock) & (irf.h == h)].set_index("outcome").reindex(present)
            fig.add_bar(x=[names[o] for o in present], y=d.coef, name=label, legendgroup=label, showlegend=(k == 0),
                        marker=dict(color=colour, line=dict(width=0)),
                        error_y=dict(type="data", array=1.645 * d.se, color="#9aa6b5", thickness=1, width=3),
                        hovertemplate="%{y:+.1f}bp<extra>" + label + "</extra>", row=r, col=c)
        fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=r, col=c)
    fig.update_annotations(font=dict(size=12, color=CHARCOAL, family=FONT))
    fig = _style(fig, 640, barmode="group", bargap=0.3, hovermode="closest")
    fig.update_xaxes(tickfont=dict(size=10), tickangle=0)
    fig.update_yaxes(ticksuffix="bp")
    fig.update_layout(legend=dict(y=1.08))
    return _html(fig)


def fig_validation(path):
    if not path.exists():
        return None
    j = pd.read_csv(path, index_col=0)
    j.index = pd.PeriodIndex(j.index, freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_scatter(x=j.index, y=j.ACMTP10, name="NY Fed published", line=dict(color="#b8c2ce", width=3.5),
                    hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=j.index, y=j.tp120, name="This project", line=dict(color=NAVY, width=1.4),
                    hovertemplate="%{y:.2f}%")
    fig.update_yaxes(ticksuffix="%")
    return _html(_style(fig, 320))


# ---- page -----------------------------------------------------------------------------------

def build(out_dir: Path = config.SITE_OUT):
    daily = pd.read_csv(P / "acm_daily.csv", index_col=0, parse_dates=True)
    shocks = pd.read_csv(P / "shocks.csv", index_col=0, parse_dates=True)
    fomc_days = pd.read_csv(P / "fomc_days.csv", index_col=0, parse_dates=True)
    irf = pd.read_csv(P / "irf.csv")
    events = pd.read_csv(P / "event_regressions.csv")

    def read_json(name):
        f = P / name
        return json.loads(f.read_text()) if f.exists() else None

    tiles = []
    for key, label, colour, tip in [
        ("y120", "10y yield", CHARCOAL, "Fitted 10-year zero-coupon Treasury yield."),
        ("rn120", "Expected short rates", NAVY, "Average short-term rate the market expects over the next 10 years."),
        ("tp120", "Term premium", SKY, "Extra yield investors demand for holding a 10y bond instead of rolling "
                                       "short-term bills."),
    ]:
        s = daily[key]
        tiles.append({"label": label, "colour": colour, "tip": tip, "value": f"{s.iloc[-1]:.2f}%",
                      "d1": _chg(s, 1), "d21": _chg(s, 21)})

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

    robust = []
    if (P / "robustness.csv").exists():
        rb = pd.read_csv(P / "robustness.csv")
        cell = {(r.result, r.check): r for r in rb.itertuples()}
        for title, pick in [("Full sample and by period", lambda c: not c.startswith("2008-present,")),
                            ("Inside the post-2008 sample, where the guidance results live",
                             lambda c: c == "2008-present" or c.startswith("2008-present,"))]:
            checks = [c for c in dict.fromkeys(rb.check) if pick(c)]
            results = [r for r in dict.fromkeys(rb.result) if any((r, c) in cell for c in checks)]
            robust.append({"title": title,
                           "checks": [c.replace("2008-present, ", "") if c != "2008-present" else "All 2008+"
                                      for c in checks],
                           "rows": [{"result": res, "cells": [cell.get((res, c)) for c in checks]}
                                    for res in results]})

    single = None
    if (P / "single_meeting.csv").exists():
        single = pd.read_csv(P / "single_meeting.csv").to_dict("records")

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    html = env.get_template("index.html").render(
        updated=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        last_date=daily.index[-1].strftime("%d %b %Y"), n_meetings=len(fomc_days),
        tiles=tiles, tp_share=100 * daily.tp120.iloc[-1] / daily.y120.iloc[-1], curve=curve,
        read=todays_read(daily, fomc_days), findings=key_findings(irf),
        fig_history=_html(fig_history(daily), "fig-history"), fig_curve=_html(fig_curve_today(daily)),
        fig_shocks=_html(fig_shocks(shocks), "fig-shocks"), fig_split=_html(fig_fomc_split(fomc_days)),
        fig_irf_jk=_html(fig_irf(irf, ("MP_median", "CBI_median"))),
        fig_irf_gss=_html(fig_irf(irf, ("target", "path"))),
        fig_valid=fig_validation(P / "acm_vs_official.csv"), fig_fwd=fig_forwards(P / "irf_forwards.csv"),
        regime_rows=regime_rows, robust=robust, single=single, acm_valid=read_json("acm_validation.json"),
        shock_valid=read_json("shocks_validation.json"),
        labels=list(LABELS.items()), units=UNITS, scen_json=json.dumps(scen),
        meetings_json=json.dumps(meeting_explorer(daily, fomc_days)), repo_url=config.REPO_URL,
        colours={"navy": NAVY, "sky": SKY, "charcoal": CHARCOAL, "muted": MUTED, "grid": GRID, "ink2": INK2,
                 "font": FONT},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    (out_dir / ".nojekyll").write_text("")
    return out_dir / "index.html"
