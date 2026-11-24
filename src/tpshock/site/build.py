"""Builds the static dashboard in site/ from data/processed/. Deployed with GitHub Pages."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

from tpshock import config

P = config.PROCESSED
TEMPLATES = Path(__file__).parent / "templates"
BLUE, ORANGE, GREY = "#1f4e79", "#c55a11", "#7f7f7f"
LABELS = {"y24": "2y yield", "y120": "10y yield", "rn120": "10y expected rates",
          "tp120": "10y term premium", "T10YIE": "10y breakeven", "DFII10": "10y real yield",
          "BAA10Y": "Baa spread", "DTWEXBGS": "Broad dollar (%)"}


def _layout(fig, height=380, **kw):
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(template="simple_white", height=height, margin=dict(l=50, r=20, t=40, b=40),
                      font=dict(family="Helvetica, Arial, sans-serif", size=12),
                      legend=dict(orientation="h", y=1.08, x=0), **kw)
    return fig


def _html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def snapshot(daily: pd.DataFrame, fomc_days: pd.DataFrame) -> list[dict]:
    last = daily.index[-1]
    last_fomc = fomc_days.index[-1].normalize()
    pos_fomc = daily.index.searchsorted(last_fomc)
    rows = []
    for m, name in [(24, "2y"), (60, "5y"), (120, "10y")]:
        for part, label in [("y", "Yield"), ("rn", "Expected rates"), ("tp", "Term premium")]:
            s = daily[f"{part}{m}"]

            def chg(k, s=s):
                return (s.iloc[-1] - s.iloc[-1 - k]) * 100

            rows.append({"tenor": name, "part": label, "level": s.iloc[-1],
                         "d1": chg(1), "d5": chg(5), "d21": chg(21),
                         "since_fomc": (s.iloc[-1] - s.iloc[max(pos_fomc - 1, 0)]) * 100})
    return rows, last


def fig_history(daily):
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily.y120, name="10y yield", line=dict(color="black", width=1.2))
    fig.add_scatter(x=daily.index, y=daily.rn120, name="Expected short rates", line=dict(color=BLUE, width=1.2))
    fig.add_scatter(x=daily.index, y=daily.tp120, name="Term premium", line=dict(color=ORANGE, width=1.2))
    fig.update_yaxes(title="percent")
    fig.update_xaxes(rangeselector=dict(x=1, xanchor="right", y=1.08, buttons=[
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(count=20, label="20y", step="year", stepmode="backward"),
        dict(step="all", label="All")]))
    return _layout(fig, 420)


def fig_shocks(shocks):
    s = shocks.loc["1994":]
    x = s.index.normalize()
    width = 1000 * 3600 * 24 * 30  # bars a month wide so they are visible on a date axis
    fig = go.Figure()
    fig.add_bar(x=x, y=s.MP_median * 100, name="Monetary policy shock", marker_color=BLUE,
                width=width, marker_line_width=0)
    fig.add_bar(x=x, y=s.CBI_median * 100, name="Information shock", marker_color=ORANGE,
                width=width, marker_line_width=0)
    fig.update_yaxes(title="bp")
    return _layout(fig, 340, barmode="relative")


def fig_fomc_split(fomc_days, n=40):
    t = fomc_days.dropna(subset=["d_yield"]).tail(n)
    x = t.index.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_bar(x=x, y=t.d_expectations, name="Expected rates", marker_color=BLUE)
    fig.add_bar(x=x, y=t.d_term_premium, name="Term premium", marker_color=ORANGE)
    fig.add_scatter(x=x, y=t.d_yield, name="10y yield change", mode="markers",
                    marker=dict(color="black", size=6))
    fig.update_yaxes(title="bp, close to close")
    fig.update_xaxes(type="category", tickangle=-60, nticks=12)
    return _layout(fig, 380, barmode="relative")


def fig_irf(irf, shocks=("MP_median", "CBI_median"), titles=("Monetary policy shock", "Information shock")):
    outcomes = [o for o in LABELS if o in set(irf.outcome)]
    fig = make_subplots(rows=len(shocks), cols=len(outcomes), shared_xaxes=True,
                        subplot_titles=[f"{LABELS[o]}" for o in outcomes] * len(shocks),
                        vertical_spacing=0.14, horizontal_spacing=0.035)
    for i, (s, colour) in enumerate(zip(shocks, [BLUE, ORANGE], strict=False), start=1):
        for j, o in enumerate(outcomes, start=1):
            d = irf[(irf.shock == s) & (irf.outcome == o)].sort_values("h")
            lo, hi = d.coef - 1.645 * d.se, d.coef + 1.645 * d.se
            fig.add_scatter(x=list(d.h) + list(d.h[::-1]), y=list(hi) + list(lo[::-1]), fill="toself", mode="lines",
                            fillcolor="rgba(127,127,127,0.18)", line=dict(width=0), showlegend=False,
                            hoverinfo="skip", row=i, col=j)
            fig.add_scatter(x=d.h, y=d.coef, mode="lines+markers", marker=dict(size=4, color=colour),
                            line=dict(color=colour, width=1.6), showlegend=False,
                            name=f"{titles[i-1]}: {LABELS[o]}", row=i, col=j)
            fig.add_hline(y=0, line=dict(color=GREY, width=0.6), row=i, col=j)
    for i, t in enumerate(titles, start=1):
        fig.update_yaxes(title_text=t + " (bp)" if i else "", row=i, col=1)
    fig.update_xaxes(title_text="trading days after meeting", row=len(shocks))
    fig.update_annotations(font_size=11)
    return _layout(fig, 330 * len(shocks), hovermode="closest")


def fig_validation(path):
    if not path.exists():
        return None
    j = pd.read_csv(path, index_col=0)
    j.index = pd.PeriodIndex(j.index, freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_scatter(x=j.index, y=j.ACMTP10, name="NY Fed published", line=dict(color=GREY, width=2))
    fig.add_scatter(x=j.index, y=j.tp120, name="This project", line=dict(color=ORANGE, width=1.2))
    fig.update_yaxes(title="10y term premium, percent")
    return _html(_layout(fig, 340))


def build(out_dir: Path = config.SITE_OUT):
    daily = pd.read_csv(P / "acm_daily.csv", index_col=0, parse_dates=True)
    shocks = pd.read_csv(P / "shocks.csv", index_col=0, parse_dates=True)
    fomc_days = pd.read_csv(P / "fomc_days.csv", index_col=0, parse_dates=True)
    irf = pd.read_csv(P / "irf.csv")
    events = pd.read_csv(P / "event_regressions.csv")

    def read_json(name):
        f = P / name
        return json.loads(f.read_text()) if f.exists() else None

    snap, last_date = snapshot(daily, fomc_days)
    last = fomc_days.iloc[-1]
    recent = fomc_days.tail(12).iloc[::-1]

    names = {"MP_median": "1 Monetary policy", "CBI_median": "2 Information", "target": "3 Target", "path": "4 Path"}
    ev = events[events.shock.isin(names)].assign(shock=lambda d: d.shock.map(names))
    ev_table = ev.pivot_table(index=["regime", "shock"], columns="outcome", values=["bp_per_10bp", "se"])
    ev_table = ev_table.rename(index=lambda v: v[2:] if v[:1].isdigit() else v, level=1)
    ev_table = ev_table.reindex(["all", "1990-2007", "2008-present"], level=0)

    # coefficients for the in-browser scenario tool
    scen = {s: {o: irf[(irf.shock == s) & (irf.outcome == o)].sort_values("h")[["h", "coef", "se"]]
                .to_dict("list") for o in irf.outcome.unique()} for s in irf.shock.unique()}

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    html = env.get_template("index.html").render(
        updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        last_date=last_date.strftime("%Y-%m-%d"),
        snap=snap, last=last, last_meeting=fomc_days.index[-1].strftime("%Y-%m-%d"), recent=recent,
        fig_history=_html(fig_history(daily)), fig_shocks=_html(fig_shocks(shocks)),
        fig_split=_html(fig_fomc_split(fomc_days)),
        fig_irf=_html(fig_irf(irf)),
        fig_irf_gss=_html(fig_irf(irf, ("target", "path"), ("Target factor", "Path factor"))),
        fig_valid=fig_validation(P / "acm_vs_official.csv"),
        ev_table=ev_table, acm_meta=read_json("acm_meta.json"),
        acm_valid=read_json("acm_validation.json"), shock_valid=read_json("shocks_validation.json"),
        labels=LABELS, scen_json=json.dumps(scen), repo_url=config.REPO_URL,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    (out_dir / ".nojekyll").write_text("")
    return out_dir / "index.html"
