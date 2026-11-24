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

# Colour roles (validated categorical slots 1 and 2, plus neutral ink)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ebeae6"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND_BLUE, BAND_ORANGE = "rgba(42,120,214,0.14)", "rgba(235,104,52,0.14)"
FONT = "Figtree, Helvetica, Arial, sans-serif"

LABELS = {"y24": "2y yield", "y120": "10y yield", "rn120": "10y expected rates",
          "tp120": "10y term premium", "DFII10": "10y real yield", "T10YIE": "10y breakeven",
          "BAA10Y": "Baa credit spread", "DTWEXBGS": "Broad dollar"}
UNITS = {"DTWEXBGS": "%"}
SHOCK_NAMES = {"MP_median": "Monetary policy", "CBI_median": "Information",
               "target": "Target (today's rate)", "path": "Path (guidance)"}
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _style(fig, height=360, **kw):
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=52, r=16, t=36, b=36), font=dict(family=FONT, size=12, color=INK2),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=12, color=INK2)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
        **kw)
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", ticks="")
    return fig


def _html(fig, div_id=None):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG, div_id=div_id)


# ---- numbers for the header tiles and tables ----------------------------------

def _chg(s, k):
    return (s.iloc[-1] - s.iloc[-1 - k]) * 100


def tiles(daily, fomc_days):
    out = []
    for key, label in [("y120", "10y yield"), ("rn120", "Expected short rates"), ("tp120", "Term premium")]:
        s = daily[key]
        out.append({"label": label, "value": f"{s.iloc[-1]:.2f}%", "d1": _chg(s, 1), "d21": _chg(s, 21)})
    return out


def curve_table(daily, fomc_days):
    pos = daily.index.searchsorted(fomc_days.index[-1].normalize())
    rows = []
    for m, tenor in [(24, "2y"), (60, "5y"), (120, "10y")]:
        row = {"tenor": tenor}
        for part in ("y", "rn", "tp"):
            s = daily[f"{part}{m}"]
            row[part] = s.iloc[-1]
            row[part + "_d21"] = _chg(s, 21)
            row[part + "_fomc"] = (s.iloc[-1] - s.iloc[max(pos - 1, 0)]) * 100
        rows.append(row)
    return rows


def last_meeting(fomc_days):
    r = fomc_days.iloc[-1]
    mp, info = r["MP_median"], r["CBI_median"]
    kind = "Monetary policy" if abs(mp) >= abs(info) else "Information"
    direction = "hawkish" if r["pc1"] > 0 else "dovish"
    return {"date": fomc_days.index[-1].strftime("%d %b %Y"), "pc1": r["pc1"] * 100, "mp": mp * 100,
            "info": info * 100, "kind": kind, "direction": direction, "target": r["target"] * 100,
            "path": r["path"] * 100, "d_yield": r["d_yield"], "d_exp": r["d_expectations"],
            "d_tp": r["d_term_premium"]}


# ---- charts ------------------------------------------------------------------------

def fig_history(daily):
    fig = go.Figure()
    for col, name, colour in [("y120", "10y yield", INK), ("rn120", "Expected short rates", BLUE),
                              ("tp120", "Term premium", ORANGE)]:
        fig.add_scatter(x=daily.index, y=daily[col], name=name, line=dict(color=colour, width=1.6),
                        hovertemplate="%{y:.2f}%")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    start = daily.index[-1] - pd.DateOffset(years=20)
    vis = daily.loc[start:, ["y120", "rn120", "tp120"]]
    pad = 0.08 * (vis.max().max() - vis.min().min())
    fig.update_yaxes(ticksuffix="%", range=[vis.min().min() - pad, vis.max().max() + pad])
    fig.update_xaxes(range=[start, daily.index[-1]],
                     rangeselector=dict(x=1, xanchor="right", y=1.02, yanchor="bottom", bgcolor="#f2f1ed",
                                        activecolor="#dcdad3", font=dict(size=11, color=INK2), buttons=[
                                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                                            dict(count=20, label="20Y", step="year", stepmode="backward"),
                                            dict(step="all", label="Since 1961")]))
    return _style(fig, 400)


def fig_shocks(shocks):
    s = shocks.loc["1994":]
    x = s.index.normalize()
    width = 1000 * 3600 * 24 * 40
    fig = go.Figure()
    for col, name, colour in [("MP_median", "Monetary policy shock", BLUE),
                              ("CBI_median", "Information shock", ORANGE)]:
        fig.add_bar(x=x, y=s[col] * 100, name=name, marker=dict(color=colour, line=dict(width=0)),
                    width=width, hovertemplate="%{y:+.1f}bp")
    fig.update_yaxes(ticksuffix="bp")
    return _style(fig, 300, barmode="relative", hovermode="closest")


def fig_fomc_split(fomc_days, n=24):
    t = fomc_days.dropna(subset=["d_yield"]).tail(n)
    x = t.index.strftime("%b %y")
    fig = go.Figure()
    fig.add_bar(x=x, y=t.d_expectations, name="Expected rates", marker=dict(color=BLUE, line=dict(width=0)),
                hovertemplate="%{y:+.1f}bp")
    fig.add_bar(x=x, y=t.d_term_premium, name="Term premium", marker=dict(color=ORANGE, line=dict(width=0)),
                hovertemplate="%{y:+.1f}bp")
    fig.add_scatter(x=x, y=t.d_yield, name="10y change", mode="markers",
                    marker=dict(color=INK, size=8, line=dict(color="white", width=2)),
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
    for (s, colour, band) in zip(shocks, [BLUE, ORANGE], [BAND_BLUE, BAND_ORANGE], strict=True):
        for k, o in enumerate(outcomes):
            r, c = k // ncols + 1, k % ncols + 1
            d = irf[(irf.shock == s) & (irf.outcome == o)].sort_values("h")
            lo, hi = d.coef - 1.645 * d.se, d.coef + 1.645 * d.se
            fig.add_scatter(x=list(d.h) + list(d.h[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                            mode="lines", fillcolor=band, line=dict(width=0), showlegend=False,
                            hoverinfo="skip", row=r, col=c)
            unit = UNITS.get(o, "bp")
            fig.add_scatter(x=d.h, y=d.coef, mode="lines+markers", name=SHOCK_NAMES[s], legendgroup=s,
                            showlegend=(k == 0), line=dict(color=colour, width=2),
                            marker=dict(size=5, color=colour),
                            hovertemplate=f"{SHOCK_NAMES[s]}: %{{y:+.2f}}{unit}<extra>day %{{x}}</extra>",
                            row=r, col=c)
    for k in range(len(outcomes)):
        fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=k // ncols + 1, col=k % ncols + 1)
    fig.update_annotations(font=dict(size=12, color=INK, family=FONT))
    fig = _style(fig, 300 * nrows, hovermode="closest")
    fig.update_xaxes(tickvals=[0, 20, 40, 60])
    fig.update_layout(legend=dict(y=1.06))
    return fig


def fig_validation(path):
    if not path.exists():
        return None
    j = pd.read_csv(path, index_col=0)
    j.index = pd.PeriodIndex(j.index, freq="M").to_timestamp()
    fig = go.Figure()
    fig.add_scatter(x=j.index, y=j.ACMTP10, name="NY Fed published", line=dict(color=MUTED, width=3),
                    hovertemplate="%{y:.2f}%")
    fig.add_scatter(x=j.index, y=j.tp120, name="This project", line=dict(color=ORANGE, width=1.4),
                    hovertemplate="%{y:.2f}%")
    fig.update_yaxes(ticksuffix="%")
    return _html(_style(fig, 320))


# ---- page --------------------------------------------------------------------------

def build(out_dir: Path = config.SITE_OUT):
    daily = pd.read_csv(P / "acm_daily.csv", index_col=0, parse_dates=True)
    shocks = pd.read_csv(P / "shocks.csv", index_col=0, parse_dates=True)
    fomc_days = pd.read_csv(P / "fomc_days.csv", index_col=0, parse_dates=True)
    irf = pd.read_csv(P / "irf.csv")
    events = pd.read_csv(P / "event_regressions.csv")

    def read_json(name):
        f = P / name
        return json.loads(f.read_text()) if f.exists() else None

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

    recent = fomc_days.tail(10).iloc[::-1]
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    html = env.get_template("index.html").render(
        updated=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        last_date=daily.index[-1].strftime("%d %b %Y"),
        tiles=tiles(daily, fomc_days), curve=curve_table(daily, fomc_days), last=last_meeting(fomc_days),
        recent=recent, n_meetings=len(fomc_days),
        fig_history=_html(fig_history(daily), "fig-history"), fig_shocks=_html(fig_shocks(shocks)),
        fig_split=_html(fig_fomc_split(fomc_days)),
        fig_irf_jk=_html(fig_irf(irf, ("MP_median", "CBI_median"))),
        fig_irf_gss=_html(fig_irf(irf, ("target", "path"))),
        fig_valid=fig_validation(P / "acm_vs_official.csv"),
        regime_rows=regime_rows, acm_meta=read_json("acm_meta.json"),
        acm_valid=read_json("acm_validation.json"), shock_valid=read_json("shocks_validation.json"),
        labels=list(LABELS.items()), units=UNITS, scen_json=json.dumps(scen), repo_url=config.REPO_URL,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    (out_dir / ".nojekyll").write_text("")
    return out_dir / "index.html"
