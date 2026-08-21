"""Minimal dependency-free SVG plotting. Charts must render with no network."""
import math
from html import escape

PALETTE = ["#2b6cb0", "#c53030", "#2f855a", "#805ad5", "#b7791f", "#00767b", "#97266d"]


def _ticks(lo, hi, count=5):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / count
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(v)
        v += step
    return out


def _fmt(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000 or a < 0.01:
        return f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


class Axes:
    def __init__(self, width=720, height=380, pad=(64, 20, 52, 66),
                 logx=False, logy=False, title="", xlabel="", ylabel=""):
        self.w, self.h = width, height
        self.pl, self.pr, self.pb, self.pt = pad
        self.logx, self.logy = logx, logy
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.series, self.hlines, self.bands, self.notes = [], [], [], []

    def add(self, xs, ys, label="", colour=None, marker=False, dashed=False, width=2.0):
        self.series.append({"xs": list(xs), "ys": list(ys), "label": label,
                            "colour": colour or PALETTE[len(self.series) % len(PALETTE)],
                            "marker": marker, "dashed": dashed, "width": width})

    def hline(self, y, label="", colour="#718096", dashed=True):
        self.hlines.append({"y": y, "label": label, "colour": colour, "dashed": dashed})

    def band(self, y0, y1, label="", colour="#2f855a"):
        self.bands.append({"y0": y0, "y1": y1, "label": label, "colour": colour})

    def _limits(self):
        xs = [x for s in self.series for x in s["xs"]]
        ys = [y for s in self.series for y in s["ys"]]
        ys += [h["y"] for h in self.hlines]
        for b in self.bands:
            ys += [b["y0"], b["y1"]]
        if self.logx:
            xs = [x for x in xs if x > 0]
        if self.logy:
            ys = [y for y in ys if y > 0]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if self.logy:
            y0, y1 = math.log10(y0), math.log10(y1)
        if self.logx:
            x0, x1 = math.log10(x0), math.log10(x1)
        if y1 == y0:
            y0, y1 = y0 - 1, y1 + 1
        margin = (y1 - y0) * 0.08
        return x0, x1, y0 - margin, y1 + margin

    def render(self):
        x0, x1, y0, y1 = self._limits()
        iw = self.w - self.pl - self.pr
        ih = self.h - self.pt - self.pb

        def px(x):
            v = math.log10(x) if self.logx else x
            return self.pl + (v - x0) / (x1 - x0) * iw

        def py(y):
            v = math.log10(y) if self.logy else y
            return self.pt + ih - (v - y0) / (y1 - y0) * ih

        o = [f'<svg viewBox="0 0 {self.w} {self.h}" class="chart" '
             f'xmlns="http://www.w3.org/2000/svg" role="img">']
        if self.title:
            o.append(f'<text x="{self.pl}" y="14" class="ct">{escape(self.title)}</text>')
        o.append(f'<rect x="{self.pl}" y="{self.pt}" width="{iw}" height="{ih}" class="pa"/>')

        for b in self.bands:
            ya, yb = py(b["y1"]), py(b["y0"])
            o.append(f'<rect x="{self.pl}" y="{ya:.1f}" width="{iw}" height="{abs(yb-ya):.1f}" '
                     f'fill="{b["colour"]}" opacity="0.10"/>')
            if b["label"]:
                o.append(f'<text x="{self.pl+iw-4:.1f}" y="{ya+12:.1f}" class="bl" '
                         f'text-anchor="end" fill="{b["colour"]}">{escape(b["label"])}</text>')

        yt = _ticks(y0, y1) if not self.logy else list(range(math.floor(y0), math.ceil(y1) + 1))
        for t in yt:
            val = 10 ** t if self.logy else t
            yy = py(val)
            if not (self.pt - 1 <= yy <= self.pt + ih + 1):
                continue
            o.append(f'<line x1="{self.pl}" y1="{yy:.1f}" x2="{self.pl+iw}" y2="{yy:.1f}" class="gr"/>')
            lab = f"1e{t}" if self.logy else _fmt(val)
            o.append(f'<text x="{self.pl-8}" y="{yy+4:.1f}" class="tk" text-anchor="end">{lab}</text>')

        xt = _ticks(x0, x1) if not self.logx else list(range(math.floor(x0), math.ceil(x1) + 1))
        for t in xt:
            val = 10 ** t if self.logx else t
            xx = px(val)
            if not (self.pl - 1 <= xx <= self.pl + iw + 1):
                continue
            o.append(f'<line x1="{xx:.1f}" y1="{self.pt}" x2="{xx:.1f}" y2="{self.pt+ih}" class="gr"/>')
            lab = f"1e{t}" if self.logx else _fmt(val)
            o.append(f'<text x="{xx:.1f}" y="{self.pt+ih+18}" class="tk" text-anchor="middle">{lab}</text>')

        for h in self.hlines:
            yy = py(h["y"])
            dash = ' stroke-dasharray="5 4"' if h["dashed"] else ""
            o.append(f'<line x1="{self.pl}" y1="{yy:.1f}" x2="{self.pl+iw}" y2="{yy:.1f}" '
                     f'stroke="{h["colour"]}" stroke-width="1.4"{dash}/>')
            if h["label"]:
                o.append(f'<text x="{self.pl+4}" y="{yy-5:.1f}" class="bl" '
                         f'fill="{h["colour"]}">{escape(h["label"])}</text>')

        for s in self.series:
            pts = [(px(x), py(y)) for x, y in zip(s["xs"], s["ys"])
                   if not (self.logy and y <= 0) and not (self.logx and x <= 0)]
            if not pts:
                continue
            dash = ' stroke-dasharray="6 4"' if s["dashed"] else ""
            if len(pts) > 1 and s["width"] > 0:
                d = " ".join(f'{"M" if i==0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
                o.append(f'<path d="{d}" fill="none" stroke="{s["colour"]}" '
                         f'stroke-width="{s["width"]}" stroke-linejoin="round"{dash}/>')
            if s["marker"]:
                # Fill, opacity and radius are hoisted onto the group. With ten thousand
                # points a per-circle copy of those attributes is most of the page weight.
                r = 2.0 if len(pts) > 2000 else (2.6 if len(pts) > 40 else 3.4)
                o.append(f'<g fill="{s["colour"]}" opacity="0.8">')
                o.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>' for x, y in pts)
                o.append("</g>")

        if self.ylabel:
            o.append(f'<text transform="translate(14,{self.pt+ih/2}) rotate(-90)" '
                     f'class="al" text-anchor="middle">{escape(self.ylabel)}</text>')
        if self.xlabel:
            o.append(f'<text x="{self.pl+iw/2}" y="{self.h-6}" class="al" '
                     f'text-anchor="middle">{escape(self.xlabel)}</text>')
        o.append("</svg>")

        leg = [s for s in self.series if s["label"]]
        legend = ""
        if leg:
            items = "".join(
                f'<span class="li"><i style="background:{s["colour"]}"></i>{escape(s["label"])}</span>'
                for s in leg)
            legend = f'<div class="legend">{items}</div>'
        return f'<figure>{"".join(o)}{legend}</figure>'


def grouped_bars(groups, categories, values, title="", ylabel="", width=720, height=340,
                 fmt="{:.3f}", hline=None, hlabel=""):
    """values[category_index][group_index]."""
    pl, pr, pt, pb = 64, 20, 26, 46
    iw, ih = width - pl - pr, height - pt - pb
    flat = [v for row in values for v in row]
    lo, hi = min(flat + ([hline] if hline is not None else [])), max(flat + ([hline] if hline is not None else []))
    lo = min(lo, 0.0)
    hi = hi + (hi - lo) * 0.12
    o = [f'<svg viewBox="0 0 {width} {height}" class="chart" xmlns="http://www.w3.org/2000/svg">']
    if title:
        o.append(f'<text x="{pl}" y="14" class="ct">{escape(title)}</text>')
    o.append(f'<rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" class="pa"/>')
    def py(v):
        return pt + ih - (v - lo) / (hi - lo) * ih
    for t in _ticks(lo, hi):
        yy = py(t)
        if not (pt - 1 <= yy <= pt + ih + 1):
            continue
        o.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+iw}" y2="{yy:.1f}" class="gr"/>')
        o.append(f'<text x="{pl-8}" y="{yy+4:.1f}" class="tk" text-anchor="end">{_fmt(t)}</text>')
    gw = iw / len(groups)
    bw = gw * 0.72 / len(categories)
    for gi, g in enumerate(groups):
        gx = pl + gi * gw
        for ci in range(len(categories)):
            v = values[ci][gi]
            x = gx + gw * 0.14 + ci * bw
            yy, zero = py(v), py(0.0)
            o.append(f'<rect x="{x:.1f}" y="{min(yy,zero):.1f}" width="{bw*0.88:.1f}" '
                     f'height="{max(abs(zero-yy),0.6):.1f}" fill="{PALETTE[ci%len(PALETTE)]}" '
                     f'opacity="0.88"/>')
            o.append(f'<text x="{x+bw*0.44:.1f}" y="{min(yy,zero)-4:.1f}" class="vl" '
                     f'text-anchor="middle">{fmt.format(v)}</text>')
        o.append(f'<text x="{gx+gw/2:.1f}" y="{pt+ih+18}" class="tk" '
                 f'text-anchor="middle">{escape(str(g))}</text>')
    if hline is not None:
        yy = py(hline)
        o.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+iw}" y2="{yy:.1f}" stroke="#c53030" '
                 f'stroke-width="1.5" stroke-dasharray="5 4"/>')
        if hlabel:
            o.append(f'<text x="{pl+iw-4}" y="{yy-5:.1f}" class="bl" text-anchor="end" '
                     f'fill="#c53030">{escape(hlabel)}</text>')
    if ylabel:
        o.append(f'<text transform="translate(14,{pt+ih/2}) rotate(-90)" class="al" '
                 f'text-anchor="middle">{escape(ylabel)}</text>')
    o.append("</svg>")
    items = "".join(f'<span class="li"><i style="background:{PALETTE[i%len(PALETTE)]}"></i>'
                    f'{escape(c)}</span>' for i, c in enumerate(categories))
    return f'<figure>{"".join(o)}<div class="legend">{items}</div></figure>'
