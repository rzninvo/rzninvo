#!/usr/bin/env python3
"""
gen.py -- deterministic generator for the animated "trajectory" SLAM banner (v3).

Writes banner-dark.svg and banner-light.svg next to this file.

Story (one loop, P seconds of story time): a hand-held camera (an open top-down
view cone) walks a corridor loop around a building core, starting from rest on
keyframe node 0. Map points light up (accent at 60 %, briefly) as they enter its
field of view and settle to grey; the trajectory draws in behind it and
pose-graph keyframes draw their rings as the camera passes. The estimate drifts
(a yaw error that builds up and partly unwinds, plus monocular scale drift), so
when the camera comes back the re-observed outer wall shows up as a grey
"ghost" copy. The camera comes to rest on the place it started from, but its
drifted estimate sits off node 0: the loop closes (the view cone fades out, a
dashed accent constraint current pose -> node 0 appears between two ringed
endpoints that pulse 2x). The pose graph is corrected (every keyframe's drift
eases to identity) and the estimate docks onto node 0; the current-pose ring
and the edge fade once they get short, the ghosts turn bright-neutral
("matched") and fuse, node 0's ring fades and the clean floor plan holds
(>= 2 s, fully static). The map then dissolves; the camera never fades and never
moves across the seam (it is parked exactly on the start pose).

Layout tiers (media queries inside the SVG evaluate against the <img> width;
they cascade with max-width only, so fractional widths never fall between two
tiers):
  * > 780 px  desktop: one-line name + tagline flush left, map flush right;
  * <= 780 / 700 / 620 / 540 px: tagline x1.05 / 1.12 / 1.2 / 1.25 (tapered so a
    resize never jumps by more than ~1 px); <= 700 px: strokes x1.3, camera x1.15;
  * <= 440 px (phones; README column 278-350 px, 398 on a 768 tablet): stacked
    name on two lines ("Roham" / "Zendehdel Nobari") + a one-line tagline
    "SLAM · 3D vision", the map group scaled into the right-hand column, all ink
    inside vb y 12..246 so the hero keeps >= 14 CSS px of air above the README's
    first paragraph; <= 400 / <= 350 px: the text block at x0.82 / x0.9 / x1.

Constraints honoured (GitHub serves repo SVGs inside <img> with a strict CSP):
  * no JS, no SMIL, no <text>, no external refs, no fonts: text is outlined
    with fontTools (GPOS pair kerning applied);
  * only inline CSS @keyframes; every transform is written about the local
    origin (transform-origin 0 0), no vector-effect; dashed paths are single
    subpaths (browsers restart dash patterns per subpath);
  * loops forever: every map layer is at opacity 0 at the seam, the camera is
    at the start pose on both sides of it;
  * prefers-reduced-motion (animation:none) shows the static values, which are
    the finished, corrected map with the camera parked on node 0.
"""
import math
import os
import re
import sys

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# canvas, copy, palette
# ----------------------------------------------------------------------------
W, H = 1200, 300

NAME = "Roham Zendehdel Nobari"
NAME_LINES = ("Roham", "Zendehdel Nobari")                      # phones
TAGLINE = "SLAM · 3D vision · egocentric perception"
PH_TAGLINE = "SLAM · 3D vision"                                 # phones: one line
FONT_NAME = "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf"
FONT_TAG = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
NAME_SIZE, NAME_TRACK = 61.0, -0.014       # px, em
TAG_SIZE, TAG_TRACK = 24.0, 0.0
NAME_BASE = 146.0
TAG_BASE = 194.0
X_INK = 1.5                 # left ink edge of the name (flush with the README text)
MAP_RIGHT_INK = 1198.5      # right ink edge of the (corrected) map

# phones: the README column is 278 px (360 px Android) .. 350 px, 398 on a 768 tablet
PH_MAX_W = 440
PH_REF_W = 308.0            # the 390 px iPhone
PH_MIN_W = 278.0            # the 360 px Android
PH_NAME_SIZE = 86.5         # -> 20.0 CSS px at 278, 22.2 at 308
PH_NAME_LEAD = 1.04         # baseline-to-baseline, em
PH_TAG_SIZE = 47.5          # -> 11.0 CSS px at 278, 12.2 at 308
PH_TAG_GAP = 0.34           # name baseline 2 -> tagline cap top, in name em
PH_CLEAR_CSS = 12.0         # required clearance map/camera ink <-> text at 278 px (CSS px)
PH_BAND = (12.0, 243.0)     # all phone ink stays inside this vb y band at every t (>= 13 CSS px
                            # of empty image below it even at 278 px)
# text block scale on the upper phone widths (cascading max-width): smooths the
# stacked -> one-line change and keeps landscape / tablet names from growing
PH_SUB = ((400, 0.9), (350, 1.0))
PH_SUB_TOP = 0.82           # 401..440 px

# tagline taper above the phone tier: (max-width, multiplier), cascading
TAG_BANDS = ((780, 1.05), (700, 1.12), (620, 1.2), (540, 1.25))
MID_MAX_W = 700             # strokes x1.3 at or below this width (down to PH_MAX_W)

CLEAR_MIN = 50.0            # hard requirement on desktop/mid: map/camera ink vs name bbox
MAP_H_MAX = 236.0           # desktop map ink may use at most this much height
MAP_DY = -3.0               # desktop map lifted a little: the transient ghost wall hangs below the clean map
MID_MAP_DY = -7.0           # <= MID_MAX_W: lifted further (more air above the README's first paragraph)

# stroke multipliers per tier (rendered = SW * mult); cam = camera glyph scale,
# q = loop-closure ring scale
TIERS = {
    "desk": dict(p=1.0, w=1.0, t=1.0, n=1.0, e=1.0, f=1.0, cam=1.0, q=1.0),
    "mid": dict(p=1.3, w=1.3, t=1.3, n=1.3, e=1.3, f=1.3, cam=1.15, q=1.15),
    "phone": dict(p=2.1, w=2.1, t=2.0, n=1.8, e=1.7, f=1.9, cam=1.5, q=1.5),
}

THEMES = {
    "dark": dict(
        name="#e6edf3", tag="#9198a1", bg="#0d1117",
        pt="#7d8590", pt_new="#c9d1d9", traj="#b7bfc8",
        acc="#ff8f4a", w_op=.62, wedge_op=.14,
    ),
    "light": dict(
        name="#1f2328", tag="#59636e", bg="#ffffff",
        pt="#6e7781", pt_new="#3d444d", traj="#4b535d",
        acc="#e8620e", w_op=.8, wedge_op=.11,
    ),
}

# ----------------------------------------------------------------------------
# timeline (story seconds)
# ----------------------------------------------------------------------------
T_IN = (0.0, 0.5)           # map layer fades in, ease-out
T_MOVE = (0.0, 10.3)        # camera walks the loop (from rest on node 0 back to rest on node 0)
T_CLOSE = 10.42             # loop detected: constraint edge + endpoint pulses
T_GREY = T_CLOSE - 0.1      # every landmark / ghost is grey (no accent) by now
T_EDGE_IN = 0.3             # edge fade-in duration
T_CORR = (10.95, 12.95)     # pose-graph correction (2.0 s, eased)
T_MATCH = 0.35              # ghosts turn "matched" over the first 0.35 s of it
T_FUSE = (11.9, 12.95)      # duplicate landmarks fuse (fade) as they align
T_EDGE_OUT = (12.65, 13.0)  # node-0 ring / highlight fade, the view cone returns
T_STATIC = 13.0             # from here on nothing moves until the fade
T_OUT = (15.1, 16.0)        # map fades (0.9 s, eased); the camera stays
P = 16.0                    # loop period
EDGE_SHORT = (20.0, 11.0)   # current-pose ring + edge fade while the edge shrinks between these lengths (vb)
PHASE = 4.8                 # uniform negative delay: first paint = story t
BUCKET_DT = 0.25            # map (reveal + drift) update rate while moving: 4 Hz
CAM_DT = 0.125              # camera pose keyframes: 8 Hz
RAMP = 0.2                  # landmark light-up ramp
ACC_HOLD = 0.25             # accent hold after the ramp
ACC_FADE = 0.45             # accent -> bright neutral
NEW_HOLD = 2.4              # bright -> settled grey (ends by T_CORR[1] at the latest)
FRESH_A = 0.6               # fresh landmarks: accent at 60 % alpha
PULSE = (0.3, 0.8)          # endpoint rings: 1x -> 2x at +0.3 s, back to 1x at +0.8 s
PULSE_S = 2.0
FADE_IN_CSS = "cubic-bezier(.2,.6,.35,1)"     # ease-out
FADE_OUT_CSS = "cubic-bezier(.4,0,.6,1)"      # ease-in-out, gentle at both ends

# motion caps (story time, on screen, desktop viewBox px)
CAP_OMEGA_DEG = 120.0       # camera yaw rate
CAP_SPEED_PX = 128.0        # camera speed
CAP_CORR_PX = 52.0          # any point during the pose-graph correction
WALK_A0 = 0.0               # the walk starts (and ends) at rest on node 0

# camera glyph: an open top-down view cone (two rays, no image-plane bar, no
# fill) and an apex dot big enough to cover node 0's ring when parked on it
FR_RAY_D = 19.0
DOT_R = 3.3
CONE_DIM = 0                # the cone steps out completely while the constraint is shown

# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def num(v, d=1):
    """Compact number formatting for SVG/CSS."""
    s = f"{v:.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def pct(t):
    assert -1e-9 <= t <= P + 1e-9, t
    return num(100.0 * min(max(t, 0.0), P) / P, 3) + "%"


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cubic_bezier(p1x, p1y, p2x, p2y):
    """CSS cubic-bezier easing as a python function of progress in [0,1]."""
    def bx(s):
        return 3 * p1x * s * (1 - s) ** 2 + 3 * p2x * s * s * (1 - s) + s ** 3

    def by(s):
        return 3 * p1y * s * (1 - s) ** 2 + 3 * p2y * s * s * (1 - s) + s ** 3

    def f(x):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if bx(mid) < x:
                lo = mid
            else:
                hi = mid
        return by(0.5 * (lo + hi))
    return f


EASE_CSS = "cubic-bezier(.45,0,.55,1)"       # pose-graph correction: symmetric, low peak speed
EASE = cubic_bezier(.45, 0, .55, 1)
EASE_OUT_CSS = "cubic-bezier(.2,.6,.35,1)"
EASE_OUT = cubic_bezier(.2, .6, .35, 1)
EASE_INOUT_CSS = "cubic-bezier(.45,0,.55,1)"
EASE_INOUT = cubic_bezier(.45, 0, .55, 1)
PRE = cubic_bezier(.333, 0, .667, .333)


class Keyframes:
    """Collects (time, {prop: value}) stops, merges equal times, emits CSS."""

    def __init__(self, name):
        self.name, self.stops = name, {}

    def at(self, t, **props):
        key = round(float(t), 4)
        d = self.stops.setdefault(key, {})
        for k, v in props.items():
            d[k.replace("_", "-")] = v
        return self

    def css(self):
        rules = {}
        for t in sorted(self.stops):
            body = ";".join(f"{k}:{v}" for k, v in self.stops[t].items())
            rules.setdefault(body, []).append(pct(t))
        return "@keyframes %s{%s}" % (self.name, "".join(
            "%s{%s}" % (",".join(sels), body) for body, sels in rules.items()))


# ----------------------------------------------------------------------------
# text -> outlines (fontTools, with GPOS pair kerning)
# ----------------------------------------------------------------------------
class Font:
    def __init__(self, path):
        self.f = TTFont(path)
        self.gs = self.f.getGlyphSet()
        self.cmap = self.f.getBestCmap()
        self.hmtx = self.f["hmtx"]
        self.upm = self.f["head"].unitsPerEm
        self.cap = self.f["OS/2"].sCapHeight
        self.xh = self.f["OS/2"].sxHeight
        gpos = self.f["GPOS"].table
        idx = set()
        for fr in gpos.FeatureList.FeatureRecord:
            if fr.FeatureTag == "kern":
                idx.update(fr.Feature.LookupListIndex)
        self.lookups = [gpos.LookupList.Lookup[i] for i in sorted(idx)]

    def kern(self, a, b):
        total = 0
        for lk in self.lookups:
            for st in lk.SubTable:
                if lk.LookupType == 9:
                    st = st.ExtSubTable
                cov = st.Coverage.glyphs
                if a not in cov:
                    continue
                v = None
                if st.Format == 1:
                    for r in st.PairSet[cov.index(a)].PairValueRecord:
                        if r.SecondGlyph == b:
                            v = getattr(r.Value1, "XAdvance", 0) if r.Value1 else 0
                            break
                else:
                    c1 = st.ClassDef1.classDefs.get(a, 0)
                    c2 = st.ClassDef2.classDefs.get(b, 0)
                    rec = st.Class1Record[c1].Class2Record[c2]
                    v = getattr(rec.Value1, "XAdvance", 0) if rec.Value1 else 0
                if v is not None:
                    total += v or 0
                    break
        return total

    def layout(self, s, size, track_em=0.0):
        """Return [(char, glyph, x_px)] and advance width in px."""
        sc = size / self.upm
        glyphs = [self.cmap[ord(c)] for c in s]
        out, x = [], 0.0
        for i, g in enumerate(glyphs):
            out.append((s[i], g, x * sc))
            x += self.hmtx[g][0] + track_em * self.upm
            if i + 1 < len(glyphs):
                x += self.kern(g, glyphs[i + 1])
        x -= track_em * self.upm
        return out, x * sc

    def outline(self, items, size, x0, y0):
        sc = size / self.upm
        pen = SVGPathPen(self.gs, ntos=lambda v: num(v, 2))
        for _, g, gx in items:
            tp = TransformPen(pen, (sc, 0, 0, -sc, x0 + gx, y0))
            self.gs[g].draw(tp)
        return pen.getCommands()

    def bbox(self, items, size, x0, y0):
        sc = size / self.upm
        pen = BoundsPen(self.gs)
        for _, g, gx in items:
            tp = TransformPen(pen, (sc, 0, 0, -sc, x0 + gx, y0))
            self.gs[g].draw(tp)
        return pen.bounds          # xmin, ymin, xmax, ymax (screen y down)


def union(*bbs):
    return (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))


def text_layout():
    fname, ftag = Font(FONT_NAME), Font(FONT_TAG)
    # ---- desktop: one line, left ink flush at X_INK
    name_items, name_w = fname.layout(NAME, NAME_SIZE, NAME_TRACK)
    name_x = X_INK - fname.bbox(name_items, NAME_SIZE, 0.0, NAME_BASE)[0]
    name_bb = fname.bbox(name_items, NAME_SIZE, name_x, NAME_BASE)
    tag_items, tag_w = ftag.layout(TAGLINE, TAG_SIZE, TAG_TRACK)
    s_left = ftag.bbox(tag_items[:1], TAG_SIZE, 0.0, TAG_BASE)[0]
    tag_x = name_bb[0] - s_left - 0.3      # optical: round S overshoots the R stem slightly
    tag_bb = ftag.bbox(tag_items, TAG_SIZE, tag_x, TAG_BASE)
    # tagline bands: the tagline is scaled about its cap-top-left and the name
    # lifted by half the growth, so the block stays centred
    tag_cap_top = TAG_BASE - ftag.cap / ftag.upm * TAG_SIZE
    bands = []
    for wmax, m in TAG_BANDS:
        lift = (m - 1) * (tag_bb[3] - tag_cap_top) / 2
        tx0 = tag_bb[0]
        tb = (tx0, tag_cap_top - lift + (tag_bb[1] - tag_cap_top) * m,
              tx0 + (tag_bb[2] - tx0) * m, tag_cap_top - lift + (tag_bb[3] - tag_cap_top) * m)
        nb = (name_bb[0], name_bb[1] - lift, name_bb[2], name_bb[3] - lift)
        bands.append(dict(wmax=wmax, m=m, lift=lift, bb=[nb, tb],
                          tier="mid" if wmax <= MID_MAX_W else "desk"))
    # ---- phones: two name lines + a one-line tagline, each line's ink flush at
    # X_INK; baselines relative to the first line (placed by solve_phone)
    cap_t = ftag.cap / ftag.upm * PH_TAG_SIZE
    lines = []
    for ln in NAME_LINES:
        it, _ = fname.layout(ln, PH_NAME_SIZE, NAME_TRACK)
        lines.append(("n", it, X_INK - fname.bbox(it, PH_NAME_SIZE, 0.0, 0.0)[0], PH_NAME_SIZE))
    it, _ = ftag.layout(PH_TAGLINE, PH_TAG_SIZE)
    lines.append(("t", it, X_INK - 0.4 - ftag.bbox(it[:1], PH_TAG_SIZE, 0.0, 0.0)[0], PH_TAG_SIZE))
    b1 = PH_NAME_LEAD * PH_NAME_SIZE
    bases = [0.0, b1, b1 + PH_TAG_GAP * PH_NAME_SIZE + cap_t]
    ph_rel = []
    for (k, it, x, sz), b in zip(lines, bases):
        f = fname if k == "n" else ftag
        ph_rel.append(dict(kind=k, items=it, x=x, base=b, size=sz, bb=f.bbox(it, sz, x, b)))
    return dict(fname=fname, ftag=ftag, name_items=name_items, name_w=name_w, name_x=name_x, name_bb=name_bb,
                tag_items=tag_items, tag_w=tag_w, tag_x=tag_x, tag_bb=tag_bb, tag_cap_top=tag_cap_top,
                bands=bands, ph_rel=ph_rel, ph=None, ph_name_bb=None, ph_text_bb=None)


def place_phone_text(text, yc):
    """Put the phone text block's ink centre at y = yc."""
    rel = text["ph_rel"]
    top = min(l["bb"][1] for l in rel)
    bot = max(l["bb"][3] for l in rel)
    dy = yc - (top + bot) / 2
    ph = []
    for l in rel:
        bb = l["bb"]
        ph.append(dict(l, base=l["base"] + dy, bb=(bb[0], bb[1] + dy, bb[2], bb[3] + dy)))
    text["ph"] = ph
    text["ph_yc"] = yc
    text["ph_name_bb"] = union(*[p["bb"] for p in ph if p["kind"] == "n"])
    text["ph_text_bb"] = union(*[p["bb"] for p in ph])
    return text["ph_text_bb"]


# ----------------------------------------------------------------------------
# the world: a corridor loop around a building core (metres, y = north)
# The long east-west runs are compressed by KX so the plan fills the banner's
# right-hand column; beyond the east corridor the plan is compressed by KE.
# ----------------------------------------------------------------------------
KX = 0.80
XC0, XC1 = 1.3, 30.7
XC2, KE = 33.6, 0.45


def xm(x):
    if x > XC2:
        return xm(XC2) + (x - XC2) * KE
    if x <= XC0:
        return x
    if x <= XC1:
        return XC0 + (x - XC0) * KX
    return x - (1 - KX) * (XC1 - XC0)


DROP_KINDS = ("office", "atrium")          # simulated, neither drawn nor occluding
NODRAW_KINDS = DROP_KINDS + ("part",)      # "part": the core's interior partitions occlude but
                                           # are not drawn (seen through doors they read as specks)
OUTSIDE_TOL_M = 0.55
THIN_KEEP = 0.8
GHOST_KINDS = ("shell",)    # ghost walls only for the outer shell


def build_world(rng):
    occ = []      # occluding segments
    feat = []     # (segment, density, kind) feature-bearing surfaces
    kind = ["shell"]

    def add(p, q, occluder=True, dens=1.0):
        p, q = (xm(p[0]), p[1]), (xm(q[0]), q[1])
        if occluder and kind[0] not in DROP_KINDS:
            occ.append((p, q))
        feat.append(((p, q), dens, kind[0]))

    def hwall(y, x1, x2, gaps=(), **kw):
        xs = [x1]
        for a, b in sorted(gaps):
            xs += [a, b]
        xs.append(x2)
        for i in range(0, len(xs), 2):
            if xs[i + 1] - xs[i] > 0.05:
                add((xs[i], y), (xs[i + 1], y), **kw)

    def vwall(x, y1, y2, gaps=(), **kw):
        ys = [y1]
        for a, b in sorted(gaps):
            ys += [a, b]
        ys.append(y2)
        for i in range(0, len(ys), 2):
            if ys[i + 1] - ys[i] > 0.05:
                add((x, ys[i]), (x, ys[i + 1]), **kw)

    def box(cx, cy, w, h, **kw):
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        hwall(y0, x0, x1, **kw)
        hwall(y1, x0, x1, **kw)
        vwall(x0, y0, y1, **kw)
        vwall(x1, y0, y1, **kw)

    # outer shell
    hwall(-1.3, -1.3, 38.0, dens=2.2)       # south wall: rich texture (the ghost wall)
    hwall(15.3, -1.3, 38.0, gaps=[(5.4, 7.0), (15.6, 17.2), (24.6, 26.2)])
    vwall(-1.3, -1.3, 15.3, gaps=[(6.3, 7.9)])
    vwall(38.0, -1.3, 15.3)
    # building core (rooms behind doors)
    kind[0] = "core"
    hwall(1.3, 1.3, 30.7, gaps=[(8.6, 10.2), (19.6, 21.2)])
    hwall(12.7, 1.3, 30.7, gaps=[(11.6, 13.2), (25.8, 27.4)])
    kind[0] = "part"            # the core's short end walls and interior partitions: occluders only
    vwall(1.3, 1.3, 12.7)
    vwall(30.7, 1.3, 12.7)
    hwall(7.0, 1.3, 30.7, dens=0.7)
    for x in (6.5, 14.5, 23.0):
        vwall(x, 1.3, 7.0, dens=0.7)
    for x in (9.0, 18.5, 24.5):
        vwall(x, 7.0, 12.7, dens=0.7)
    # offices north of the top corridor (simulated, not drawn)
    kind[0] = "office"
    hwall(19.6, -1.3, 30.0, dens=0.7)
    vwall(30.0, 15.3, 19.6, dens=0.7)
    for x in (3.5, 11.0, 21.0):
        vwall(x, 15.3, 19.6, dens=0.7)
    # atrium: pillars and two low tables (simulated, not drawn)
    kind[0] = "atrium"
    for cy in (3.6, 7.0, 10.4):
        box(34.9, cy, 0.7, 0.7, dens=1.5)
    for cy in (5.3, 8.7):
        box(36.5, cy, 1.1, 0.7, occluder=False, dens=1.2)

    pts, strength, kinds = [], [], []
    for (p, q), dens, kd in feat:
        p, q = np.array(p, float), np.array(q, float)
        L = np.linalg.norm(q - p)
        d = (q - p) / L
        n = np.array([-d[1], d[0]])
        m = max(2, int(L / 0.11))
        s = (np.arange(m) + rng.uniform(0.1, 0.9, m)) / m * L
        ph = rng.uniform(0, 2 * np.pi, 4)
        tex = (0.5 + 0.28 * np.sin(s / 1.9 + ph[0]) + 0.22 * np.sin(s / 0.83 + ph[1])
               + 0.14 * np.sin(s / 0.37 + ph[2]) + 0.1 * np.sin(s / 3.7 + ph[3]))
        prob = dens * (0.05 + 0.8 * np.clip(tex, 0, 1) ** 2.0)
        prob += 0.9 * (np.exp(-(s / 0.22) ** 2) + np.exp(-((L - s) / 0.22) ** 2))
        keep = rng.uniform(0, 1, m) < prob
        for si in s[keep]:
            pts.append(p + d * si + n * rng.normal(0, 0.025))
            strength.append(rng.uniform())
            kinds.append(kd)
    drawn = [(np.array(p, float), np.array(q, float)) for (p, q), _, kd in feat if kd not in NODRAW_KINDS]
    return occ, np.array(pts), np.array(strength), np.array(kinds), drawn


def seg_dist(pts, segs):
    best = np.full(len(pts), np.inf)
    for a, b in segs:
        ab = b - a
        u = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(pts - (a + u[:, None] * ab), axis=1))
    return best


# ----------------------------------------------------------------------------
# the walk: smooth hand-held loop, capped speeds, heading = direction of travel.
# It starts from rest on node 0 (the start pose) and ends at rest exactly on it,
# with the start heading (so the camera can stay parked across the loop seam).
# ----------------------------------------------------------------------------
START = (6.0, 0.05)
WAYPOINTS_MID = [
    (10.0, 0.05), (16.0, -0.1), (24.0, 0.1), (29.6, 0.0),
    (31.9, 1.2), (32.7, 3.4), (33.3, 7.0), (32.8, 10.6), (31.8, 13.2),
    (29.2, 14.05), (22.0, 13.9), (14.0, 14.1), (6.0, 13.95), (2.0, 13.85),
    (0.0, 12.2), (-0.1, 8.0), (0.1, 4.0), (0.25, 1.8), (1.8, 0.3),
]
KF_EVERY = 4.2              # keyframe spacing (m); node 0 is the start pose

DRIFT_YAW_PER_M = math.radians(0.0)
DRIFT_YAW_PER_RAD = 0.0
DRIFT_SCALE = 0.06
# yaw error as a sum of smooth steps (deg, u0, u1) over u = s / S: it builds up
# counter-clockwise on the east / north legs, unwinds on the top leg and goes
# slightly clockwise on the way back, so the loop ends mostly *behind* node 0
# (the constraint then falls inside the view cone) and a little south of it
# (the re-observed outer wall ghosts just outside the real one)
YAW_TERMS = [(5.5, 0.20, 0.36), (-5.5, 0.40, 0.50), (-5.5, 0.68, 0.82)]

_F_GRID = np.linspace(0, 1, 4001)


def _ease_table(a0):
    f = (a0 + (1 - a0) * smoothstep(0, 0.06, _F_GRID)) * smoothstep(0, 0.10, 1 - _F_GRID) + 1e-4
    F = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2)])
    F /= F[-1]
    return F, float(np.max(np.gradient(F, _F_GRID)))


_F, _F_PEAK = _ease_table(WALK_A0)


def build_walk(rng, sc_px_per_m):
    wps = [START] + WAYPOINTS_MID + [(3.6, START[1]), START]
    wp = np.array([(xm(x), y) for x, y in wps], float)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    s_wp = np.concatenate([[0], np.cumsum(seg)])
    ds = 0.05
    s = np.arange(0, s_wp[-1] + 1e-9, ds)
    xy = np.stack([np.interp(s, s_wp, wp[:, 0]), np.interp(s, s_wp, wp[:, 1])], 1)
    sig = int(1.1 / ds)
    pad = 4 * sig
    d0 = xy[1] - xy[0]
    d1 = xy[-1] - xy[-2]
    pre = xy[0] - d0 * np.arange(pad, 0, -1)[:, None]
    post = xy[-1] + d1 * np.arange(1, pad + 1)[:, None]
    ext = np.concatenate([pre, xy, post])
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / sig) ** 2)
    k /= k.sum()
    sm = np.stack([np.convolve(ext[:, i], k, mode="same") for i in range(2)], 1)[pad:-pad]
    seg = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    S = s[-1]
    s2 = np.linspace(0, S, int(round(S / ds)) + 1)
    xy = np.stack([np.interp(s2, s, sm[:, 0]), np.interp(s2, s, sm[:, 1])], 1)
    s = s2
    tan = np.gradient(xy, axis=0)
    psi_smooth = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    ph = rng.uniform(0, 2 * np.pi, 3)
    lat = 0.07 * np.sin(2 * np.pi * s / 9.5 + ph[0]) + 0.025 * np.sin(2 * np.pi * s / 4.1 + ph[1])
    lat *= smoothstep(0, 1.5, s) * smoothstep(0, 1.5, S - s)
    xy = xy + nrm * lat[:, None]
    # close the loop exactly: the (smoothed) end lands on the start point and the
    # end heading equals the start heading (both blended in over the last 3 m)
    blend = smoothstep(S - 3.0, S, s)
    xy = xy + (xy[0] - xy[-1])[None] * blend[:, None]
    tan = np.gradient(xy, axis=0)
    psi = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    turns = round((psi[-1] - psi[0]) / (2 * np.pi))
    psi = psi + (psi[0] + 2 * np.pi * turns - psi[-1]) * blend
    kap = np.abs(np.gradient(psi) / ds)
    kap = np.convolve(kap, np.ones(9) / 9, mode="same")
    dur = T_MOVE[1] - T_MOVE[0]
    om = math.radians(CAP_OMEGA_DEG) * 0.9 / _F_PEAK
    vmax = CAP_SPEED_PX / sc_px_per_m * 0.95 / _F_PEAK
    gk = np.exp(-0.5 * (np.arange(-40, 41) * ds / 0.9) ** 2)
    gk /= gk.sum()

    def pace(V):
        v = np.minimum(np.minimum(V, vmax), om / np.maximum(kap, 1e-6))
        pc = 1.0 / v
        pc = np.convolve(np.pad(pc, 40, mode="edge"), gk, mode="same")[40:-40]
        return np.maximum(pc, 1.0 / v)
    lo, hi = 0.1, 100.0
    for _ in range(80):
        V = 0.5 * (lo + hi)
        if np.sum(pace(V)[:-1] * ds) > dur:
            lo = V
        else:
            hi = V
    pc = pace(hi)
    tau = np.concatenate([[0], np.cumsum(pc[:-1] * ds)])
    assert abs(tau[-1] - dur) < 0.05, ("walk does not fit T_MOVE under the speed caps", tau[-1], dur)
    turn = np.concatenate([[0], np.cumsum(np.abs(np.diff(psi_smooth)))])
    e = DRIFT_YAW_PER_M * s + DRIFT_YAW_PER_RAD * turn
    for deg, u0, u1 in YAW_TERMS:
        e = e + math.radians(deg) * smoothstep(u0, u1, s / S)
    scale = 1.0 + DRIFT_SCALE * s / S
    dxy = np.diff(xy, axis=0)
    ce, se = np.cos(e[:-1]), np.sin(e[:-1])
    rot_ = np.stack([ce * dxy[:, 0] - se * dxy[:, 1], se * dxy[:, 0] + ce * dxy[:, 1]], 1)
    est = np.concatenate([xy[:1], xy[0] + np.cumsum(rot_ * scale[:-1, None], axis=0)])
    return dict(s=s, xy=xy, psi=psi, tau=tau, e=e, est=est, S=S, V=hi)


def walk_at(walk, t):
    t = np.asarray(t, float)
    u = np.clip((t - T_MOVE[0]) / (T_MOVE[1] - T_MOVE[0]), 0, 1)
    tau = np.interp(u, _F_GRID, _F) * walk["tau"][-1]
    s = np.interp(tau, walk["tau"], walk["s"])
    sp = walk["s"]
    xy = np.stack([np.interp(s, sp, walk["xy"][:, 0]), np.interp(s, sp, walk["xy"][:, 1])], -1)
    est = np.stack([np.interp(s, sp, walk["est"][:, 0]), np.interp(s, sp, walk["est"][:, 1])], -1)
    psi = np.interp(s, sp, walk["psi"])
    e = np.interp(s, sp, walk["e"])
    return s, xy, psi, est, e


# ----------------------------------------------------------------------------
# visibility simulation -> when each landmark is first observed
# ----------------------------------------------------------------------------
FOV = math.radians(78)
RANGE = 8.5
MIN_DEPTH = 0.9           # closer points are not triangulated (and would light up behind the cone)
DUP_EARLY, DUP_LATE = 0.22, 0.86


def occluded(c, p, A, B):
    r = p - c
    sv = B - A
    denom = r[:, None, 0] * sv[None, :, 1] - r[:, None, 1] * sv[None, :, 0]
    qp = A[None, :, :] - c
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (qp[..., 0] * sv[None, :, 1] - qp[..., 1] * sv[None, :, 0]) / denom
        uu = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-9) & (tt > 1e-3) & (tt < 0.975) & (uu >= -1e-3) & (uu <= 1 + 1e-3)
    return hit.any(axis=1)


def bucket_times():
    n_b = int(round((T_MOVE[1] - T_MOVE[0]) / BUCKET_DT))
    dt = (T_MOVE[1] - T_MOVE[0]) / n_b
    return T_MOVE[0] + dt * np.arange(n_b + 1), dt


def cam_times():
    n_b = len(bucket_times()[0]) - 1
    sub = int(round(BUCKET_DT / CAM_DT))
    n_c = n_b * sub
    dt = (T_MOVE[1] - T_MOVE[0]) / n_c
    return T_MOVE[0] + dt * np.arange(n_c + 1), dt, sub


def rendered_cam_world(walk, ts):
    tb, _, _ = cam_times()
    _, kxy, kpsi, _, _ = walk_at(walk, tb)
    kpsi = np.unwrap(kpsi)
    ts = np.asarray(ts, float)
    xy = np.stack([np.interp(ts, tb, kxy[:, 0]), np.interp(ts, tb, kxy[:, 1])], -1)
    return xy, np.interp(ts, tb, kpsi)


def simulate(walk, occ, pts, kinds):
    A = np.array([o[0] for o in occ])
    B = np.array([o[1] for o in occ])
    ts = np.arange(0.0, T_MOVE[1] + 1e-9, 1 / 60)
    cxy, cpsi = rendered_cam_world(walk, ts)
    first = np.full(len(pts), np.nan)
    first_cam = np.zeros((len(pts), 2))
    late_seen = np.full(len(pts), np.nan)
    dur = T_MOVE[1] - T_MOVE[0]
    for k, t in enumerate(ts):
        c = cxy[k]
        d = pts - c
        dist = np.linalg.norm(d, axis=1)
        bear = np.angle(np.exp(1j * (np.arctan2(d[:, 1], d[:, 0]) - cpsi[k])))
        cand = np.where((dist < RANGE) & (dist > MIN_DEPTH) & (np.abs(bear) < FOV / 2))[0]
        if len(cand) == 0:
            continue
        vis = cand[~occluded(c, pts[cand], A, B)]
        new = vis[np.isnan(first[vis])]
        first[new] = t
        first_cam[new] = c
        if t > T_MOVE[0] + DUP_LATE * dur:
            lv = vis[np.isnan(late_seen[vis])]
            late_seen[lv] = t
    ok = ~np.isnan(first)
    dups = [(i, late_seen[i]) for i in np.where(ok)[0]
            if first[i] < T_MOVE[0] + DUP_EARLY * dur and not np.isnan(late_seen[i])]
    return first, first_cam, ok, dups, (A, B)


# ----------------------------------------------------------------------------
# geometry: world -> screen (pivot-relative), rigid drift transforms
# ----------------------------------------------------------------------------
class Screen:
    def __init__(self, sc, o):
        self.sc = sc
        self.M = sc * np.array([[1.0, 0.0], [0.0, -1.0]])
        self.o = np.asarray(o, float)

    def pt(self, p):
        return np.asarray(p) @ self.M.T + self.o

    def to_world(self, q):
        return (np.asarray(q) - self.o) @ np.linalg.inv(self.M).T

    @staticmethod
    def heading(psi):
        return -np.degrees(psi)

    def rigid(self, e, t_w, pivot):
        R = np.array([[math.cos(e), -math.sin(e)], [math.sin(e), math.cos(e)]])
        Q = self.M @ R @ np.linalg.inv(self.M)
        b = -Q @ self.o + self.M @ t_w + self.o
        tcss = b - pivot + Q @ pivot
        ang = math.degrees(math.atan2(Q[1, 0], Q[0, 0]))
        return np.array(tcss), ang


def rot(v, ang_deg):
    a = math.radians(ang_deg)
    v = np.asarray(v, float)
    c, s = math.cos(a), math.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def apply_tf(p, tcss, ang, k=1.0):
    return k * np.asarray(tcss) + rot(p, k * ang)


def tf(tcss, ang):
    return "translate(%spx,%spx)rotate(%sdeg)" % (num(tcss[0], 2), num(tcss[1], 2), num(ang, 3))


TF_I = "translate(0px,0px)rotate(0deg)"


def drift_at(walk, t):
    _, xy, _, est, e = walk_at(walk, t)
    R = np.array([[math.cos(e), -math.sin(e)], [math.sin(e), math.cos(e)]])
    return float(e), est - R @ xy


def poly_d(pts):
    out, last = [], None
    for p in pts:
        q = (num(p[0]), num(p[1]))
        if q != last:
            out.append(q)
            last = q
    if len(out) == 1:
        out.append(out[0])
    return f"M{out[0][0]} {out[0][1]}L" + " ".join(f"{a} {b}" for a, b in out[1:])


def dots_d(pts):
    return "".join(f"M{num(p[0])} {num(p[1])}h.1" for p in pts)


def rings_d(pts, r):
    rr = num(r)
    return "".join(f"M{num(p[0] - r)} {num(p[1])}a{rr} {rr} 0 1 0 {num(2 * r)} 0a{rr} {rr} 0 1 0 {num(-2 * r)} 0"
                   for p in pts)


# stroke widths (desktop, viewBox px)
SW = dict(p=2.6, w=1.9, t=1.6, n=1.2, e=3.4, f=1.5)
NODE_R = 2.5
NODE_GAP = NODE_R + 0.5     # the trajectory is broken this far either side of a node centre
HALO_R, HALO_SW = 5.0, 2.0  # endpoint rings: radius at rest, rendered stroke
EDGE_DASH, EDGE_GAP = 5.0, 3.5


def cut_intervals(poly, nodes, d):
    """Arc intervals of the polyline within distance d of any node centre."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    L = arc[-1]
    sg = np.linspace(0, L, max(int(L / 0.05), 2) + 1)
    q = np.stack([np.interp(sg, arc, poly[:, 0]), np.interp(sg, arc, poly[:, 1])], 1)
    cuts = []
    for p in nodes:
        m = np.linalg.norm(q - p, axis=1) < d
        if not m.any():
            continue
        idx = np.where(m)[0]
        # contiguous runs
        br = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[br + 1]]
        ends = np.r_[idx[br], idx[-1]]
        for a, b in zip(starts, ends):
            cuts.append((sg[max(a - 1, 0)] if a > 0 else 0.0, sg[min(b + 1, len(sg) - 1)] if b < len(sg) - 1 else L))
    return arc, L, cuts


def split_poly(poly, nodes, d):
    """Pieces [(a0, a1, points)] of the polyline with the node neighbourhoods removed."""
    arc, L, cuts = cut_intervals(poly, nodes, d)
    keep = [(0.0, L)]
    for a, b in sorted(cuts):
        new = []
        for k0, k1 in keep:
            if b <= k0 or a >= k1:
                new.append((k0, k1))
                continue
            if a > k0:
                new.append((k0, a))
            if b < k1:
                new.append((b, k1))
        keep = new
    out = []
    for k0, k1 in keep:
        if k1 - k0 < 0.25:
            continue

        def at(s):
            return np.array([np.interp(s, arc, poly[:, 0]), np.interp(s, arc, poly[:, 1])])
        inner = [poly[j] for j in range(len(poly)) if k0 + 1e-6 < arc[j] < k1 - 1e-6]
        out.append((k0, k1, np.array([at(k0)] + inner + [at(k1)])))
    return out


# ----------------------------------------------------------------------------
# build everything once (theme independent)
# ----------------------------------------------------------------------------
def fit_screen(world_pts, x_right, x_left_min):
    x0, y0 = world_pts.min(0)
    x1, y1 = world_pts.max(0)
    sc = min((x_right - x_left_min) / (x1 - x0), MAP_H_MAX / (y1 - y0))
    o = np.array([x_right - sc * x1, H / 2 + MAP_DY + sc * (y0 + y1) / 2])
    return Screen(sc, o)


def build(x_left_min, text):
    rng = np.random.default_rng(20260922)
    occ, cand, strength, kinds, drawn = build_world(rng)
    drawn_kind = ~np.isin(kinds, NODRAW_KINDS)
    x0, y0 = cand[drawn_kind].min(0)
    x1, y1 = cand[drawn_kind].max(0)
    x_right = MAP_RIGHT_INK - SW["p"] / 2
    sc_est = min((x_right - x_left_min) / (x1 - x0 + 0.3), MAP_H_MAX / (y1 - y0 + 0.3))
    walk = build_walk(rng, sc_est)
    walk["gap"] = float(np.linalg.norm(walk["xy"][0] - walk["xy"][-1]))
    walk["dpsi"] = float(np.degrees(walk["psi"][-1] - walk["psi"][0]))
    first, first_cam, ok, dups, occAB = simulate(walk, occ, cand, kinds)

    pts = cand.copy()
    ray = pts - first_cam
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-6)
    depth_sig = np.where(rng.uniform(0, 1, len(pts)) < 0.05, 0.35, 0.06)
    pts += ray * (rng.normal(0, 1, len(pts)) * depth_sig)[:, None]
    pts += rng.normal(0, 0.02, pts.shape)
    dup_q = {i: cand[i] + rng.normal(0, 0.05, 2) for i, _ in dups}

    near = seg_dist(pts, drawn) <= OUTSIDE_TOL_M
    thin = np.random.default_rng(7).uniform(0, 1, len(pts)) < THIN_KEEP
    keep = drawn_kind & near & thin
    n_seen = int(ok.sum())
    ok = ok & keep
    dups = [(i, tr) for i, tr in dups if keep[i] and kinds[i] in GHOST_KINDS]

    scr = fit_screen(np.concatenate([pts[ok], walk["xy"]]), x_right, x_left_min)
    pivot = scr.pt(walk["xy"][0])

    def R(p):
        return scr.pt(p) - pivot

    tb, dt = bucket_times()
    n_b = len(tb) - 1
    strong = strength > 0.38

    def rev_bucket(t):
        return max(int(round((t - T_MOVE[0]) / dt)), 0)

    rb = np.array([rev_bucket(f) if o else -1 for f, o in zip(first, ok)])
    dup_b = [rev_bucket(tr) for _, tr in dups]
    n_all = max(n_b, int(rb.max()) + 1, max(dup_b, default=0) + 1)
    buckets = []
    for i in range(n_all):
        t0 = tb[i] if i <= n_b else tb[-1]
        t1 = tb[i + 1] if i < n_b else t0
        e, tw = drift_at(walk, 0.5 * (t0 + t1))
        tcss, ang = scr.rigid(e, tw, pivot)
        sel = rb == i
        b = dict(i=i, t0=t0, t1=t1, tcss=tcss, ang=ang, sel=np.where(sel)[0],
                 pts=R(pts[sel & strong]), wpts=R(pts[sel & ~strong]), traj=None, nodes=[], dups=[])
        if i < n_b:
            tt = np.linspace(t0, t1, 13)
            s, xy, _, _, _ = walk_at(walk, tt)
            spts = R(xy)
            seg = np.linalg.norm(np.diff(spts, axis=0), axis=1)
            arc = np.concatenate([[0], np.cumsum(seg)])
            b.update(traj=spts, L=float(arc[-1]), arc_mid=float(arc[6]), tmid=float(tt[6]),
                     s0=float(s[0]), s1=float(s[-1]))
        buckets.append(b)
    # pose-graph nodes (keyframes): node 0 = the start pose, then every KF_EVERY m
    fine = np.linspace(T_MOVE[0], T_MOVE[1], 6000)
    fs, fxy, _, _, _ = walk_at(walk, fine)
    next_s, kf0 = 0.0, None
    for k in range(len(fine)):
        if fs[k] >= next_s - 1e-9 and fs[k] < walk["S"] - 2.6:
            bi = min(int((fine[k] - T_MOVE[0]) / dt), n_b - 1)
            b = buckets[bi]
            p = R(fxy[k])
            u = (fs[k] - b["s0"]) / max(b["s1"] - b["s0"], 1e-9)
            b["nodes"].append(dict(p=p, t=float(fine[k]), sn=float(np.clip(u, 0, 1)) * b["L"]))
            if kf0 is None:
                kf0 = (p, bi)
            next_s += KF_EVERY
    all_nodes = np.array([n["p"] for b in buckets for n in b["nodes"]])
    for b in buckets:
        if b["traj"] is not None:
            b["pieces"] = split_poly(b["traj"], all_nodes, NODE_GAP)
    for (i, tr), bi in zip(dups, dup_b):
        buckets[bi]["dups"].append(R(dup_q[i]))

    tc, _, sub = cam_times()
    cam = []
    for t in tc:
        _, xy, psi, _, _ = walk_at(walk, t)
        e, tw = drift_at(walk, t)
        cam.append(dict(t=t, p=R(xy), a=Screen.heading(psi), d=scr.rigid(e, tw, pivot)))
    for j in range(1, len(cam)):
        cam[j]["a"] += 360 * round((cam[j - 1]["a"] - cam[j]["a"]) / 360)
    # start pose seen from the end of the loop: same place, heading unwrapped next to the park heading
    a_start = cam[0]["a"] + 360 * round((cam[-1]["a"] - cam[0]["a"]) / 360)
    model = dict(scr=scr, pivot=pivot, buckets=buckets, cam=cam, cam_sub=sub, kf0=kf0, walk=walk, cand=cand,
                 n_pts=int(ok.sum()), n_seen=n_seen, n_dups=len(dups), first=first, pts_w=pts, ok=ok,
                 occAB=occAB, tb=tb, dt=dt, R=R, a_start=a_start, kinds=kinds)
    model["ph"] = None
    return model


# ----------------------------------------------------------------------------
# a python model of what the browser draws at story time t
# ----------------------------------------------------------------------------
def lin_keys(keys, t):
    ts = np.array([k[0] for k in keys])
    vs = np.array([k[1] for k in keys], float)
    if t <= ts[0]:
        return vs[0]
    if t >= ts[-1]:
        return vs[-1]
    j = int(np.searchsorted(ts, t) - 1)
    u = (t - ts[j]) / (ts[j + 1] - ts[j])
    return vs[j] * (1 - u) + vs[j + 1] * u


def corr_k(t):
    if t <= T_CORR[0]:
        return 1.0
    if t >= T_CORR[1]:
        return 0.0
    return 1.0 - EASE((t - T_CORR[0]) / (T_CORR[1] - T_CORR[0]))


def cam_pose(model, t):
    """Rendered camera: (apex pivot-rel screen pos, heading deg) including drift,
    plus the undrifted pose. Pose keys 8 Hz, drift keys 4 Hz; parked on node 0
    (the start pose) after the walk, through the seam."""
    cams = model["cam"]
    t_end = cams[-1]["t"]
    pk = [(c["t"], [c["p"][0], c["p"][1], c["a"]]) for c in cams]
    x, y, a = lin_keys(pk, t)
    if t < t_end:
        dk = [(c["t"], [c["d"][0][0], c["d"][0][1], c["d"][1]]) for c in cams[::model["cam_sub"]]]
        tx, ty, da = lin_keys(dk, t)
        k = 1.0
    else:
        (tx, ty), da = cams[-1]["d"]
        k = corr_k(t)
    p = apply_tf(np.array([x, y]), (tx, ty), da, k)
    return p, a + k * da, np.array([x, y]), a


def frustum_segments(scale=1.0):
    """The open view cone: two rays from the apex."""
    ta = math.tan(FOV / 2)
    rd = FR_RAY_D * scale
    o = np.zeros(2)
    return [(o, np.array([rd, -rd * ta])), (o, np.array([rd, rd * ta]))]


EASE_FADE_IN = cubic_bezier(.2, .6, .35, 1)
EASE_FADE_OUT = cubic_bezier(.4, 0, .6, 1)


def global_alpha(t):
    if t < T_IN[1]:
        return EASE_FADE_IN(t / T_IN[1])
    if t > T_OUT[0]:
        return 1.0 - EASE_FADE_OUT((t - T_OUT[0]) / (T_OUT[1] - T_OUT[0]))
    return 1.0


def edge_ends(model, t):
    """Constraint edge endpoints (on the two rings) at time t, pivot-relative."""
    A = model["kf0"][0]
    E = cam_pose(model, t)[0]
    v = A - E
    L = float(np.linalg.norm(v))
    u = v / max(L, 1e-9)
    return E + u * HALO_R, A - u * HALO_R, L


def halo_scale(t):
    """Endpoint ring scale: 1 -> PULSE_S (ease-out) -> 1 (ease-in-out), then held."""
    u = t - T_CLOSE
    if u < 0:
        return 1.0
    if u < PULSE[0]:
        return 1 + (PULSE_S - 1) * EASE_OUT(u / PULSE[0])
    if u < PULSE[1]:
        return PULSE_S - (PULSE_S - 1) * EASE_INOUT((u - PULSE[0]) / (PULSE[1] - PULSE[0]))
    return 1.0


def ink_at(model, t, tier="desk"):
    """All animated map/camera ink at time t as (points (N,2) pivot-rel, radius (N,))."""
    m = TIERS[tier]
    P_, R_ = [], []
    k = corr_k(t)
    if global_alpha(t) > 0.001:
        for b in model["buckets"]:
            if t < b["t0"] - 0.05:
                continue
            for key, r in (("pts", SW["p"] / 2 * m["p"]), ("wpts", SW["w"] / 2 * m["w"]),
                           ("dups", SW["p"] / 2 * m["p"])):
                if len(b[key]):
                    if key == "dups" and t > T_FUSE[1]:
                        continue
                    q = apply_tf(np.asarray(b[key]), b["tcss"], b["ang"], k)
                    P_.append(q)
                    R_.append(np.full(len(q), r))
            if b["traj"] is not None:
                q = apply_tf(b["traj"], b["tcss"], b["ang"], k)
                P_.append(q)
                R_.append(np.full(len(q), SW["t"] / 2 * m["t"]))
            if b["nodes"]:
                q = apply_tf(np.asarray([n["p"] for n in b["nodes"]]), b["tcss"], b["ang"], k)
                P_.append(q)
                R_.append(np.full(len(q), NODE_R + SW["n"] / 2 * m["n"]))
    # camera glyph (never faded)
    p, a, _, _ = cam_pose(model, t)
    cs = m["cam"]
    outline = []
    for s0, s1 in frustum_segments(cs):
        for u in np.linspace(0, 1, 12):
            outline.append(s0 + u * (s1 - s0))
    q = p + rot(np.array(outline), a)
    P_.append(q)
    R_.append(np.full(len(q), SW["f"] / 2 * m["f"]))
    P_.append(p[None])
    R_.append(np.array([DOT_R * cs]))
    # loop-closure edge + rings (the current-pose ring and the edge fade out
    # while the edge shrinks, node 0's ring stays until T_EDGE_OUT)
    t_s0, t_s1 = short_times(model)
    if T_CLOSE <= t <= T_EDGE_OUT[1]:
        rr = m["q"] * (HALO_R * halo_scale(t) + HALO_SW / 2)
        P_.append(model["kf0"][0][None])
        R_.append(np.array([rr]))
        if t <= t_s1:
            e0, e1, L = edge_ends(model, t)
            if L > 2 * HALO_R:
                q = e0 + np.linspace(0, 1, 40)[:, None] * (e1 - e0)
                P_.append(q)
                R_.append(np.full(len(q), SW["e"] / 2 * m["e"]))
            P_.append(p[None])
            R_.append(np.array([rr]))
    return np.concatenate(P_), np.concatenate(R_)


def short_times(model):
    """Times at which the constraint (centre to centre) gets shorter than EDGE_SHORT[0] / [1]."""
    if "t_short" not in model:
        A = model["kf0"][0]
        ts = np.arange(T_CORR[0], T_CORR[1] + 1e-9, 0.002)
        L = np.array([np.linalg.norm(A - cam_pose(model, t)[0]) for t in ts])
        model["t_short"] = tuple(float(ts[np.argmax(L < lim)]) for lim in EDGE_SHORT)
        model["L_close"] = float(np.linalg.norm(A - cam_pose(model, T_CLOSE)[0]))
    return model["t_short"]


def rect_dist(pts, rad, bb):
    x0, y0, x1, y1 = bb
    dx = np.maximum(np.maximum(x0 - pts[:, 0], 0), pts[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - pts[:, 1], 0), pts[:, 1] - y1)
    return np.hypot(dx, dy) - rad


def to_vb(model, q, r, tier):
    """pivot-relative ink -> viewBox coordinates for a tier (phones: the #mp transform)."""
    q = q + model["pivot"]
    if tier == "phone":
        s, tx, ty = model["ph"]
        return q * s + np.array([tx, ty]), r * s
    if tier == "mid":
        return q + np.array([0.0, MID_MAP_DY]), r
    return q, r


def clearance(model, bb, tier="desk", step=0.02):
    """Min distance between any animated map/camera ink and the text box(es) bb
    (one rectangle, or a list of rectangles = one per text line), over the loop."""
    bbs = bb if isinstance(bb, list) else [bb]
    best, best_t = 1e9, None
    for t in np.arange(0, P, step):
        q, r = ink_at(model, t, tier)
        q, r = to_vb(model, q, r, tier)
        d = min(rect_dist(q, r, b).min() for b in bbs)
        if d < best:
            best, best_t = d, t
    return best, best_t


def ink_bounds(model, tier="desk", step=0.05, t_range=(0, P)):
    lo, hi = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
    for t in np.arange(t_range[0], t_range[1], step):
        q, r = ink_at(model, t, tier)
        q, r = to_vb(model, q, r, tier)
        lo = np.minimum(lo, (q - r[:, None]).min(0))
        hi = np.maximum(hi, (q + r[:, None]).max(0))
    return lo, hi


def solve_phone(model, text, verbose=True):
    """#mp transform on phones: the map's all-time ink (every t, drift, ghosts and
    the loop-closure glyph included) fits PH_BAND vertically and is centred in
    it, the clean map's right ink edge sits at MAP_RIGHT_INK, and the text block
    is centred on the clean map. Largest scale that keeps PH_CLEAR_CSS between
    the map/camera ink and the text at the narrowest phone column (PH_MIN_W)."""
    need = PH_CLEAR_CSS * W / PH_MIN_W + 1.0
    model["ph"] = (1.0, 0.0, 0.0)
    lo, hi = ink_bounds(model, "phone", 0.02)
    clo, chi = ink_bounds(model, "phone", 0.1, (T_STATIC + 0.1, T_OUT[0]))
    b0, b1 = PH_BAND
    s = (b1 - b0 - 1.0) / (hi[1] - lo[1])
    for _ in range(80):
        tx = MAP_RIGHT_INK - s * chi[0]
        ty = (b0 + b1) / 2 - s * (lo[1] + hi[1]) / 2
        model["ph"] = (s, tx, ty)
        bb = place_phone_text(text, s * (clo[1] + chi[1]) / 2 + ty)
        c, ct = clearance(model, bb, "phone", 0.05)
        ok = c >= need and bb[1] >= b0 and bb[3] <= b1 and s * hi[0] + tx <= W - 0.3
        if verbose:
            print("  phone map scale %.3f -> clearance %.1f vb (%.1f CSS px at %d, %.1f at %d) t=%.2f;"
                  " map ink y %.1f..%.1f, text y %.1f..%.1f" % (
                      s, c, c * PH_MIN_W / W, PH_MIN_W, c * PH_REF_W / W, PH_REF_W, ct,
                      s * lo[1] + ty, s * hi[1] + ty, bb[1], bb[3]))
        if ok:
            model["ph_bounds"] = (s * lo + np.array([tx, ty]), s * hi + np.array([tx, ty]))
            return
        s *= 0.99
    raise RuntimeError("phone layout did not converge")


# ----------------------------------------------------------------------------
# asserts: what the viewer sees is physically consistent
# ----------------------------------------------------------------------------
def check_model(model, verbose=True):
    scr = model["scr"]
    A, B = model["occAB"]
    first, ok = model["first"], model["ok"]
    tol = math.radians(2.0)
    worst_bear, n_checked, n_occ, worst_lag = 0.0, 0, 0, 0.0
    rev_bear, n_rev_out = 0.0, 0
    dt = model["dt"]
    for b in model["buckets"]:
        for i in b["sel"]:
            tf_ = first[i]
            lag = b["t0"] - max(tf_, T_MOVE[0])
            assert abs(lag) <= dt / 2 + 1e-6, ("reveal more than 1/8 s off the observation", i, lag)
            worst_lag = max(worst_lag, abs(lag))
            _, _, cp_true, ca = cam_pose(model, max(tf_, 0.0))
            cw = scr.to_world(cp_true + model["pivot"])
            cand = model["cand"][i]
            dc = cand - cw
            bear = abs(math.remainder(math.atan2(dc[1], dc[0]) - math.radians(-ca), 2 * math.pi))
            dist = float(np.hypot(*dc))
            assert bear <= FOV / 2 + tol, ("landmark lit outside FOV", i, tf_, math.degrees(bear))
            assert dist <= RANGE + 0.3, ("landmark lit beyond range", i, dist)
            if occluded(cw, cand[None], A, B)[0]:
                n_occ += 1
            worst_bear = max(worst_bear, bear)
            n_checked += 1
            _, _, cp2, ca2 = cam_pose(model, b["t0"] + RAMP / 2)
            cw2 = scr.to_world(cp2 + model["pivot"])
            d2 = cand - cw2
            bear2 = abs(math.remainder(math.atan2(d2[1], d2[0]) - math.radians(-ca2), 2 * math.pi))
            rev_bear = max(rev_bear, bear2)
            n_rev_out += bear2 > FOV / 2 + math.radians(5)
    assert n_occ == 0, ("landmarks lit while occluded from the rendered camera", n_occ)
    ts = np.arange(T_MOVE[0] + 0.3, T_MOVE[1] - 0.3, 0.01)
    worst_h = 0.0
    for t in ts:
        _, _, p0, _ = cam_pose(model, t - 0.03)
        _, _, p1, a = cam_pose(model, t + 0.03)
        v = p1 - p0
        if np.hypot(*v) < 0.3:
            continue
        tang = math.degrees(math.atan2(v[1], v[0]))
        _, _, _, a = cam_pose(model, t)
        err = abs((a - tang + 180) % 360 - 180)
        worst_h = max(worst_h, err)
    assert worst_h < 6.0, ("frustum heading off the trajectory tangent", worst_h)
    # camera motion caps
    ts = np.arange(0, T_MOVE[1] + 0.2, 0.01)
    poses = np.array([np.r_[cam_pose(model, t)[2], cam_pose(model, t)[3] % 360] for t in ts])
    ang = np.unwrap(np.radians(poses[:, 2]))
    v = np.hypot(*np.diff(poses[:, :2], axis=0).T) / 0.01
    w = np.degrees(np.abs(np.diff(ang))) / 0.01
    assert v.max() <= CAP_SPEED_PX + 1e-6, ("camera too fast", v.max())
    assert w.max() <= CAP_OMEGA_DEG + 1e-6, ("camera turns too fast", w.max())
    # the camera is parked on the start pose across the seam: same place, same heading, at rest
    p_end, a_end, _, _ = cam_pose(model, P - 1e-3)
    p_0, a_0, _, _ = cam_pose(model, 0.0)
    seam_dp = float(np.linalg.norm(p_end - p_0))
    seam_da = abs((a_end - a_0 + 180) % 360 - 180)
    assert seam_dp < 0.01 and seam_da < 0.01, ("camera jumps at the seam", seam_dp, seam_da)
    assert np.linalg.norm(model["kf0"][0] - p_0) < 1e-6, "node 0 is not the start pose"
    v_pre_end = np.linalg.norm(cam_pose(model, P - 1e-3)[0] - cam_pose(model, P - 0.101)[0]) / 0.1
    v_walk0 = np.linalg.norm(cam_pose(model, 0.101)[0] - cam_pose(model, 0.001)[0]) / 0.1
    corr_v = 0.0
    tc = np.linspace(T_CORR[0], T_CORR[1], 141)
    for b in model["buckets"]:
        q = np.concatenate([np.asarray(b[k]).reshape(-1, 2) for k in ("traj", "pts", "wpts", "dups")
                            if b[k] is not None and len(b[k])])
        pos = np.array([apply_tf(q, b["tcss"], b["ang"], corr_k(t)) for t in tc])
        sp = np.linalg.norm(np.diff(pos, axis=0), axis=2) / (tc[1] - tc[0])
        corr_v = max(corr_v, float(sp.max()))
    assert corr_v <= CAP_CORR_PX, ("correction too fast", corr_v)
    # the constraint edge vs the view cone at the moment it fires: angle between the
    # edge and the camera heading (inside the cone when < FOV/2)
    e_ang = []
    for t in np.arange(T_CLOSE, T_EDGE_OUT[0], 0.05):
        p, a, _, _ = cam_pose(model, t)
        v_ = model["kf0"][0] - p
        e_ang.append(abs((math.degrees(math.atan2(v_[1], v_[0])) - a + 180) % 360 - 180))
    if verbose:
        print("check: %d landmarks lit when observed inside FOV (worst bearing %.1f of %.1f deg, %d occluded,"
              " |reveal - observation| max %.3f s; at 50%% of the ramp worst bearing %.1f deg, %d beyond FOV+5),"
              " heading err max %.2f deg, cam speed max %.0f px/s, yaw rate max %.0f deg/s, correction speed max"
              " %.1f px/s; seam: camera parked (|dp| %.4f px, |da| %.4f deg), mean speed last 0.1 s %.2f / first 0.1 s %.1f px/s;"
              " edge vs heading %.1f..%.1f deg"
              " (cone half-angle %.0f)" % (
                  n_checked, math.degrees(worst_bear), math.degrees(FOV / 2), n_occ, worst_lag,
                  math.degrees(rev_bear), n_rev_out, worst_h, v.max(), w.max(), corr_v, seam_dp, seam_da, v_pre_end, v_walk0,
                  min(e_ang), max(e_ang), math.degrees(FOV / 2)))
    return dict(speed=float(v.max()), omega=float(w.max()), corr=corr_v, heading=worst_h, n_occ=n_occ,
                rev_out=int(n_rev_out), edge_ang=(min(e_ang), max(e_ang)), seam_v=(v_pre_end, v_walk0))


# ----------------------------------------------------------------------------
# SVG emission
# ----------------------------------------------------------------------------
def hexa(c, a):
    return c + "%02x" % int(round(255 * a))


def recency(t_rev):
    """Landmark colour stops after it lights up: (t, colour-key). Accent (60 %) only
    briefly; everything is grey by T_GREY and settled by T_CORR[1]."""
    t_on = t_rev + RAMP
    t_acc = t_on + ACC_HOLD
    t_new = t_acc + ACC_FADE
    t_old = min(t_new + NEW_HOLD, T_CORR[1])
    if abs(t_old - T_CORR[0]) < 0.06:          # keep clear of the eased correction keyframe
        t_old = T_CORR[0] - 0.06
    if t_new > T_GREY:
        # too late for an accent flash: light up straight into bright neutral
        t_old = max(t_old, t_on + 0.3)
        if abs(t_old - T_CORR[0]) < 0.06:
            t_old = T_CORR[0] + 0.06
        return [(t_rev, "new0"), (t_on, "new"), (t_old, "old")]
    return [(t_rev, "acc0"), (t_on, "acc"), (t_acc, "acc"), (t_new, "new"), (t_old, "old")]


def _check_recency():
    """No accent after T_GREY (= T_CLOSE - 0.1 s) for any landmark or ghost, whatever
    its reveal time; everything settled by the end of the correction."""
    for t_rev in np.arange(T_MOVE[0], T_MOVE[1] + 0.5, 0.01):
        st = recency_safe(float(t_rev))
        acc_end = max((b[0] for a, b in zip(st, st[1:]) if a[1] == "acc"), default=-1)
        assert acc_end <= T_GREY + 1e-9, (t_rev, st)
        assert st[-1][0] <= T_CORR[1] + 0.07, (t_rev, st)


def recency_safe(t_rev):
    """recency() with every stop kept >= 0.06 s away from the eased T_CORR[0] key."""
    out = []
    for tt, key in recency(t_rev):
        if abs(tt - T_CORR[0]) < 0.06:
            tt = T_CORR[0] - 0.06 if tt < T_CORR[0] else T_CORR[0] + 0.06
        out.append((tt, key))
    assert all(a[0] < b[0] for a, b in zip(out, out[1:])), out
    return out


def emit(model, theme, text):
    C = THEMES[theme]
    fname, ftag = text["fname"], text["ftag"]
    name_d = fname.outline(text["name_items"], NAME_SIZE, text["name_x"], NAME_BASE)
    tag_items, tag_x = text["tag_items"], text["tag_x"]
    words = [it for it in tag_items if it[0] != "·"]
    seps = [it for it in tag_items if it[0] == "·"]
    tag_d = ftag.outline(words, TAG_SIZE, tag_x, TAG_BASE)
    sep_d = ftag.outline(seps, TAG_SIZE, tag_x, TAG_BASE)
    # phones: two name lines + one tagline line
    n2, t2w, t2s = "", "", ""
    for ln in text["ph"]:
        f = fname if ln["kind"] == "n" else ftag
        if ln["kind"] == "n":
            n2 += f.outline(ln["items"], ln["size"], ln["x"], ln["base"])
        else:
            t2w += f.outline([it for it in ln["items"] if it[0] != "·"], ln["size"], ln["x"], ln["base"])
            t2s += f.outline([it for it in ln["items"] if it[0] == "·"], ln["size"], ln["x"], ln["base"])

    pv = model["pivot"]
    ACC, PT, PTN, BG = C["acc"], C["pt"], C["pt_new"], C["bg"]
    COL = {"acc0": hexa(ACC, 0), "acc": hexa(ACC, FRESH_A), "new0": hexa(PTN, 0), "new": PTN, "old": PT}
    PTN0 = hexa(PTN, 0)
    css, body = [], []
    css.append(".a{animation:%ss linear -%ss infinite}" % (num(P, 2), num(PHASE, 2)))
    css.append(".z{transform-box:view-box;transform-origin:0 0}")

    def stroke_rules(mult, skip_unit=False):
        r = (".p{stroke-width:%s}.w{stroke-width:%s}.t{stroke-width:%s}.n{stroke-width:%s}#e{stroke-width:%s}"
             ".f{stroke-width:%s}.g{transform:scale(%s)}.q{transform:scale(%s)}" % (
                 num(SW["p"] * mult["p"], 2), num(SW["w"] * mult["w"], 2), num(SW["t"] * mult["t"], 2),
                 num(SW["n"] * mult["n"], 2), num(SW["e"] * mult["e"], 2),
                 num(SW["f"] * mult["f"] / mult["cam"], 2), num(mult["cam"], 3), num(mult["q"], 3)))
        if skip_unit:
            r = r.replace(".g{transform:scale(1)}", "").replace(".q{transform:scale(1)}", "")
        return r
    css.append(".p,.w,.t,.f{stroke-linecap:round}.t,.f{stroke-linejoin:round}.w{stroke-opacity:%s}" % num(C["w_op"], 2))
    css.append(stroke_rules(TIERS["desk"], True))
    css.append("#ph{display:none}")
    # global map fade: ease-out in, gentle ease-in-out out
    css.append(Keyframes("m").at(0, opacity=0, animation_timing_function=FADE_IN_CSS).at(T_IN[1], opacity=1)
               .at(T_OUT[0], opacity=1, animation_timing_function=FADE_OUT_CSS).at(P, opacity=0).css())
    css.append(".h,#e,#o{opacity:0}")

    # ---- buckets: one group, ONE animation per 1/4 s of walking: drift (eased to
    # identity at loop closure), landmark light-up + recency colour (inherited
    # `stroke`; the dots have no stroke of their own) and the trajectory draw-in
    # (inherited `stroke-dashoffset`; only the trajectory pieces and node rings
    # have a dash array). The trajectory is broken around every node (no fills).
    for b in model["buckets"]:
        i, t0 = b["i"], b["t0"]
        D = tf(b["tcss"], b["ang"])
        kf = Keyframes("k%d" % i)
        kf.at(0, transform=D)
        kf.at(T_CORR[0], transform=D + ";animation-timing-function:" + EASE_CSS)
        kf.at(T_CORR[1], transform=TF_I)
        kf.at(P, transform=TF_I)
        if len(b["pts"]) or len(b["wpts"]):
            stops = recency_safe(t0)
            for tt, _ in stops:
                if abs(tt - T_CORR[0]) < 0.01:      # would inherit the eased timing function
                    raise AssertionError("keyframe collision")
            kf.at(0, stroke=COL[stops[0][1]])
            for tt, key in stops:
                kf.at(tt, stroke=COL[key])
            kf.at(P, stroke=PT)
        parts = []
        if len(b["wpts"]):
            parts.append(f'<path class="w" d="{dots_d(b["wpts"])}"/>')
        if len(b["pts"]):
            parts.append(f'<path class="p" d="{dots_d(b["pts"])}"/>')
        if b["dups"]:
            # ghost copies: own colour schedule (grey by T_GREY), turn bright
            # neutral ("matched", no accent) as the correction starts, fuse (fade)
            # once aligned; hidden when static
            gk = Keyframes("g%d" % i)
            stops = recency_safe(t0)
            gk.at(0, stroke=COL[stops[0][1]])
            for tt, key in stops:
                if tt < T_CORR[0] - 0.05:
                    gk.at(tt, stroke=COL[key])
            last = [key for tt, key in stops if tt < T_CORR[0] - 0.05][-1]
            gk.at(T_CORR[0], stroke=COL[last]).at(T_CORR[0] + T_MATCH, stroke=PTN)
            gk.at(T_FUSE[0], stroke=PTN).at(T_FUSE[1], stroke=PTN0).at(P, stroke=PTN0)
            css.append(gk.css())
            parts.append(f'<path class="p a" style="animation-name:g{i}" stroke="none" d="{dots_d(b["dups"])}"/>')
        attrs = f' stroke="{PT}"'
        if b["traj"] is not None:
            L2 = b["L"] + 2
            Cn = 2 * math.pi * NODE_R
            G = num(L2 + Cn + 2)
            o_pre = L2 + Cn + 1
            if t0 > 0.03:
                kf.at(0, stroke_dashoffset=num(o_pre)).at(t0 - 0.03, stroke_dashoffset=num(o_pre))
            kf.at(t0, stroke_dashoffset=num(L2)).at(b["tmid"], stroke_dashoffset=num(L2 - b["arc_mid"]))
            kf.at(b["t1"], stroke_dashoffset=0).at(P, stroke_dashoffset=0)
            for a0, a1, pp in b["pieces"]:
                parts.append(f'<path class="t" stroke="{C["traj"]}" stroke-dasharray="{num(L2 - a0)} {G}" '
                             f'd="{poly_d(pp)}"/>')
            for nd in b["nodes"]:
                Dn = L2 - nd["sn"] + Cn
                parts.append(f'<path class="n" stroke="{C["traj"]}" stroke-dasharray="{num(Dn)} {G}" '
                             f'd="{rings_d([nd["p"]], NODE_R)}"/>')
        css.append(kf.css())
        body.append(f'<g class="a z" style="animation-name:k{i}"{attrs}>' + "".join(parts) + "</g>")

    cams = model["cam"]
    sub = model["cam_sub"]
    A = model["kf0"][0]
    (dT, dA) = cams[-1]["d"]
    E_true = cams[-1]["p"]
    t_s0, t_s1 = short_times(model)

    # ---- loop-closure constraint: dashed accent edge between the two rings
    # (current pose -> node 0). A unit segment 'M0 0H1' under
    # translate/rotate/scaleX; the dash pattern is divided by the length so dashes
    # keep their size while the edge shortens during the correction. It fades
    # out (with the current-pose ring) once it gets short; below that length its
    # direction is frozen (it is invisible, and the direction becomes undefined).
    ek = Keyframes("e")
    nsub = 14
    first_da = None
    last_dir = [None]

    def edge_tf(E):
        v = A - E
        L = float(np.linalg.norm(v))
        if L > EDGE_SHORT[1] - 1 or last_dir[0] is None:
            last_dir[0] = v / max(L, 1e-9)
        u = last_dir[0]
        e0 = E + u * HALO_R
        Lv = max(L - 2 * HALO_R, 0.5)
        a = math.degrees(math.atan2(u[1], u[0]))
        tr = "translate(%spx,%spx)rotate(%sdeg)scaleX(%s)" % (num(e0[0], 2), num(e0[1], 2), num(a, 2), num(Lv, 2))
        da = "%s %s" % (num(EDGE_DASH / Lv, 4), num(EDGE_GAP / Lv, 4))
        return tr, da
    for j in range(nsub + 1):
        u = j / nsub
        k = 1 - EASE(u)
        tr, da = edge_tf(apply_tf(E_true, dT, dA, k))
        t = T_CORR[0] + u * (T_CORR[1] - T_CORR[0])
        if j == 0:
            first_da = da
            ek.at(0, opacity=0, transform=tr, stroke_dasharray=da)
            ek.at(T_CLOSE, opacity=0)
            ek.at(T_CLOSE + T_EDGE_IN, opacity=1)
        ek.at(t, transform=tr, stroke_dasharray=da)
    ek.at(t_s0, opacity=1).at(t_s1, opacity=0)
    ek.at(P, opacity=0, transform=tr, stroke_dasharray=da)
    css.append(ek.css())

    # ---- endpoint rings: pulse 1x -> 2x -> 1x, then held; stroke-width divided
    # by the scale at every stop (constant on screen). h0 = node 0 (held until
    # T_EDGE_OUT), h1 = current pose (fades with the edge once it gets short).
    def ring_kf(name, fade):
        hk = Keyframes(name).at(0, opacity=0, transform="scale(1)", stroke_width=num(HALO_SW, 3))
        hk.at(T_CLOSE, opacity=0, transform="scale(1)", stroke_width=num(HALO_SW, 3))
        for u in np.linspace(0, 1, 5):
            t = T_CLOSE + u * PULSE[0]
            sc_ = halo_scale(t)
            hk.at(t, transform="scale(%s)" % num(sc_, 3), stroke_width=num(HALO_SW / sc_, 3))
        for u in np.linspace(0, 1, 6)[1:]:
            t = T_CLOSE + PULSE[0] + u * (PULSE[1] - PULSE[0])
            sc_ = halo_scale(t)
            hk.at(t, transform="scale(%s)" % num(sc_, 3), stroke_width=num(HALO_SW / sc_, 3))
        hk.at(T_CLOSE + 0.06, opacity=1).at(fade[0], opacity=1).at(fade[1], opacity=0)
        hk.at(P, opacity=0, transform="scale(1)", stroke_width=num(HALO_SW, 3))
        return hk.css()
    css.append(ring_kf("h0", T_EDGE_OUT))
    css.append(ring_kf("h1", (t_s0, t_s1)))
    # ---- node 0 turns accent while the constraint is shown
    css.append(Keyframes("o").at(0, opacity=0).at(T_CLOSE, opacity=0).at(T_CLOSE + 0.2, opacity=1)
               .at(T_EDGE_OUT[0], opacity=1).at(T_EDGE_OUT[1], opacity=0).at(P, opacity=0).css())
    # ---- the view cone steps out while the loop-closure glyph is shown
    css.append(Keyframes("fz").at(0, opacity=1).at(T_CLOSE, opacity=1).at(T_CLOSE + 0.25, opacity=CONE_DIM)
               .at(T_EDGE_OUT[0], opacity=CONE_DIM).at(T_EDGE_OUT[1], opacity=1).at(P, opacity=1).css())

    # ---- camera: outer drift group (4 Hz keys) + inner pose group (8 Hz keys),
    # parked on node 0 (= the start pose) from the end of the walk through the seam
    def pose(p, a):
        return "translate(%spx,%spx)rotate(%sdeg)" % (num(p[0], 2), num(p[1], 2), num(a, 2))
    cd = Keyframes("cd")
    cp = Keyframes("cp")
    cd.at(0, transform=TF_I)
    for c in cams[::sub]:
        cd.at(c["t"], transform=tf(*c["d"]))
    for c in cams:
        cp.at(c["t"], transform=pose(c["p"], c["a"]))
    park = pose(cams[-1]["p"], cams[-1]["a"])
    assert park == pose(cams[0]["p"], model["a_start"]), (park, pose(cams[0]["p"], model["a_start"]))
    cp.at(P, transform=park)
    cd.at(T_CORR[0], transform=tf(dT, dA) + ";animation-timing-function:" + EASE_CSS)
    cd.at(T_CORR[1], transform=TF_I).at(P, transform=TF_I)
    css.append(cd.css())
    css.append(cp.css())
    css.append("#ci{transform:%s}" % park)
    css.append("@media (prefers-reduced-motion:reduce){.a{animation:none!important}}")

    # ---- narrower images (cascading max-width queries, widest first): tapered
    # tagline, thicker strokes at <= MID_MAX_W, then the stacked phone layout
    tcy = text["tag_cap_top"]
    tx0 = text["tag_bb"][0]
    for bd in text["bands"]:
        rules = ""
        if bd["wmax"] == MID_MAX_W:
            rules += stroke_rules(TIERS["mid"]) + "#mp{transform:translate(0px,%spx)}" % num(MID_MAP_DY, 2)
        rules += "#nm{transform:translate(0px,%spx)}#tg{transform:translate(%spx,%spx)scale(%s)translate(%spx,%spx)}" % (
            num(-bd["lift"], 2), num(tx0, 2), num(tcy - bd["lift"], 2), num(bd["m"], 3), num(-tx0, 2), num(-tcy, 2))
        css.append("@media (max-width:%dpx){%s}" % (bd["wmax"], rules))
    s, tx, ty = model["ph"]
    css.append("@media (max-width:%dpx){%s#nm,#tg{display:none}#ph{display:inline}"
               "#mp{transform:translate(%spx,%spx)scale(%s)}}" % (
                   PH_MAX_W, stroke_rules(TIERS["phone"]), num(tx, 2), num(ty, 2), num(s, 4)))
    yc = text["ph_yc"]
    subs = [(PH_MAX_W, PH_SUB_TOP)] + list(PH_SUB)
    for wmax, m in subs:
        rule = ("#ph{transform:translate(%spx,%spx)scale(%s)translate(%spx,%spx)}" % (
            num(X_INK, 2), num(yc, 2), num(m, 3), num(-X_INK, 2), num(-yc, 2))) if m != 1 else "#ph{transform:none}"
        if wmax == PH_MAX_W:
            css[-1] = css[-1][:-1] + rule + "}"
        else:
            css.append("@media (max-width:%dpx){%s}" % (wmax, rule))

    fr = frustum_segments()
    rays_d = "M%s %sL0 0L%s %s" % (num(fr[0][1][0]), num(fr[0][1][1]), num(fr[1][1][0]), num(fr[1][1][1]))
    ring = (f'<g class="q z"><circle class="h a z" style="animation-name:%s" r="{num(HALO_R)}" '
            f'stroke="{ACC}"/></g>')
    camera = ('<g class="a z" style="animation-name:cd">'
              f'<g transform="translate({num(E_true[0], 2)} {num(E_true[1], 2)})">' + ring % "h1" + '</g>'
              '<g id="ci" class="a z" style="animation-name:cp"><g class="g z">'
              f'<path class="a" style="animation-name:fz" fill="{ACC}" fill-opacity="{num(C["wedge_op"], 2)}" d="{rays_d}Z"/>'
              f'<path class="f a" style="animation-name:fz" stroke="{ACC}" d="{rays_d}"/>'
              f'<circle r="{num(DOT_R)}" fill="{ACC}"/></g></g></g>')
    edge_svg = (f'<path id="e" class="a z" style="animation-name:e" d="M0 0H1" stroke="{ACC}" '
                f'stroke-dasharray="{first_da}"/>')
    node0 = (f'<g transform="translate({num(A[0], 2)} {num(A[1], 2)})">' + ring % "h0" +
             f'<circle id="o" class="a n" style="animation-name:o" r="{num(NODE_R)}" stroke="{ACC}"/></g>')

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" fill="none" '
           'role="img" aria-label="%s: %s">' % (W, H, W, H, NAME, TAGLINE.replace("·", "&#183;")),
           "<title>%s</title>" % NAME,
           "<style>" + "".join(css) + "</style>",
           f'<path id="nm" class="z" fill="{C["name"]}" d="{name_d}"/>',
           f'<g id="tg" class="z"><path fill="{C["tag"]}" d="{tag_d}"/><path fill="{ACC}" d="{sep_d}"/></g>',
           f'<g id="ph" class="z"><path fill="{C["name"]}" d="{n2}"/>'
           f'<path fill="{C["tag"]}" d="{t2w}"/><path fill="{ACC}" d="{t2s}"/></g>',
           '<g id="mp" class="z">',
           f'<g transform="translate({num(pv[0], 2)} {num(pv[1], 2)})">',
           '<g class="a" style="animation-name:m">' + "".join(body) + "</g>",
           camera + edge_svg + node0,
           "</g></g></svg>"]
    return "\n".join(x for x in svg if x)


def solve_layout(text, verbose=True):
    """Desktop: the map is right-aligned at MAP_RIGHT_INK and as large as MAP_H_MAX
    allows; its left bound only moves right if the clearance to the name needs it."""
    x_left = text["name_bb"][2] + CLEAR_MIN + 30
    for _ in range(40):
        model = build(x_left, text)
        c_desk, t_desk = clearance(model, text["name_bb"], "desk", 0.05)
        if verbose:
            print("  map-left bound %.1f -> scale %.2f px/m, clearance desktop %.1f (t=%.2f)" % (
                x_left, model["scr"].sc, c_desk, t_desk))
        need = CLEAR_MIN + 4 - c_desk
        if need <= 0.5:
            return model, x_left
        x_left += max(1.0, need)
    raise RuntimeError("layout did not converge")


def tier_at(w):
    """(layout, stroke tier, text multiplier) for an <img> width w (CSS px)."""
    if w <= PH_MAX_W:
        m = PH_SUB_TOP
        for wmax, mm in PH_SUB:
            if w <= wmax:
                m = mm
        return "phone", "phone", m
    m, tier = 1.0, "desk"
    for bd in text_bands_cache:
        if w <= bd["wmax"]:
            m, tier = bd["m"], bd["tier"]
    return "line", tier, m


text_bands_cache = []


def font_px(w):
    """Effective (em) font sizes in CSS px of the name and the tagline at <img> width w."""
    lay, _, m = tier_at(w)
    k = w / W
    if lay == "phone":
        return PH_NAME_SIZE * m * k, PH_TAG_SIZE * m * k
    return NAME_SIZE * k, TAG_SIZE * m * k


def main():
    text = text_layout()
    text_bands_cache[:] = text["bands"]
    model, x_left = solve_layout(text)
    solve_phone(model, text)
    chk = check_model(model)
    _check_recency()
    c_desk, t_desk = clearance(model, text["name_bb"], "desk", 0.01)
    c_bands = [(bd["wmax"], bd["m"]) + clearance(model, bd["bb"], bd["tier"], 0.02) for bd in text["bands"]]
    c_ph, t_ph = clearance(model, text["ph_text_bb"], "phone", 0.01)
    sizes = {}
    for theme in ("dark", "light"):
        svg = emit(model, theme, text)
        for bad in ("<text", "<script", "foreignObject", "<image", "http://", "https://", "@import", "url(", "data:"):
            if bad in svg.replace('xmlns="http://www.w3.org/2000/svg"', ""):
                raise AssertionError("forbidden construct " + bad)
        path = os.path.join(HERE, f"banner-{theme}.svg")
        with open(path, "w") as fh:
            fh.write(svg)
        sizes[theme] = len(svg.encode())
        n_anim = len(re.findall(r'animation-name:', svg))
        print(f"{path}: {len(svg.encode())} bytes, {svg.count('<')} tags, {n_anim} animated elements, "
              f"{svg.count('@keyframes')} @keyframes")
        assert len(svg.encode()) <= 120 * 1024
    w = model["walk"]
    err = w["est"][-1] - w["xy"][-1]
    bb = text["name_bb"]
    lo, hi = ink_bounds(model, "desk", 0.05)
    loc, hic = ink_bounds(model, "desk", 0.1, (T_STATIC + 0.1, T_OUT[0]))
    lop, hip = ink_bounds(model, "phone", 0.02)
    lom, him = ink_bounds(model, "mid", 0.05)
    t_s0, t_s1 = short_times(model)
    print("landmarks %d drawn (of %d observed), ghost dups %d, path %.1f m, end error %.2f m, final yaw err %.2f deg,"
          " last bucket drift %.2f deg, P %.2f s, end gap %.4f m, end-start heading %.4f deg" % (
              model["n_pts"], model["n_seen"], model["n_dups"], w["S"], np.linalg.norm(err),
              math.degrees(w["e"][-1]), model["buckets"][-1]["ang"], P, w["gap"], w["dpsi"]))
    print("loop closure: edge %.1f vb at T_CLOSE; current ring + edge fade %.2f..%.2f s (edge %g..%g vb)" % (
        model["L_close"], t_s0, t_s1, *EDGE_SHORT))
    print("name bbox (%.1f, %.1f)-(%.1f, %.1f); phone name bb %s, text bb %s" % (
        bb[0], bb[1], bb[2], bb[3], [round(v, 1) for v in text["ph_name_bb"]],
        [round(v, 1) for v in text["ph_text_bb"]]))
    print("desktop ink all-time x %.1f..%.1f y %.1f..%.1f; clean hold x %.1f..%.1f y %.1f..%.1f; scale %.2f px/m" % (
        lo[0], hi[0], lo[1], hi[1], loc[0], hic[0], loc[1], hic[1], model["scr"].sc))
    print("mid (<= %d px) ink all-time y %.1f..%.1f" % (MID_MAX_W, lom[1], him[1]))
    print("phone ink all-time x %.1f..%.1f y %.1f..%.1f (band %g..%g); #mp scale %.3f" % (
        lop[0], hip[0], lop[1], hip[1], *PH_BAND, model["ph"][0]))
    print("font sizes (em, CSS px) name / tagline:")
    for wd in (278, 293, 308, 348, 350, 351, 398, 400, 401, 440, 441, 540, 541, 590, 620, 621, 700, 701, 780, 781, 846):
        n_, t_ = font_px(wd)
        print("   %4d px [%s]: %5.1f / %5.1f" % (wd, tier_at(wd)[0] + "/" + tier_at(wd)[1], n_, t_))
    print("CLEARANCE: desktop %.1f vb (t=%.2f); bands %s; phone %.1f vb = %.1f CSS px at 278, %.1f at 308 (t=%.2f)" % (
        c_desk, t_desk, ", ".join("<=%d x%.2f: %.1f (t=%.2f)" % c for c in c_bands),
        c_ph, c_ph * PH_MIN_W / W, c_ph * PH_REF_W / W, t_ph))
    assert min([c_desk] + [c[2] for c in c_bands]) >= CLEAR_MIN
    assert c_ph * PH_MIN_W / W >= PH_CLEAR_CSS
    assert PH_BAND[0] - 0.5 <= lop[1] and hip[1] <= PH_BAND[1] + 0.5, (lop, hip)
    return model, text


if __name__ == "__main__":
    main()
