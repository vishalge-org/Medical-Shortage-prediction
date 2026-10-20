"""Generates the original SVG illustrations used by the dashboard (python assets/make_assets.py)."""
from pathlib import Path
OUT = Path(__file__).parent


def capsule(cx, cy, w, h, ang, left, right, uid, shine=0.55, blur=0, shadow=True):
    r, x0, y0 = h / 2, -w / 2, -h / 2
    flt = f' filter="url(#blur{blur})"' if blur else ""
    sh = (f'<ellipse cx="{cx + 14}" cy="{cy + h * 0.95}" rx="{w * 0.42}" ry="{h * 0.16}" fill="#000" opacity=".35" filter="url(#blur12)"/>'
          if shadow and not blur else "")
    return f'''{sh}<g transform="translate({cx} {cy}) rotate({ang})"{flt}>
<defs><clipPath id="c{uid}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="{r}"/></clipPath>
<linearGradient id="l{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{left[0]}"/><stop offset="1" stop-color="{left[1]}"/></linearGradient>
<linearGradient id="r{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{right[0]}"/><stop offset="1" stop-color="{right[1]}"/></linearGradient>
<linearGradient id="s{uid}" x1="0" y1="0" x2="0" y2="1"><stop offset=".55" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".30"/></linearGradient></defs>
<g clip-path="url(#c{uid})"><rect x="{x0}" y="{y0}" width="{w / 2}" height="{h}" fill="url(#l{uid})"/>
<rect x="0" y="{y0}" width="{w / 2}" height="{h}" fill="url(#r{uid})"/>
<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="url(#s{uid})"/>
<rect x="{x0 + r * 0.55}" y="{y0 + h * 0.13}" width="{w - r * 1.1}" height="{h * 0.15}" rx="{h * 0.075}" fill="#fff" opacity="{shine}"/></g>
<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="{r}" fill="none" stroke="#fff" stroke-opacity=".28"/></g>'''


DEFS = '''<defs>
<filter id="blur2" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="2"/></filter>
<filter id="blur6" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="6"/></filter>
<filter id="blur12" x="-100%" y="-300%" width="300%" height="700%"><feGaussianBlur stdDeviation="12"/></filter>
<filter id="blur40" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="40"/></filter>
<radialGradient id="glow" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#34D17F" stop-opacity=".55"/><stop offset="1" stop-color="#34D17F" stop-opacity="0"/></radialGradient>
<radialGradient id="tab" cx=".38" cy=".3" r=".8"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#B9D8C6"/></radialGradient>
</defs>'''

GREEN = ("#3BD98A", "#0E7447")
WHITE = ("#F8FCF9", "#BFD9CA")
AMBER = ("#FFD36B", "#E39A12")
MINT = ("#8FEAB8", "#22A065")


def hero_art():
    wells = []
    empty = {(3, 0), (1, 1), (2, 2)}
    red = {(3, 0), (2, 2)}
    for j in range(3):
        for i in range(4):
            x, y = 205 + 90 * i, 140 + 85 * j
            if (i, j) in empty:
                ring = '#FF6B5E' if (i, j) in red else '#ffffff'
                op = '.95' if (i, j) in red else '.4'
                wells.append(f'<circle cx="{x}" cy="{y}" r="30" fill="#000" fill-opacity=".38" stroke="{ring}" stroke-opacity="{op}" stroke-width="2" stroke-dasharray="5 6"/>')
            else:
                wells.append(f'<circle cx="{x}" cy="{y}" r="31" fill="#000" fill-opacity=".25"/>'
                             f'<circle cx="{x}" cy="{y - 1}" r="25" fill="url(#tab)"/><ellipse cx="{x - 7}" cy="{y - 11}" rx="9" ry="5" fill="#fff" opacity=".8"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 470">{DEFS}
<circle cx="360" cy="235" r="270" fill="url(#glow)"/>
<g transform="rotate(-9 330 240)"><rect x="140" y="62" width="400" height="318" rx="36" fill="#fff" fill-opacity=".07" stroke="#fff" stroke-opacity=".3"/>
<rect x="152" y="74" width="376" height="294" rx="28" fill="none" stroke="#fff" stroke-opacity=".12"/>{''.join(wells)}</g>
{capsule(135, 372, 200, 78, -32, GREEN, WHITE, 'a')}
{capsule(512, 92, 150, 58, 24, AMBER, ("#FFFFFF", "#E9E3CF"), 'b')}
{capsule(566, 338, 112, 44, -58, MINT, WHITE, 'c')}
<circle cx="70" cy="140" r="4" fill="#9BE3B5" opacity=".8"/><circle cx="590" cy="215" r="3" fill="#fff" opacity=".7"/><circle cx="300" cy="30" r="3" fill="#9BE3B5" opacity=".7"/>
</svg>'''


def hero_small():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 300">{DEFS}
<circle cx="250" cy="150" r="170" fill="url(#glow)"/>
{capsule(280, 120, 190, 74, -26, GREEN, WHITE, 'a')}
{capsule(170, 215, 130, 50, 20, AMBER, ("#FFFFFF", "#E9E3CF"), 'b')}
{capsule(345, 235, 90, 36, -50, MINT, WHITE, 'c')}
</svg>'''


def card_art():
    bokeh = ''.join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#{c}" opacity="{o}" filter="url(#blur6)"/>' for x, y, r, c, o in [
        (70, 90, 46, "34D17F", .35), (440, 70, 60, "2FBF71", .28), (460, 330, 38, "9BE3B5", .25), (60, 420, 52, "1D8F5A", .35), (330, 40, 22, "ffffff", .25)])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 640" preserveAspectRatio="xMidYMid slice">{DEFS}
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#14543A"/><stop offset=".55" stop-color="#0A2A1D"/><stop offset="1" stop-color="#06130D"/></linearGradient>
<linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset=".45" stop-color="#04100A" stop-opacity="0"/><stop offset="1" stop-color="#04100A" stop-opacity=".82"/></linearGradient></defs>
<rect width="520" height="640" fill="url(#bg)"/><circle cx="300" cy="220" r="240" fill="url(#glow)"/>{bokeh}
{capsule(420, 470, 260, 100, 28, AMBER, ("#FFFFFF", "#E9E3CF"), 'd', blur=6)}
{capsule(250, 250, 340, 132, -28, GREEN, WHITE, 'e')}
{capsule(150, 430, 220, 86, 16, MINT, WHITE, 'f')}
<rect width="520" height="640" fill="url(#fade)"/></svg>'''



WHITEA = lambda o: f'rgba(255,255,255,{o})'


def wrap(vb, body, extra=""):
    w, h = vb
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">{DEFS}{extra}{body}</svg>'


def hero_risk():
    lens = ('<circle cx="215" cy="125" r="84" fill="#fff" fill-opacity=".07"/>'
            '<circle cx="215" cy="125" r="84" fill="none" stroke="#fff" stroke-opacity=".9" stroke-width="11"/>'
            '<circle cx="215" cy="125" r="94" fill="none" stroke="#fff" stroke-opacity=".18" stroke-width="2"/>'
            '<path d="M285 188 L352 258" stroke="#fff" stroke-opacity=".9" stroke-width="20" stroke-linecap="round"/>'
            '<path d="M285 188 L352 258" stroke="#3BD98A" stroke-opacity=".55" stroke-width="6" stroke-linecap="round"/>')
    gauge = ('<defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#35B26E"/><stop offset=".55" stop-color="#F2C94C"/><stop offset="1" stop-color="#E5534B"/></linearGradient></defs>'
             '<path d="M163 160 A52 52 0 0 1 267 160" fill="none" stroke="#fff" stroke-opacity=".18" stroke-width="12" stroke-linecap="round"/>'
             '<path d="M163 160 A52 52 0 0 1 267 160" fill="none" stroke="url(#gg)" stroke-width="12" stroke-linecap="round" stroke-dasharray="120 200"/>'
             '<circle cx="215" cy="160" r="8" fill="#fff"/><path d="M215 160 L246 126" stroke="#fff" stroke-width="5" stroke-linecap="round"/>')
    body = (f'<circle cx="230" cy="150" r="175" fill="url(#glow)"/>{lens}{gauge}'
            '<circle cx="272" cy="64" r="17" fill="#E5534B"/><path d="M272 55v11M272 70v.5" stroke="#fff" stroke-width="3.2" stroke-linecap="round"/>'
            f'{capsule(95, 232, 120, 46, -30, AMBER, ("#FFFFFF", "#E9E3CF"), "a")}'
            f'{capsule(345, 90, 100, 40, 28, MINT, WHITE, "b")}')
    return wrap((420, 300), body)


def hero_watch():
    rows = ""
    data = [(.92, "#E5534B"), (.74, "#E5534B"), (.55, "#F2C94C"), (.38, "#F2C94C"), (.2, "#35B26E")]
    for k, (v, c) in enumerate(data):
        y = 92 + k * 36
        rows += (f'<circle cx="150" cy="{y}" r="9" fill="#fff" fill-opacity=".85"/>'
                 f'<rect x="172" y="{y - 7}" width="46" height="5" rx="2.5" fill="#fff" fill-opacity=".45"/>'
                 f'<rect x="172" y="{y + 3}" width="30" height="4" rx="2" fill="#fff" fill-opacity=".22"/>'
                 f'<rect x="232" y="{y - 6}" width="96" height="12" rx="6" fill="#fff" fill-opacity=".14"/>'
                 f'<rect x="232" y="{y - 6}" width="{96 * v:.0f}" height="12" rx="6" fill="{c}"/>')
    body = (f'<circle cx="235" cy="150" r="170" fill="url(#glow)"/>'
            '<g transform="rotate(5 230 150)"><rect x="118" y="40" width="226" height="226" rx="26" fill="#fff" fill-opacity=".08" stroke="#fff" stroke-opacity=".32"/>'
            '<rect x="186" y="26" width="90" height="30" rx="10" fill="#3BD98A"/><rect x="206" y="36" width="50" height="8" rx="4" fill="#06130D" fill-opacity=".5"/>'
            f'{rows}</g>'
            f'{capsule(86, 240, 130, 50, -34, GREEN, WHITE, "a")}'
            f'{capsule(364, 70, 96, 38, 30, AMBER, ("#FFFFFF", "#E9E3CF"), "b")}')
    return wrap((420, 300), body)


def hero_forecast():
    xs = [90, 160, 230, 300]
    hs = [34, 64, 98, 132]
    cols = ["#35B26E", "#A8D45A", "#F2C94C", "#E5534B"]
    bars, labels = "", ""
    for x, h, c, m in zip(xs, hs, cols, (3, 6, 9, 12)):
        bars += (f'<rect x="{x - 17}" y="{206 - h}" width="34" height="{h}" rx="9" fill="{c}" fill-opacity=".92"/>'
                 f'<rect x="{x - 11}" y="{210 - h}" width="9" height="{max(h - 14, 8)}" rx="4.5" fill="#fff" fill-opacity=".28"/>')
        labels += (f'<circle cx="{x}" cy="232" r="13" fill="#fff"/>'
                   f'<text x="{x}" y="237" font-family="DM Sans,Arial,sans-serif" font-size="14" font-weight="700" text-anchor="middle" fill="#0C1F16">{m}</text>')
    body = ('<circle cx="215" cy="160" r="175" fill="url(#glow)"/>'
            '<rect x="40" y="52" width="330" height="204" rx="30" fill="#fff" fill-opacity=".07" stroke="#fff" stroke-opacity=".28"/>'
            '<path d="M60 232 H350" stroke="#fff" stroke-opacity=".35" stroke-width="3" stroke-linecap="round"/>'
            '<path d="M72 150 C130 150 150 118 200 104 S300 70 340 66" fill="none" stroke="#fff" stroke-opacity=".5" stroke-width="3" stroke-dasharray="3 8" stroke-linecap="round"/>'
            f'{bars}{labels}'
            f'{capsule(338, 56, 110, 42, 26, GREEN, WHITE, "a")}'
            f'{capsule(60, 70, 84, 34, -30, AMBER, ("#FFFFFF", "#E9E3CF"), "b")}'
            '<text x="215" y="82" font-family="DM Sans,Arial,sans-serif" font-size="13" fill="#fff" fill-opacity=".7" text-anchor="middle">months ahead</text>')
    return wrap((420, 300), body)


def hero_models():
    def node(x, y, r=17, hot=False):
        c = "#3BD98A" if hot else "#fff"
        return (f'<circle cx="{x}" cy="{y}" r="{r + 9}" fill="{c}" fill-opacity=".14"/>'
                f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" fill-opacity="{1 if hot else .85}"/>')
    lines = ""
    for a, b, hot in [((210, 62), (130, 140), True), ((210, 62), (290, 140), False), ((130, 140), (92, 222), False),
                      ((130, 140), (168, 222), True), ((290, 140), (252, 222), False), ((290, 140), (330, 222), False)]:
        lines += (f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" stroke="{"#3BD98A" if hot else "#fff"}" stroke-opacity="{.95 if hot else .35}" '
                  f'stroke-width="{5 if hot else 3}" stroke-linecap="round"/>')
    body = ('<circle cx="215" cy="150" r="175" fill="url(#glow)"/>' + lines
            + node(210, 62, 20, True) + node(130, 140, 16, True) + node(290, 140, 16) + node(92, 222, 12) + node(168, 222, 12, True)
            + node(252, 222, 12) + node(330, 222, 12)
            + capsule(60, 70, 96, 38, -28, AMBER, ("#FFFFFF", "#E9E3CF"), "a") + capsule(360, 66, 104, 40, 30, MINT, WHITE, "b"))
    return wrap((420, 300), body)


def hero_data():
    def disc(y, shade):
        return (f'<ellipse cx="210" cy="{y + 46}" rx="112" ry="30" fill="#000" fill-opacity=".3"/>'
                f'<rect x="98" y="{y}" width="224" height="46" fill="url(#dk{shade})"/>'
                f'<ellipse cx="210" cy="{y + 46}" rx="112" ry="30" fill="url(#dk{shade})"/>'
                f'<ellipse cx="210" cy="{y}" rx="112" ry="30" fill="#E8F5EE" fill-opacity=".92"/>'
                f'<ellipse cx="210" cy="{y}" rx="112" ry="30" fill="none" stroke="#fff" stroke-opacity=".6"/>'
                f'<rect x="150" y="{y + 18}" width="46" height="6" rx="3" fill="#fff" fill-opacity=".5"/>'
                f'<circle cx="286" cy="{y + 24}" r="5" fill="#3BD98A"/><circle cx="268" cy="{y + 24}" r="5" fill="#F2C94C"/>')
    grad = ''.join(f'<linearGradient id="dk{i}" x1="0" x2="1"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>'
                   for i, (a, b) in enumerate([("#1F8A57", "#0E5A38"), ("#2AA568", "#137046"), ("#37BC78", "#1A8453")]))
    body = (f'<defs>{grad}</defs><circle cx="210" cy="150" r="175" fill="url(#glow)"/>' + disc(168, 0) + disc(116, 1) + disc(64, 2)
            + capsule(340, 238, 120, 46, -30, AMBER, ("#FFFFFF", "#E9E3CF"), "a") + capsule(70, 90, 100, 40, 28, GREEN, WHITE, "b"))
    return wrap((420, 300), body)


def hero_setup():
    cloud = ('<path d="M120 188 a44 44 0 0 1 8-86 a62 62 0 0 1 118-14 a50 50 0 0 1 48 100 z" fill="#fff" fill-opacity=".14" stroke="#fff" stroke-opacity=".85" stroke-width="6" stroke-linejoin="round"/>')
    body = ('<circle cx="215" cy="140" r="175" fill="url(#glow)"/>' + cloud
            + '<path d="M212 118 V208" stroke="#3BD98A" stroke-width="12" stroke-linecap="round"/><path d="M180 180 L212 214 L244 180" fill="none" stroke="#3BD98A" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"/>'
            + '<rect x="128" y="232" width="168" height="16" rx="8" fill="#fff" fill-opacity=".85"/>'
            + capsule(344, 220, 118, 46, -30, AMBER, ("#FFFFFF", "#E9E3CF"), "a") + capsule(70, 238, 96, 38, 24, GREEN, WHITE, "b"))
    return wrap((420, 300), body)


def bottle(x, base, h, w, body, cap_c="#F8FCF9", label="#fff"):
    top = base - h
    return (f'<rect x="{x + w * .28}" y="{top - 16}" width="{w * .44}" height="18" rx="5" fill="{cap_c}"/>'
            f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="12" fill="{body}"/>'
            f'<rect x="{x + 6}" y="{top + h * .38}" width="{w - 12}" height="{h * .34}" rx="6" fill="{label}" fill-opacity=".92"/>'
            f'<rect x="{x + 12}" y="{top + h * .38 + 9}" width="{w * .42}" height="5" rx="2.5" fill="#0C1F16" fill-opacity=".35"/>'
            f'<rect x="{x + 8}" y="{top + 8}" width="6" height="{h - 16}" rx="3" fill="#fff" fill-opacity=".28"/>')

def box(x, base, h, w, c1, c2):
    top = base - h
    return (f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="7" fill="{c1}"/><rect x="{x}" y="{top}" width="{w}" height="{h * .3}" rx="7" fill="{c2}"/>'
            f'<rect x="{x + 9}" y="{top + h * .46}" width="{w - 18}" height="6" rx="3" fill="#fff" fill-opacity=".85"/>'
            f'<rect x="{x + 9}" y="{top + h * .46 + 14}" width="{w * .5}" height="5" rx="2.5" fill="#fff" fill-opacity=".45"/>')

def gap(x, base, w, h):
    return (f'<rect x="{x}" y="{base - h}" width="{w}" height="{h}" rx="12" fill="#E5534B" fill-opacity=".10" stroke="#FF6B5E" stroke-opacity=".9" stroke-width="2" stroke-dasharray="6 7"/>'
            f'<circle cx="{x + w / 2}" cy="{base - h / 2}" r="15" fill="#E5534B"/><path d="M{x + w / 2} {base - h / 2 - 7}v9M{x + w / 2} {base - h / 2 + 8}v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>')


def banner_shelf():
    """A pharmacy shelf with a few empty spaces: the picture of a shortage."""
    shelf1 = [bottle(60, 118, 78, 44, "#3BD98A"), bottle(112, 118, 64, 40, "#F8FCF9", "#E8F5EE", "#3BD98A"), box(160, 118, 70, 58, "#F8FCF9", "#3BD98A"),
              gap(236, 118, 50, 76), bottle(300, 118, 72, 44, "#FFD36B"), box(352, 118, 58, 52, "#8FEAB8", "#22A065"), bottle(416, 118, 80, 46, "#3BD98A"),
              gap(502, 118, 44, 64), box(556, 118, 66, 60, "#F8FCF9", "#E39A12"), bottle(628, 118, 74, 42, "#8FEAB8")]
    shelf2 = [box(60, 218, 62, 56, "#8FEAB8", "#22A065"), bottle(128, 218, 78, 44, "#F8FCF9", "#E8F5EE", "#E39A12"), bottle(180, 218, 70, 42, "#3BD98A"),
              box(228, 218, 70, 64, "#FFD36B", "#E39A12"), gap(304, 218, 46, 70), bottle(362, 218, 76, 44, "#8FEAB8"), box(414, 218, 60, 56, "#F8FCF9", "#3BD98A"),
              bottle(480, 218, 82, 46, "#3BD98A"), bottle(534, 218, 66, 40, "#FFD36B"), box(606, 218, 68, 58, "#8FEAB8", "#22A065")]
    bg = ('<circle cx="380" cy="130" r="300" fill="url(#glow)"/>'
          '<rect x="36" y="118" width="680" height="12" rx="6" fill="#fff" fill-opacity=".55"/><rect x="36" y="218" width="680" height="12" rx="6" fill="#fff" fill-opacity=".55"/>'
          '<rect x="36" y="130" width="680" height="6" fill="#000" fill-opacity=".25"/><rect x="36" y="230" width="680" height="6" fill="#000" fill-opacity=".25"/>')
    return wrap((760, 250), bg + "".join(shelf1) + "".join(shelf2))


def banner_search():
    items = [bottle(70, 190, 82, 46, "#3BD98A"), box(126, 190, 66, 60, "#F8FCF9", "#3BD98A"), bottle(198, 190, 90, 48, "#FFD36B"),
             box(254, 190, 70, 62, "#8FEAB8", "#22A065"), gap(338, 190, 56, 82), bottle(420, 190, 84, 46, "#3BD98A"),
             box(476, 190, 66, 60, "#F8FCF9", "#E39A12"), bottle(548, 190, 78, 44, "#8FEAB8"), bottle(600, 190, 88, 48, "#FFD36B"),
             box(656, 190, 62, 56, "#8FEAB8", "#22A065")]
    lens = ('<circle cx="366" cy="128" r="74" fill="#fff" fill-opacity=".07"/><circle cx="366" cy="128" r="74" fill="none" stroke="#fff" stroke-opacity=".92" stroke-width="9"/>'
            '<path d="M420 182 L462 232" stroke="#fff" stroke-opacity=".92" stroke-width="16" stroke-linecap="round"/><path d="M420 182 L462 232" stroke="#3BD98A" stroke-opacity=".55" stroke-width="5" stroke-linecap="round"/>')
    bg = ('<circle cx="380" cy="130" r="300" fill="url(#glow)"/><rect x="36" y="190" width="690" height="12" rx="6" fill="#fff" fill-opacity=".55"/>'
          '<rect x="36" y="202" width="690" height="6" fill="#000" fill-opacity=".25"/>')
    return wrap((760, 250), bg + "".join(items) + lens)


def banner_podium():
    def block(x, h, c1, c2, n):
        return (f'<defs><linearGradient id="pb{n}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient></defs>'
                f'<rect x="{x}" y="{214 - h}" width="132" height="{h}" rx="14" fill="url(#pb{n})"/>'
                f'<rect x="{x + 10}" y="{214 - h + 10}" width="12" height="{h - 20}" rx="6" fill="#fff" fill-opacity=".22"/>'
                f'<text x="{x + 66}" y="{214 - h + 46}" font-family="DM Sans,Arial,sans-serif" font-size="34" font-weight="700" text-anchor="middle" fill="#fff">{n}</text>')
    body = ('<circle cx="380" cy="130" r="290" fill="url(#glow)"/>'
            + block(170, 92, "#F2C94C", "#C98F0A", 2) + block(304, 120, "#EE6A5F", "#B93028", 1) + block(438, 68, "#3BD98A", "#14804F", 3)
            + bottle(350, 94, 56, 40, "#F8FCF9", "#E8F5EE", "#E5534B") + bottle(216, 122, 52, 38, "#FFD36B") + box(480, 146, 50, 52, "#8FEAB8", "#22A065")
            + '<circle cx="408" cy="56" r="14" fill="#E5534B"/><path d="M408 49v8M408 61v.5" stroke="#fff" stroke-width="3" stroke-linecap="round"/>'
            + '<rect x="150" y="214" width="440" height="12" rx="6" fill="#fff" fill-opacity=".55"/>'
            + capsule(80, 160, 120, 46, -30, GREEN, WHITE, "a") + capsule(670, 96, 110, 42, 28, AMBER, ("#FFFFFF", "#E9E3CF"), "b"))
    return wrap((760, 250), body)


def banner_timeline():
    n = 12
    bars, labs = "", ""
    cols = ["#35B26E", "#3FB86C", "#58BF62", "#7BC85A", "#A3CF58", "#CDD055", "#F2C94C", "#F4B445", "#F29A42", "#EE7C45", "#EA6048", "#E5534B"]
    for i in range(n):
        x = 70 + i * 54
        h = 26 + i * 10.5
        key = (i + 1) % 3 == 0
        bars += (f'<rect x="{x - 15}" y="{206 - h:.0f}" width="30" height="{h:.0f}" rx="8" fill="{cols[i]}"/>'
                 f'<rect x="{x - 9}" y="{210 - h:.0f}" width="7" height="{max(h - 14, 8):.0f}" rx="3.5" fill="#fff" fill-opacity=".28"/>')
        labs += (f'<circle cx="{x}" cy="228" r="{12 if key else 8}" fill="{"#fff" if key else "#fff"}" fill-opacity="{1 if key else .35}"/>'
                 + (f'<text x="{x}" y="233" font-family="DM Sans,Arial,sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="#0C1F16">{i + 1}</text>' if key else ""))
    body = ('<circle cx="380" cy="130" r="300" fill="url(#glow)"/><path d="M40 214 H720" stroke="#fff" stroke-opacity=".4" stroke-width="3" stroke-linecap="round"/>'
            '<path d="M70 160 C220 150 330 120 420 96 S620 40 706 30" fill="none" stroke="#fff" stroke-opacity=".5" stroke-width="3" stroke-dasharray="3 9" stroke-linecap="round"/>'
            + bars + labs + capsule(210, 58, 110, 42, -18, GREEN, WHITE, "a") + capsule(52, 112, 90, 36, -30, AMBER, ("#FFFFFF", "#E9E3CF"), "b"))
    return wrap((760, 250), body)



def pattern():
    """Very faint repeating motif for the page background (capsules, crosses, dots)."""
    def pill(x, y, a, w=44, h=18):
        return (f'<g transform="translate({x} {y}) rotate({a})" fill="none" stroke="#1E8A5A" stroke-width="2" stroke-linecap="round">'
                f'<rect x="{-w / 2}" y="{-h / 2}" width="{w}" height="{h}" rx="{h / 2}"/><path d="M0 {-h / 2}v{h}"/></g>')

    def plus(x, y, s=8):
        return f'<path d="M{x - s} {y}h{2 * s}M{x} {y - s}v{2 * s}" stroke="#1E8A5A" stroke-width="2.4" stroke-linecap="round"/>'
    b = (pill(60, 60, -35) + pill(250, 150, 30, 38, 16) + pill(120, 270, 12, 34, 14) + pill(310, 320, -50, 40, 16)
         + plus(190, 50) + plus(40, 190, 7) + plus(300, 240, 7) + plus(200, 330, 6)
         + '<circle cx="150" cy="170" r="4" fill="#1E8A5A"/><circle cx="330" cy="80" r="3.5" fill="#1E8A5A"/><circle cx="70" cy="330" r="3.5" fill="#1E8A5A"/><circle cx="240" cy="250" r="3" fill="#1E8A5A"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="360" height="360" viewBox="0 0 360 360"><g opacity=".13">{b}</g></svg>'


ALL = [("hero_art.svg", hero_art), ("hero_small.svg", hero_small), ("card_capsules.svg", card_art),
       ("hero_risk.svg", hero_risk), ("hero_watch.svg", hero_watch), ("hero_forecast.svg", hero_forecast), ("hero_models.svg", hero_models),
       ("hero_data.svg", hero_data), ("hero_setup.svg", hero_setup), ("banner_shelf.svg", banner_shelf), ("banner_search.svg", banner_search), ("banner_podium.svg", banner_podium), ("banner_timeline.svg", banner_timeline), ("pattern.svg", pattern)]
for name, fn in ALL:
    (OUT / name).write_text(fn(), encoding="utf-8")
    print("wrote", name)
