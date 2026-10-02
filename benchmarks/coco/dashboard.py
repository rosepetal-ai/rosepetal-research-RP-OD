"""A self-refreshing web page with the state of a training run: epoch, progress inside the epoch, estimated end,
validation curves, loss, learning rate, and the official COCO numbers once they exist.

Runs next to the training (same machine), rewrites the page every minute; the page reloads itself every minute too.
Served by Jupyter at  http://localhost:8890/files/dashboard/index.html?token=<token>  (through the SSH tunnel).

    python dashboard.py --run max_b2                          # --reference <run> adds another run's curve
"""
import argparse
import csv
import html
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo('Europe/Madrid')
REFERENCES = [('RT-DETR-R18', 46.5), ('D-FINE-M', 52.3)]                     # official COCO val2017 AP, as published   # official COCO val2017 AP, as published
PROGRESS = re.compile(r'^\s+epoch (\d+)/(\d+) \[.*\]\s+(\d+)% · (\d+)/(\d+) steps · loss ([\d.]+) · (\d+) img/s · ~(\d+) min left')


def read_csv(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def running(pid_file):
    try:
        os.kill(int(pid_file.read_text()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False


def state(work, run, reference):
    rows = read_csv(work / 'runs/train' / run / 'results.csv')
    ref = read_csv(work / 'runs/train' / reference / 'results.csv') if reference else []
    log = (work / 'logs' / f'{run}.log').read_text(errors='replace').splitlines() if (work / 'logs' / f'{run}.log').exists() else []
    progress = next((m for m in map(PROGRESS.match, reversed(log)) if m), None)
    args_file = work / 'runs/train' / run / 'args.yaml'
    total = int(re.search(r'^epochs: (\d+)', args_file.read_text(), re.M).group(1)) if args_file.exists() else None
    official = []
    f = work / 'logs/official.txt'
    for line in (f.read_text().splitlines() if f.exists() else []):
        what, _, js = line.partition(' | ')
        if f'runs/train/{run}/' in what:
            try:
                official.append({'options': what.split('best.pt')[-1].strip() or 'as trained', **json.loads(js)})
            except json.JSONDecodeError:
                pass
    alive = running(work / 'logs' / f'{run}.pid')
    done = len(rows)
    status = ('finished' if any('all done' in l for l in log[-5:]) else 'evaluating' if alive and total and done >= total
              else 'training' if alive else 'stopped')
    now = datetime.now(TZ)
    eta = None
    in_epoch = progress and int(progress.group(1)) == done + 1
    if status == 'training' and total and (rows or in_epoch):
        if rows:
            recent = [float(r['seconds']) for r in rows[-5:]]
            per_epoch = sum(recent) / len(recent)
        else:                      # first epoch: from its own progress line, plus ~3 min of validation
            per_epoch = int(progress.group(8)) * 60 / max(1 - int(progress.group(3)) / 100, 0.01) + 180
        current_left = int(progress.group(8)) * 60 + 180 if in_epoch else per_epoch
        eta = now + timedelta(seconds=current_left + per_epoch * max(0, total - done - 1))
    return {'run': run, 'reference': reference, 'rows': rows, 'ref': ref, 'total': total, 'done': done, 'status': status,
            'progress': progress.groups() if progress else None, 'eta': eta, 'now': now, 'official': official,
            'log': [l for l in log if l.startswith(('epoch ', 'val ', '  epoch', '===', 'no improvement', 'best epoch'))][-12:],
            'args': args_file.read_text() if args_file.exists() else ''}


def page(s):
    rows, total, done = s['rows'], s['total'] or 0, s['done']
    best = max(rows, key=lambda r: float(r['map50_95']), default=None)
    prog = s['progress']
    cur_pct = int(prog[2]) if prog and int(prog[0]) == done + 1 else (100 if s['status'] != 'training' else 0)
    overall = 100 * (done + cur_pct / 100) / total if total else 0
    status_cls = {'training': 'good', 'evaluating': 'good', 'finished': 'good', 'stopped': 'critical'}[s['status']]
    status_txt = {'training': 'Training', 'evaluating': 'Official evaluation', 'finished': 'Finished',
                  'stopped': 'Stopped — resume with: bash /work/run_max_b2.sh resume'}[s['status']]
    eta = s['eta'].strftime('%a %d %b, %H:%M') if s['eta'] else '—'
    data = {
        'map': [{'name': 'val mAP50-95', 'slot': 1, 'points': [[int(r['epoch']), 100 * float(r['map50_95'])] for r in rows], 'label': True},
                {'name': 'val mAP50', 'slot': 2, 'points': [[int(r['epoch']), 100 * float(r['map50'])] for r in rows], 'label': True}]
               + ([{'name': f"reference: {s['reference']}, 6-epoch pilot of 1 Oct (mAP50-95)", 'slot': 3,
                    'points': [[int(r['epoch']), 100 * float(r['map50_95'])] for r in s['ref']], 'label': False}]
                  if s['ref'] and rows else []),          # only next to real data, never alone
        'loss': [{'name': 'train loss', 'slot': 1, 'points': [[int(r['epoch']), float(r['loss'])] for r in rows], 'label': True}],
        'lr': [{'name': 'learning rate (×1e-4)', 'slot': 1, 'points': [[int(r['epoch']), 1e4 * float(r['lr'])] for r in rows], 'label': True}],
        'refs': [{'label': f'{n} {v}', 'y': v} for n, v in REFERENCES],
        'total': total,
    }
    epoch_rows = ''.join(
        f"<tr><td>{r['epoch']}</td><td>{html.escape(r['phase'])}</td><td>{float(r['lr']):.2e}</td><td>{float(r['loss']):.3f}</td>"
        f"<td><b>{100 * float(r['map50_95']):.1f}</b></td><td>{100 * float(r['map50']):.1f}</td><td>{float(r['precision']):.3f}</td>"
        f"<td>{float(r['recall']):.3f}</td><td>{float(r['mean_iou']):.3f}</td><td>{float(r['fp_per_image']):.2f}</td>"
        f"<td>{float(r['seconds']) / 60:.0f}</td></tr>" for r in reversed(rows))
    official_rows = ''.join(
        f"<tr><td>{html.escape(o['model'])}</td><td>{o['params_M']}</td><td><b>{o['AP']}</b></td><td>{o['AP50']}</td>"
        f"<td>{o['AP75']}</td><td>{o['APs']}</td><td>{o['APm']}</td><td>{o['APl']}</td></tr>" for o in s['official'])
    tile_progress = (f"<div class='value'>{cur_pct}%</div><div class='meter'><span style='width:{cur_pct}%'></span></div>"
                     f"<div class='sub'>{prog[3]}/{prog[4]} steps · {prog[6]} img/s · ~{prog[7]} min left</div>"
                     if prog and s['status'] == 'training' and int(prog[0]) == done + 1 else
                     "<div class='value'>—</div><div class='sub'>first progress line after ~2 min</div>")
    return TEMPLATE.format(
        run=html.escape(s['run']), updated=s['now'].strftime('%d %b %H:%M'), status_cls=status_cls, status_txt=status_txt,
        best=f"{100 * float(best['map50_95']):.1f}" if best else '—',
        best_sub=f"epoch {best['epoch']} · mAP50 {100 * float(best['map50']):.1f}" if best else 'after the first epoch',
        epoch=f'{done} / {total}' if total else '—', overall=f'{overall:.0f}', tile_progress=tile_progress, eta=eta,
        epoch_rows=epoch_rows or "<tr><td colspan='11'>no epoch finished yet</td></tr>",
        official=(f"<h2>Official COCO numbers (pycocotools, val2017)</h2><table><thead><tr><th>model</th><th>params M</th>"
                  f"<th>AP</th><th>AP50</th><th>AP75</th><th>APs</th><th>APm</th><th>APl</th></tr></thead>"
                  f"<tbody>{official_rows}</tbody></table>") if official_rows else '',
        log=html.escape('\n'.join(s['log'])), args=html.escape(s['args']), data=json.dumps(data))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>RP-OD training</title>
<style>
.viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --track:#cde2fb;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --good:#0ca30c; --critical:#d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark;
  --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#ffffff; --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --track:#184f95; --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; }} }}
:root[data-theme="dark"] .viz-root {{ color-scheme: dark; --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#ffffff;
  --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --track:#184f95;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; }}
body {{ margin:0; }}
.viz-root {{ background:var(--page); color:var(--text-primary); font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  min-height:100vh; padding:20px 16px 40px; box-sizing:border-box; }}
.wrap {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:20px; font-weight:600; margin:0 0 4px; }} h2 {{ font-size:15px; font-weight:600; margin:28px 0 10px; }}
.meta {{ color:var(--text-secondary); font-size:13px; }}
.status {{ display:inline-flex; align-items:center; gap:6px; font-size:13px; font-weight:600; margin-top:10px; }}
.status::before {{ content:""; width:9px; height:9px; border-radius:50%; background:var(--{status_cls}); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:18px; }}
.tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.tile .label {{ color:var(--text-secondary); font-size:13px; }}
.tile .value {{ font-size:28px; font-weight:600; margin-top:4px; }} .tile.hero .value {{ font-size:48px; line-height:1.05; }}
.tile .sub {{ color:var(--muted); font-size:12px; margin-top:6px; }}
.meter {{ height:6px; border-radius:3px; background:var(--track); margin-top:8px; overflow:hidden; }}
.meter span {{ display:block; height:100%; background:var(--series-1); border-radius:3px; }}
.card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px 8px; position:relative; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--text-secondary); margin-bottom:4px; }}
.legend i {{ display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; border-radius:1px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; margin-top:12px; }}
svg {{ width:100%; height:auto; display:block; }} svg:focus {{ outline:2px solid var(--series-1); outline-offset:2px; }}
.tip {{ position:absolute; pointer-events:none; background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:8px 10px; font-size:12px; box-shadow:0 4px 14px rgba(0,0,0,.12); display:none; white-space:nowrap; }}
.tip .row {{ display:flex; align-items:center; gap:6px; color:var(--text-secondary); }} .tip b {{ color:var(--text-primary); }}
.tip i {{ display:inline-block; width:12px; height:2px; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; overflow:hidden; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:6px 10px; border-bottom:1px solid var(--grid); }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--text-secondary); font-weight:600; }}
.scroll {{ max-height:360px; overflow:auto; border-radius:10px; }}
pre {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:12px; font-size:12px;
  overflow:auto; color:var(--text-secondary); }}
.note {{ color:var(--muted); font-size:12px; margin-top:6px; }}
</style></head>
<body><div class="viz-root"><div class="wrap">
<h1>RP-OD on COCO · {run}</h1>
<div class="meta">HGNetv2-B2 · neck l · P2 · 6 decoder layers · 640×640 · lr 2e-4 / backbone 2e-5 · batch 32 · updated {updated} (Madrid) · reloads every minute</div>
<div class="status">{status_txt}</div>
<div class="tiles">
  <div class="tile hero"><div class="label">Best val mAP50-95</div><div class="value">{best}</div><div class="sub">{best_sub}</div></div>
  <div class="tile"><div class="label">Epochs done</div><div class="value">{epoch}</div><div class="meter"><span style="width:{overall}%"></span></div><div class="sub">{overall}% of the run</div></div>
  <div class="tile"><div class="label">Current epoch</div>{tile_progress}</div>
  <div class="tile"><div class="label">Estimated end</div><div class="value" style="font-size:20px;margin-top:10px">{eta}</div><div class="sub">training + official evaluation</div></div>
</div>
<h2>Validation after every epoch</h2>
<div class="card" id="map"></div>
<div class="note">RP-OD evaluator on val2017; the official pycocotools AP comes out ~0.4 higher. Reference lines: published official AP of models of similar size, after their full training.</div>
<div class="grid2"><div><h2>Training loss</h2><div class="card" id="loss"></div></div><div><h2>Learning rate</h2><div class="card" id="lr"></div></div></div>
{official}
<h2>Every epoch</h2>
<div class="scroll"><table><thead><tr><th>epoch</th><th>phase</th><th>lr</th><th>loss</th><th>mAP50-95</th><th>mAP50</th><th>P</th><th>R</th><th>IoU</th><th>FP/img</th><th>min</th></tr></thead>
<tbody>{epoch_rows}</tbody></table></div>
<h2>Latest log lines</h2><pre>{log}</pre>
<details><summary class="meta">Run arguments</summary><pre>{args}</pre></details>
</div></div>
<script>
const D = {data};
const css = n => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(n).trim();
function niceTicks(lo, hi, n) {{
  const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0), top = Math.ceil(hi / step - 1e-9) * step;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= top + 1e-9; v += step) out.push(+v.toFixed(6)); return out;
}}
function chart(id, series, opts) {{
  const host = document.getElementById(id), NS = 'http://www.w3.org/2000/svg';
  const W = opts.w || 1000, H = opts.h || 300, m = {{l: 44, r: opts.refs ? 150 : 60, t: 12, b: 28}};
  const all = series.flatMap(s => s.points);
  const legend = document.createElement('div'); legend.className = 'legend';
  if (series.length > 1) series.forEach(s => {{ const sp = document.createElement('span'); const k = document.createElement('i');
    k.style.background = css('--series-' + s.slot); sp.appendChild(k); sp.appendChild(document.createTextNode(s.name)); legend.appendChild(sp); }});
  host.appendChild(legend);
  if (!all.length) {{ const p = document.createElement('div'); p.className = 'note'; p.textContent = 'after the first epoch'; host.appendChild(p); return; }}
  const xMax = Math.max(D.total || 1, ...all.map(p => p[0]));
  let yMax = Math.max(...all.map(p => p[1]), ...(opts.refs || []).map(r => r.y)) * 1.06;
  const ticks = niceTicks(0, yMax, 5); yMax = ticks[ticks.length - 1];
  const x = v => m.l + (v / xMax) * (W - m.l - m.r), y = v => H - m.b - (v / yMax) * (H - m.t - m.b);
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`); svg.setAttribute('tabindex', '0');
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', opts.title);
  const el = (tag, a, parent = svg) => {{ const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); parent.appendChild(e); return e; }};
  const text = (s, a) => {{ const t = el('text', {{'font-size': 12, fill: css('--muted'), ...a}}); t.textContent = s; return t; }};
  ticks.forEach(v => {{ el('line', {{x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: v ? css('--grid') : css('--axis'), 'stroke-width': 1}});
    text(opts.fmt(v), {{x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end'}}); }});
  niceTicks(0, xMax, 6).filter(v => v <= xMax).forEach(v => text(String(v), {{x: x(v), y: H - 8, 'text-anchor': 'middle'}}));
  (opts.refs || []).forEach(r => {{ el('line', {{x1: m.l, x2: W - m.r, y1: y(r.y), y2: y(r.y), stroke: css('--axis'), 'stroke-width': 1}});
    text(r.label, {{x: W - m.r + 8, y: y(r.y) + 4}}); }});
  series.forEach(s => {{
    if (!s.points.length) return;
    el('path', {{d: s.points.map((p, i) => (i ? 'L' : 'M') + x(p[0]) + ' ' + y(p[1])).join(''), fill: 'none',
      stroke: css('--series-' + s.slot), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}});
    const last = s.points[s.points.length - 1];
    el('circle', {{cx: x(last[0]), cy: y(last[1]), r: 4, fill: css('--series-' + s.slot), stroke: css('--surface-1'), 'stroke-width': 2}});
    if (s.label && !opts.refs) text(opts.fmt(last[1]), {{x: x(last[0]) + 8, y: y(last[1]) + 4, fill: css('--text-secondary')}});
    if (s.label && opts.refs) text(s.name.replace('val ', '') + ' ' + opts.fmt(last[1]), {{x: x(last[0]) + 8, y: y(last[1]) + 4, fill: css('--text-secondary')}});
  }});
  const cross = el('line', {{y1: m.t, y2: H - m.b, stroke: css('--axis'), 'stroke-width': 1, visibility: 'hidden'}});
  const hit = el('rect', {{x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'}});
  host.appendChild(svg);
  const tip = document.createElement('div'); tip.className = 'tip'; host.appendChild(tip);
  const epochs = [...new Set(all.map(p => p[0]))].sort((a, b) => a - b);
  let cur = -1;
  function show(i) {{
    cur = Math.max(0, Math.min(epochs.length - 1, i)); const e = epochs[cur];
    cross.setAttribute('x1', x(e)); cross.setAttribute('x2', x(e)); cross.setAttribute('visibility', 'visible');
    tip.replaceChildren(); const h = document.createElement('div'); h.className = 'row'; h.textContent = 'epoch ' + e; tip.appendChild(h);
    series.forEach(s => {{ const p = s.points.find(q => q[0] === e); if (!p) return;
      const r = document.createElement('div'); r.className = 'row'; const k = document.createElement('i'); k.style.background = css('--series-' + s.slot);
      const b = document.createElement('b'); b.textContent = opts.fmt(p[1]); r.append(k, b, document.createTextNode(s.name)); tip.appendChild(r); }});
    const box = svg.getBoundingClientRect(), px = x(e) / W * box.width;
    tip.style.display = 'block'; tip.style.top = '40px';
    tip.style.left = (px + tip.offsetWidth + 24 > box.width ? px - tip.offsetWidth - 8 : px + 16) + 'px';
  }}
  hit.addEventListener('pointermove', ev => {{ const box = svg.getBoundingClientRect(); const v = ((ev.clientX - box.left) / box.width * W - m.l) / (W - m.l - m.r) * xMax;
    let best = 0; epochs.forEach((e, i) => {{ if (Math.abs(e - v) < Math.abs(epochs[best] - v)) best = i; }}); show(best); }});
  hit.addEventListener('pointerleave', () => {{ tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); }});
  svg.addEventListener('keydown', ev => {{ if (ev.key === 'ArrowRight') show(cur + 1); if (ev.key === 'ArrowLeft') show(cur < 0 ? epochs.length - 1 : cur - 1); }});
  svg.addEventListener('focus', () => show(epochs.length - 1)); svg.addEventListener('blur', () => {{ tip.style.display = 'none'; }});
}}
chart('map', D.map, {{title: 'Validation mAP per epoch', fmt: v => v.toFixed(1), refs: D.refs, h: 320}});
chart('loss', D.loss, {{title: 'Training loss per epoch', fmt: v => v.toFixed(1), w: 520, h: 240}});
chart('lr', D.lr, {{title: 'Learning rate per epoch', fmt: v => v.toFixed(2), w: 520, h: 240}});
</script></body></html>
"""

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='max_b2')
    p.add_argument('--reference', default=None, help='another run whose mAP50-95 curve is drawn for comparison')
    p.add_argument('--work', default='/work')
    p.add_argument('--out', default='/work/dashboard/index.html')
    p.add_argument('--every', type=int, default=60)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        tmp = out.with_suffix('.tmp')
        tmp.write_text(page(state(Path(a.work), a.run, a.reference)))
        os.replace(tmp, out)
        if a.once:
            break
        time.sleep(a.every)
