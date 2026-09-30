#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 فحص بصمة الحصاد — هل التسريب موجود فعلاً في الداتا؟
============================================================================

هذا هو الفحص الذي كشف أن `Infilteration` في CIC-IDS2018 مسحُ منافذ لا تسريب.
شغّليه على أي داتا سيت قبل الاستثمار فيها.

يقيس أربع خصائص للحصاد النشط:
    ١. الحجم    — بايتات صادرة أكبر من الطبيعي؟
    ٢. اللاتماثل — صادر ≫ وارد؟
    ٣. الاستمرار — تدفقات تدوم، لا حزمتين؟
    ٤. التركيز   — وجهات قليلة متكررة؟

الاستخدام:
    python3 harvest_signature.py --data dapt_exfil_packed.parquet \
        --label-col Stage --attack-values "Data Exfiltration"

    # لو أعمدة الاتجاه بأسماء مختلفة:
    python3 harvest_signature.py --data f.parquet --label-col Label \
        --attack-values exfiltration \
        --fwd-bytes "TotLen Fwd Pkts" --bwd-bytes "TotLen Bwd Pkts"
============================================================================
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# مرشّحات أسماء الأعمدة عبر إصدارات CICFlowMeter المختلفة
CAND = {
    'fwd_bytes': ['TotLen Fwd Pkts', 'Total Length of Fwd Packets',
                  'Total Length of Fwd Packet', 'TotLen Fwd Pkt', 'Fwd Bytes'],
    'bwd_bytes': ['TotLen Bwd Pkts', 'Total Length of Bwd Packets',
                  'Total Length of Bwd Packet', 'TotLen Bwd Pkt', 'Bwd Bytes'],
    'fwd_pkts':  ['Tot Fwd Pkts', 'Total Fwd Packets', 'Total Fwd Packet'],
    'bwd_pkts':  ['Tot Bwd Pkts', 'Total Backward Packets', 'Total Bwd packets'],
    'duration':  ['Flow Duration'],
    'dst_port':  ['Dst Port', 'Destination Port'],
    'src_ip':    ['Src IP', 'Source IP', 'src_ip'],
    'dst_ip':    ['Dst IP', 'Destination IP', 'dst_ip'],
}


def resolve(df: pd.DataFrame, key: str, override: str | None) -> str | None:
    if override:
        return override if override in df.columns else None
    for c in CAND[key]:
        if c in df.columns:
            return c
    return None


def pct(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--label-col', required=True)
    ap.add_argument('--attack-values', nargs='+', required=True)
    ap.add_argument('--benign-value', default=None)
    for k in CAND:
        ap.add_argument('--' + k.replace('_', '-'), default=None)
    a = ap.parse_args()

    df = pd.read_parquet(a.data) if a.data.endswith('.parquet') \
        else pd.read_csv(a.data, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    col = {k: resolve(df, k, getattr(a, k)) for k in CAND}
    missing = [k for k in ('fwd_bytes', 'bwd_bytes') if not col[k]]
    if missing:
        print(f'تعذّر إيجاد أعمدة: {missing}')
        print(f'الأعمدة المتاحة: {df.columns.tolist()}')
        return

    lab = df[a.label_col].astype(str).str.strip()
    atk = lab.str.lower().isin({v.strip().lower() for v in a.attack_values})
    ben = (lab == a.benign_value) if a.benign_value else ~atk

    A, B = df[atk], df[ben]
    print(f'هجوم: {len(A):,}  |  حميد: {len(B):,}  |  '
          f'نسبة الهجوم: {len(A) / max(len(df), 1) * 100:.3f}%')
    if len(A) == 0:
        print('لا توجد صفوف هجوم — راجعي --attack-values.')
        return
    print('=' * 72)

    verdict = []

    # ── ١. الحجم الصادر ──────────────────────────────────────────────
    fa, fb = pd.to_numeric(A[col['fwd_bytes']], errors='coerce'), \
             pd.to_numeric(B[col['fwd_bytes']], errors='coerce')
    print('\n١. البايتات الصادرة')
    print(f'   {"":10} {"وسيط":>12} {"p90":>12} {"p99":>14} {"أقصى":>15}')
    print(f'   {"حميد":10} {pct(fb,.5):>12,.0f} {pct(fb,.9):>12,.0f} '
          f'{pct(fb,.99):>14,.0f} {fb.max():>15,.0f}')
    print(f'   {"هجوم":10} {pct(fa,.5):>12,.0f} {pct(fa,.9):>12,.0f} '
          f'{pct(fa,.99):>14,.0f} {fa.max():>15,.0f}')
    big = float((fa > 1e6).mean() * 100)
    print(f'   تدفقات هجوم > 1 م.ب: {big:.3f}%')
    ok1 = pct(fa, .9) > pct(fb, .9)
    verdict.append(('حجم صادر أكبر من الحميد', ok1))

    # ── ٢. اللاتماثل الاتجاهي ────────────────────────────────────────
    ba, bb = pd.to_numeric(A[col['bwd_bytes']], errors='coerce'), \
             pd.to_numeric(B[col['bwd_bytes']], errors='coerce')
    ra = np.log1p(fa) - np.log1p(ba)
    rb = np.log1p(fb) - np.log1p(bb)
    print('\n٢. اللاتماثل  log(صادر) − log(وارد)   [موجب = سحب للخارج]')
    print(f'   حميد: وسيط {rb.median():+.3f}   |   هجوم: وسيط {ra.median():+.3f}')
    ok2 = ra.median() > rb.median() + 0.20
    verdict.append(('لاتماثل نحو الخارج أوضح', ok2))

    # ── ٣. الاستمرارية ───────────────────────────────────────────────
    if col['duration']:
        da = pd.to_numeric(A[col['duration']], errors='coerce') / 1e6
        db = pd.to_numeric(B[col['duration']], errors='coerce') / 1e6
        print('\n٣. مدة التدفّق (ثانية)')
        print(f'   حميد: وسيط {db.median():.2f}  p90 {pct(db,.9):.2f}')
        print(f'   هجوم: وسيط {da.median():.2f}  p90 {pct(da,.9):.2f}')
        ok3 = da.median() >= db.median()
        verdict.append(('تدفقات ليست أقصر من الحميد', ok3))

    if col['fwd_pkts'] and col['bwd_pkts']:
        tp = pd.to_numeric(A[col['fwd_pkts']], errors='coerce') + \
             pd.to_numeric(A[col['bwd_pkts']], errors='coerce')
        tiny = float((tp <= 2).mean() * 100)
        print(f'   تدفقات هجوم بحزمتين أو أقل: {tiny:.1f}%'
              f'{"   ← مؤشر مسح لا تسريب" if tiny > 30 else ""}')
        verdict.append(('نسبة التدفقات المجهرية منخفضة', tiny < 30))

    # ── ٤. التركيز ───────────────────────────────────────────────────
    if col['dst_port']:
        print('\n٤. أكثر ٨ منافذ وجهة في الهجوم')
        vc = A[col['dst_port']].astype('Int64').value_counts().head(8)
        for p, n in vc.items():
            print(f'   {p:>7} : {n:>8,}  ({n / len(A) * 100:5.1f}%)')

    # ── التجميع على مستوى المضيف ─────────────────────────────────────
    print('\n' + '=' * 72)
    if col['src_ip']:
        print(f'✅ عمود المضيف موجود ({col["src_ip"]}) — ميزات التجميع ممكنة.')
        g = A.groupby(col['src_ip'])[col['fwd_bytes']].sum().sort_values(ascending=False)
        print('   أعلى ٥ مضيفين بالحجم الصادر أثناء الهجوم:')
        for h, v in g.head(5).items():
            print(f'     {h:>18} : {v:>15,.0f} بايت')
        if col['dst_ip']:
            nd = A.groupby(col['src_ip'])[col['dst_ip']].nunique().sort_values(ascending=False)
            print(f'   أقصى عدد وجهات لمضيف واحد: {int(nd.max())}')
    else:
        print('⚠️ لا يوجد عمود Src IP — التجميع على مستوى المضيف غير ممكن من CSV.')
        print('   ستحتاجين استخراج التدفقات من PCAP عبر Zeek.')

    # ── الحكم ────────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('الحكم على بصمة الحصاد:')
    for name, ok in verdict:
        print(f'   {"✅" if ok else "❌"}  {name}')
    score = sum(ok for _, ok in verdict)
    print(f'\n   {score}/{len(verdict)} خصائص متحققة')
    if score <= 1:
        print('   ⛔ لا توجد بصمة حصاد. الفئة غالباً استطلاع أو مسح — كما في CIC-IDS2018.')
    elif score < len(verdict):
        print('   ⚠️ بصمة جزئية — تسريب خفي منخفض الحجم على الأرجح.')
        print('      هذا يدعم استخدام ميزات التجميع الزمني بدل ميزات التدفّق المفرد.')
    else:
        print('   ✅ بصمة حصاد واضحة. الكشف على مستوى التدفّق قابل للتطبيق.')


if __name__ == '__main__':
    main()
