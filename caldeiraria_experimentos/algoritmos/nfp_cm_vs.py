"""NFP-CM-VS — Lastra-Díaz & Ortuño (2023) — rotação + multi-chapa ilimitada"""
import time, random
from typing import Optional
from algoritmos.bl_nfp import _overlaps, _metrics, _try_place, _force_place

MAX_ITER=30; STEP=5.0

def run(pieces, W, L_max=None, gap_mm=2.0, seed=42):
    t0=time.perf_counter()
    rng=random.Random(seed)
    from algoritmos.bl_nfp import run as bl_run
    bl=bl_run(pieces,W,L_max,gap_mm)
    sheets=[list(s) for s in bl["sheets_layouts"]]

    for si,sheet in enumerate(sheets):
        if not sheet: continue
        improved=True; itr=0
        while improved and itr<MAX_ITER:
            improved=False; itr+=1; rng.shuffle(sheet)
            for i,piece in enumerate(sheet):
                others=sheet[:i]+sheet[i+1:]
                pw=piece["width"]; ph=piece["height"]
                best_y=piece["y"]; best_x=piece["x"]
                for delta in range(1,int(piece["y"]/STEP)+1):
                    ny=piece["y"]-delta*STEP
                    if ny<0: break
                    if not _overlaps(others,piece["x"],ny,pw+0.01,ph+0.01): best_y=ny
                    else: break
                for delta in range(1,int(piece["x"]/STEP)+1):
                    nx=piece["x"]-delta*STEP
                    if nx<0: break
                    if not _overlaps(others,nx,best_y,pw+0.01,ph+0.01): best_x=nx
                    else: break
                if best_y<piece["y"] or best_x<piece["x"]:
                    sheet[i]={**piece,"x":best_x,"y":best_y}; improved=True
        sheets[si]=sheet

    elapsed_ms=(time.perf_counter()-t0)*1000
    result=_metrics(sheets,pieces,W,L_max,elapsed_ms,"NFP-CM-VS")
    return result
