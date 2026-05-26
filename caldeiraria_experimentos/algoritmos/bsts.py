"""BS+TS com polígonos reais (Rao, Wang & Luo, 2021)"""
import time, random, math
from typing import Optional
from algoritmos.geom_real import PlacedPiece, rotate_polygon, place_bl, compute_metrics

BEAM_WIDTH=2; N_CAND=2; MAX_TS_ITER=5; TABU_TENURE=8
ANGLES=[0,90,180,270]

def run(pieces,W,L_max=None,gap_mm=2.0,seed=42,time_limit_s=10.0,**kwargs):
    t0=time.perf_counter()
    rng=random.Random(seed)
    n=len(pieces)

    beam=[{"seq":[],"z":0.0}]
    for level in range(n):
        cands=[]
        for state in beam:
            used=set(state["seq"])
            remaining=[i for i in range(n) if i not in used]
            if not remaining: continue
            top=sorted(remaining,key=lambda i:-pieces[i]["area"])[:N_CAND]
            for nxt in top:
                ns=state["seq"]+[nxt]
                sl=_decode_seq(ns,pieces,W,L_max,gap_mm)
                z=_z(sl)
                cands.append({"seq":ns,"z":z})
        if not cands: break
        cands.sort(key=lambda s:s["z"]); beam=cands[:BEAM_WIDTH]

    best_seq=beam[0]["seq"] if beam else list(range(n))
    best_z=beam[0]["z"] if beam else math.inf

    cur_seq=best_seq[:]; tabu=[]
    for _ in range(MAX_TS_ITER):
        if time.perf_counter()-t0 > time_limit_s: break
        best_move=None; best_mz=math.inf
        m=len(cur_seq)
        for i in range(m-1):
            for j in range(i+1,m):
                move=(cur_seq[i],cur_seq[j])
                ns=cur_seq[:]; ns[i],ns[j]=ns[j],ns[i]
                sl=_decode_seq(ns,pieces,W,L_max,gap_mm); nz=_z(sl)
                is_tabu=move in tabu or (move[1],move[0]) in tabu
                if (not is_tabu or nz<best_z) and nz<best_mz:
                    best_move=(i,j,move,ns,nz); best_mz=nz
        if best_move is None: break
        i,j,move,cur_seq,nz=best_move
        tabu.append(move)
        if len(tabu)>TABU_TENURE: tabu.pop(0)
        if nz<best_z: best_z=nz; best_seq=cur_seq[:]

    final=_decode_seq(best_seq,pieces,W,L_max,gap_mm)
    elapsed_ms=(time.perf_counter()-t0)*1000
    return compute_metrics(final,pieces,W,L_max,elapsed_ms,"BS+TS")

def _decode_seq(seq,pieces,W,L_max,gap_mm):
    remaining=list(seq); sheets=[]
    while remaining:
        placed=[]; still=[]
        for idx in remaining:
            p=pieces[idx]
            best_pp=None; best_key=(math.inf,math.inf)
            for ang in [0,90]:
                poly=rotate_polygon(p["polygon"],ang)
                pos=place_bl(poly,placed,W,L_max,gap_mm)
                if pos:
                    x,y=pos; h=poly.bounds[3]
                    if (y+h,x)<best_key:
                        best_key=(y+h,x)
                        best_pp=PlacedPiece(idx,x,y,ang,p["tipo"],p["label"],p["area"],poly)
            if best_pp is None: still.append(idx)
            else: placed.append(best_pp)
        if not placed:
            for idx in still:
                p=pieces[idx]; poly=rotate_polygon(p["polygon"],0)
                placed.append(PlacedPiece(idx,0,0,0,p["tipo"],p["label"],p["area"],poly))
            sheets.append(placed); break
        sheets.append(placed); remaining=still
    return sheets

def _z(sheets):
    if not sheets: return math.inf
    return sum(max(pp.poly_placed.bounds[3] for pp in s) for s in sheets if s)
