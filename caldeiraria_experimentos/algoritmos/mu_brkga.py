"""µ-BRKGA com polígonos reais (Junior, Pinheiro & Coelho, 2017)"""
import time, random, math
from typing import Optional
from shapely.geometry import Polygon
from algoritmos.geom_real import PlacedPiece, rotate_polygon, place_bl, compute_metrics

POP_SIZE=12; ELITE_FRAC=0.30; P_ELITE=0.70; N_GEN=10
ANGLES=[0,90,180,270]

def run(pieces,W,L_max=None,gap_mm=2.0,seed=42,time_limit_s=10.0,**kwargs):
    t0=time.perf_counter()
    rng=random.Random(seed)
    n=len(pieces)
    pop=[_rchrom(n,rng) for _ in range(POP_SIZE)]
    best_sheets=None; best_z=math.inf

    for _ in range(N_GEN):
        if time.perf_counter()-t0 > time_limit_s: break
        scored=[]
        for chrom in pop:
            sl=_decode(chrom,pieces,W,L_max,gap_mm)
            z=_z(sl)
            scored.append((z,chrom,sl))
            if z<best_z: best_z=z; best_sheets=sl
        scored.sort(key=lambda x:x[0])
        n_elite=max(1,int(POP_SIZE*ELITE_FRAC))
        elites=[s[1] for s in scored[:n_elite]]
        new_pop=elites[:]
        while len(new_pop)<POP_SIZE:
            new_pop.append(_crossover(rng.choice(elites),_rchrom(n,rng),P_ELITE,rng))
        pop=new_pop

    elapsed_ms=(time.perf_counter()-t0)*1000
    return compute_metrics(best_sheets or [],pieces,W,L_max,elapsed_ms,"µ-BRKGA")

def _rchrom(n,rng): return [rng.random() for _ in range(n)]
def _crossover(e,o,p,rng): return [ei if rng.random()<p else oi for ei,oi in zip(e,o)]

def _decode(chrom,pieces,W,L_max,gap_mm):
    order=sorted(range(len(pieces)),key=lambda i:chrom[i])
    remaining=list(order); sheets=[]
    while remaining:
        placed=[]; still=[]
        for idx in remaining:
            p=pieces[idx]
            best_pp=None; best_key=(math.inf,math.inf)
            for ang in ANGLES:
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
    total=0
    for sheet in sheets:
        if sheet: total+=max(pp.poly_placed.bounds[3] for pp in sheet)
    return total
