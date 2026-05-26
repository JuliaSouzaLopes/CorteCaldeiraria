"""
GSAO — Genetic Simulated Annealing Adaptativo (Qin, Jin & Zheng, 2021)
Com polígonos reais via Shapely.
"""
import time, random, math
from typing import Optional
from algoritmos.geom_real import PlacedPiece, rotate_polygon, place_bl, compute_metrics

POP_SIZE=10; K1=1.0; K2=0.2; ALPHA=0.7; BETA=-0.3
N_GEN=15; T0=1.0; COOL=0.95; CONV_DELTA=0.0005
THETA_G=9
ANGLES=list(range(0,360,THETA_G))  # 40 orientações

def run(pieces,W,L_max=None,gap_mm=2.0,seed=42,time_limit_s=10.0,**kwargs):
    t0=time.perf_counter()
    rng=random.Random(seed)
    n=len(pieces)
    pop=[_random_chrom(n,rng) for _ in range(POP_SIZE)]
    best_chrom=None; best_fit=-math.inf; best_sheets=[]
    prev_avg=None; T_k=T0

    for gen in range(N_GEN):
        if time.perf_counter()-t0 > time_limit_s: break
        scored=[]
        for chrom in pop:
            sheets=_decode(chrom,pieces,W,L_max,gap_mm)
            fit=_fitness(sheets,W,pieces)
            scored.append((fit,chrom,sheets))
            if fit>best_fit:
                best_fit=fit; best_chrom=chrom[:]; best_sheets=sheets
        scored.sort(key=lambda x:-x[0])
        fits=[s[0] for s in scored]
        avg_fit=sum(fits)/len(fits); f_max=fits[0]

        if prev_avg is not None and abs(avg_fit-prev_avg)<CONV_DELTA: break
        prev_avg=avg_fit

        elites=[scored[0][1]]
        pool=_roulette(scored,rng,POP_SIZE-1)

        children=[]
        for i in range(0,len(pool)-1,2):
            p1,p2=pool[i],pool[i+1]
            f_p=max(_fitness(_decode(p1,pieces,W,L_max,gap_mm),W,pieces),
                    _fitness(_decode(p2,pieces,W,L_max,gap_mm),W,pieces))
            p_c=min(1.0,max(0.0,K1*(f_p-avg_fit)/(f_max-avg_fit+1e-9) if f_p>=avg_fit else K1))
            if rng.random()<p_c: c1,c2=_two_point_crossover(p1,p2,rng)
            else: c1,c2=p1[:],p2[:]
            children.extend([c1,c2])
        if len(pool)%2==1: children.append(pool[-1])

        mutated=[]
        for chrom in children:
            f_ind=_fitness(_decode(chrom,pieces,W,L_max,gap_mm),W,pieces)
            p_m=min(0.3,max(0.1,K2*(f_max-f_ind)/(f_max-avg_fit+1e-9) if f_ind>=avg_fit else K2))
            if rng.random()<p_m:
                chrom=_mutate_seq(chrom,rng)
                chrom=_mutate_angle(chrom,rng)
            mutated.append(chrom)

        new_pop=elites[:]
        for i,new_c in enumerate(mutated[:POP_SIZE-1]):
            old_c=scored[i+1][1] if i+1<len(scored) else scored[-1][1]
            f_new=_fitness(_decode(new_c,pieces,W,L_max,gap_mm),W,pieces)
            f_old=_fitness(_decode(old_c,pieces,W,L_max,gap_mm),W,pieces)
            df=(1/(f_new+1e-9))-(1/(f_old+1e-9))
            if df<=0: new_pop.append(new_c)
            else:
                p_acc=math.exp(-df/(T_k+1e-9))
                new_pop.append(new_c if rng.random()<p_acc else old_c)
        pop=new_pop[:POP_SIZE]; T_k*=COOL

    elapsed_ms=(time.perf_counter()-t0)*1000
    return compute_metrics(best_sheets,pieces,W,L_max,elapsed_ms,"GSAO")

def _random_chrom(n,rng):
    idxs=list(range(n)); rng.shuffle(idxs)
    return list(zip(idxs,[rng.choice(ANGLES) for _ in range(n)]))

def _fitness(sheets,W,pieces):
    if not sheets: return -1e9
    M=len(sheets); last=sheets[-1]
    if not last: R=0.0
    else:
        z_last=max(pp.poly_placed.bounds[3] for pp in last)
        area_pecas=sum(pp.area for pp in last)
        R=area_pecas/(W*z_last+1e-9)
    return ALPHA*(1.0/(M+1e-9))+BETA*R

def _decode(chrom,pieces,W,L_max,gap_mm):
    remaining=list(chrom); sheets=[]
    while remaining:
        placed=[]; still=[]
        for (idx,ang) in remaining:
            p=pieces[idx]
            poly=rotate_polygon(p["polygon"],ang)
            pos=place_bl(poly,placed,W,L_max,gap_mm)
            if pos is None: still.append((idx,ang))
            else:
                x,y=pos
                placed.append(PlacedPiece(idx,x,y,ang,p["tipo"],p["label"],p["area"],poly))
        if not placed:
            for (idx,ang) in still:
                p=pieces[idx]; poly=rotate_polygon(p["polygon"],0)
                placed.append(PlacedPiece(idx,0,0,0,p["tipo"],p["label"],p["area"],poly))
            sheets.append(placed); break
        sheets.append(placed); remaining=still
    return sheets

def _roulette(scored,rng,k):
    fits=[max(0.0,s[0]) for s in scored]
    total=sum(fits)+1e-9; probs=[f/total for f in fits]
    chosen=[]
    for _ in range(k):
        r=rng.random(); cum=0.0
        for i,p in enumerate(probs):
            cum+=p
            if r<=cum: chosen.append(scored[i][1][:]); break
        else: chosen.append(scored[-1][1][:])
    return chosen

def _two_point_crossover(p1,p2,rng):
    n=len(p1)
    if n<3: return p1[:],p2[:]
    b1=rng.randint(1,n-2); b2=rng.randint(b1+1,n-1)
    seg1=p1[b1:b2]; used1={g[0] for g in seg1}
    rest1=[g for g in p2 if g[0] not in used1]
    c1=(rest1[:b1]+seg1+rest1[b1:])[:n]
    seg2=p2[b1:b2]; used2={g[0] for g in seg2}
    rest2=[g for g in p1 if g[0] not in used2]
    c2=(rest2[:b1]+seg2+rest2[b1:])[:n]
    return c1,c2

def _mutate_seq(chrom,rng):
    n=len(chrom)
    if n<2: return chrom
    c1=rng.randint(0,n-2); c2=rng.randint(c1+1,n-1)
    chrom=chrom[:]; chrom[c1],chrom[c2]=chrom[c2],chrom[c1]
    return chrom

def _mutate_angle(chrom,rng):
    n=len(chrom); d=rng.randint(0,n-1)
    chrom=chrom[:]; idx,_=chrom[d]
    chrom[d]=(idx,rng.choice(ANGLES))
    return chrom
