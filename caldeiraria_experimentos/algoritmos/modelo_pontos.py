"""Modelo dos Pontos — Toledo et al. (2013) — rotação + multi-chapa ilimitada"""
import time, math
from typing import Optional
from algoritmos.bl_nfp import _metrics as _bl_metrics, _rotate90

GX=10.0; GY=10.0

def run(pieces, W, L_max=None, gap_mm=2.0, gx=GX, gy=GY):
    t0=time.perf_counter()
    L_bound=L_max or _upper_bound(pieces,W)
    order=sorted(range(len(pieces)),key=lambda i:-pieces[i]["area"])
    remaining=list(order); sheets=[]

    while remaining:
        # Build tipos considering both orientations
        tipos=_build_tipos_with_rot(pieces,remaining)
        ifp={tid:_build_ifp(td,W,L_bound,gap_mm,gx,gy) for tid,td in tipos.items()}
        forbidden={tid:set() for tid in tipos}
        placed=[]; still=[]

        for idx in remaining:
            p=pieces[idx]
            # Try both orientations, pick best (lowest row, then col)
            best_pt=None; best_tid=None; best_item=None
            for rot,suffix in [(0,'_0'),(90,'_90')]:
                tid=p["tipo"]+suffix
                if tid not in ifp: continue
                pt=_find_bl_grid(ifp[tid],forbidden[tid])
                if pt is not None:
                    if best_pt is None or pt<best_pt:
                        best_pt=pt; best_tid=tid
                        pw=tipos[tid]["width"]; ph=tipos[tid]["height"]
                        poly=tipos[tid]["polygon"]
                        best_item={"idx":idx,"x":pt[0]*gx,"y":pt[1]*gy,
                            "width":pw,"height":ph,"tipo":p["tipo"],"label":p["label"],
                            "area":p["area"],"polygon":poly,"col":pt[0],"row":pt[1],"rot":rot}
            if best_item is None: still.append(idx); continue
            placed.append(best_item)
            try: ifp[best_tid].remove(best_pt)
            except ValueError: pass
            _mark_nfp(best_item,tipos,forbidden,gx,gy,gap_mm)

        if not placed:
            for idx in still:
                p=pieces[idx]
                w,h=p["width"],p["height"]
                if h>w and w<=W: w,h=h,w; poly=_rotate90(p["polygon"])
                else: poly=p["polygon"]
                placed.append({"idx":idx,"x":0,"y":0,"width":min(w,W),"height":h,
                    "tipo":p["tipo"],"label":p["label"],"area":p["area"],"polygon":poly,"rot":0})
            sheets.append(placed); break
        sheets.append(placed); remaining=still

    elapsed_ms=(time.perf_counter()-t0)*1000
    result=_bl_metrics(sheets,pieces,W,L_max,elapsed_ms,"Modelo dos Pontos")
    result["gx"]=gx; result["gy"]=gy
    # Adiciona contagem de restrições
    cc = count_constraints(pieces, W, L_max, gap_mm, gx, gy)
    result["n_restricoes"]    = cc["n_restricoes_total"]
    result["n_vars_binarias"] = cc["n_vars_binarias"]
    result["n_restricoes_detalhe"] = cc
    return result

def _build_tipos_with_rot(pieces,indices):
    t={}
    for idx in indices:
        p=pieces[idx]
        tid0=p["tipo"]+"_0"
        if tid0 not in t:
            t[tid0]={"width":p["width"],"height":p["height"],"area":p["area"],"polygon":p["polygon"],"rot":0}
        tid90=p["tipo"]+"_90"
        if tid90 not in t:
            p90w,p90h=p["height"],p["width"]
            t[tid90]={"width":p90w,"height":p90h,"area":p["area"],"polygon":_rotate90(p["polygon"]),"rot":90}
    return t

def _build_ifp(tdata,W,L_bound,gap_mm,gx,gy):
    w_t=tdata["width"]+gap_mm; h_t=tdata["height"]+gap_mm
    cols=max(0,int((W-w_t)/gx)); rows=max(0,int((L_bound-h_t)/gy))
    return [(c,r) for r in range(rows+1) for c in range(cols+1)]

def _upper_bound(pieces,W):
    return sum(p["height"] for p in pieces)*2

def _find_bl_grid(ifp_list,forbidden):
    for pt in ifp_list:
        if pt not in forbidden: return pt
    return None

def _mark_nfp(item,tipos,forbidden,gx,gy,gap_mm):
    x_d=item["col"]*gx; y_d=item["row"]*gy
    w_t=item["width"]+gap_mm; h_t=item["height"]+gap_mm
    for uid,ud in tipos.items():
        w_u=ud["width"]+gap_mm; h_u=ud["height"]+gap_mm
        c_lo=max(0,math.ceil((x_d-w_u+gap_mm)/gx))
        c_hi=max(0,math.floor((x_d+w_t-gap_mm)/gx))
        r_lo=max(0,math.ceil((y_d-h_u+gap_mm)/gy))
        r_hi=max(0,math.floor((y_d+h_t-gap_mm)/gy))
        for r in range(r_lo,r_hi+1):
            for c in range(c_lo,c_hi+1):
                forbidden[uid].add((c,r))


# ──────────────────────────────────────────────────────────────────
# CONTAGEM DE RESTRIÇÕES (formulação MIP — Toledo 2013)
# ──────────────────────────────────────────────────────────────────
def count_constraints(pieces, W, L_max=None, gap_mm=2.0, gx=GX, gy=GY):
    """
    Calcula analiticamente os contadores da formulação MIP do
    Modelo dos Pontos (Toledo et al., 2013).

    Restrições (2): uma por (tipo, ponto_IFP) — controla comprimento z.
                    = Σ_t |IFP_t|
    Restrições (3): uma por tipo — garante demanda.
                    = |T|
    Restrições (4): uma por (t, u, d∈IFP_t, e∈NFP^d_{t,u}).
                    = Σ_{t≤u} Σ_{d∈IFP_t} |NFP^d_{t,u}|
                    Pior caso: O(T²D²).
    Variáveis bin.: Σ_t |IFP_t|
    Variáveis cont.: 1  (z)

    NFP^d_{t,u} é aproximado pela bounding-box (igual ao usado no
    algoritmo): retângulo de (w_t + w_u - 2·gap) × (h_t + h_u - 2·gap)
    centrado em d, intersectado com os pontos válidos da malha.
    """
    L_bound = L_max or _upper_bound(pieces, W)
    tipos   = _build_tipos_with_rot(pieces, list(range(len(pieces))))
    tipo_list = list(tipos.keys())

    # IFP: lista de pontos para cada tipo (orientação 0° e 90°)
    ifp = {tid: _build_ifp(td, W, L_bound, gap_mm, gx, gy)
           for tid, td in tipos.items()}

    # Restrições (2): Σ |IFP_t|
    r2 = sum(len(pts) for pts in ifp.values())

    # Restrições (3): |T| (tipos base, sem considerar rotações como tipos extras)
    tipos_base = {k.rsplit('_', 1)[0] for k in tipo_list}
    r3 = len(tipos_base)

    # Variáveis binárias: Σ |IFP_t|
    n_vars_bin = r2
    n_vars_cont = 1  # z

    # Restrições (4): para cada par (t, u) com t ≤ u e cada d ∈ IFP_t,
    # conta quantos pontos caem no NFP^d_{t,u} (bbox sobre malha)
    n_cols_total = max(1, int(W / gx) + 1)
    n_rows_total = max(1, int(L_bound / gy) + 1)

    r4 = 0
    for i, tid in enumerate(tipo_list):
        wt = tipos[tid]["width"]  + gap_mm
        ht = tipos[tid]["height"] + gap_mm
        ifp_t = ifp[tid]
        if not ifp_t:
            continue
        for j, uid in enumerate(tipo_list):
            if j < i:
                continue   # t ≤ u (evita contar par duas vezes)
            wu = tipos[uid]["width"]  + gap_mm
            hu = tipos[uid]["height"] + gap_mm
            # Tamanho do NFP^d_{t,u} em número de pontos (bbox)
            # Largura: w_t + w_u - 2·gap (em mm) → colunas
            nfp_cols = max(0, math.floor((wt + wu - 2*gap_mm) / gx))
            nfp_rows = max(0, math.floor((ht + hu - 2*gap_mm) / gy))
            nfp_size = (nfp_cols + 1) * (nfp_rows + 1)
            # |IFP_t| × |NFP^d_{t,u}|
            r4 += len(ifp_t) * nfp_size

    r_total = r2 + r3 + r4

    return {
        "n_restricoes_total":    r_total,
        "n_restricoes_obj":      r2,      # (2) comprimento
        "n_restricoes_demanda":  r3,      # (3) demanda
        "n_restricoes_nao_sob":  r4,      # (4) não-sobreposição
        "n_vars_binarias":       n_vars_bin,
        "n_vars_continuas":      n_vars_cont,
        "n_pontos_malha":        n_cols_total * n_rows_total,
        "n_tipos":               len(tipos_base),
        "gx": gx, "gy": gy,
    }
