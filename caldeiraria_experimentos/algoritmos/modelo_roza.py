"""Modelo dos Pontos + Restrições Adicionais — Roza (2025) — rotação + multi-chapa ilimitada"""
import time, math
from typing import Optional
from algoritmos.bl_nfp import _metrics as _bl_metrics, _rotate90
from algoritmos.modelo_pontos import _build_ifp, _upper_bound, _mark_nfp, _find_bl_grid, _build_tipos_with_rot

GX=10.0; GY=10.0

def run(pieces, W, L_max=None, gap_mm=2.0, gx=GX, gy=GY):
    t0=time.perf_counter()
    L_bound=L_max or _upper_bound(pieces,W)
    order=sorted(range(len(pieces)),key=lambda i:-pieces[i]["area"])
    remaining=list(order); sheets=[]

    while remaining:
        tipos=_build_tipos_with_rot(pieces,remaining)
        tipo_list=list(tipos.keys())
        base_tipos={k.rsplit('_',1)[0] for k in tipo_list}
        # rest (8): cap by base tipo (orientation-agnostic)
        cap={}
        for tid,td in tipos.items():
            cap[tid]=max(1,math.floor(W/(td["height"]+gap_mm)))

        ifp={tid:_build_ifp(td,W,L_bound,gap_mm,gx,gy) for tid,td in tipos.items()}
        forbidden={tid:set() for tid in tipos}
        col_count={tid:{} for tid in tipos}
        placed=[]; still=[]

        for idx in remaining:
            p=pieces[idx]
            best_pt=None; best_tid=None; best_item=None
            for rot,suffix in [(0,'_0'),(90,'_90')]:
                tid=p["tipo"]+suffix
                if tid not in ifp: continue
                def valid(pt,tid=tid):
                    if pt in forbidden[tid]: return False
                    return col_count[tid].get(pt[0],0)<cap[tid]
                pt=_find_filtered(ifp[tid],valid)
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
            col_count[best_tid][best_pt[0]]=col_count[best_tid].get(best_pt[0],0)+1
            _mark_nfp(best_item,tipos,forbidden,gx,gy,gap_mm)
            # rest (7)
            for uid in tipo_list:
                if uid!=best_tid: forbidden[uid].add(best_pt)

        if not placed:
            for idx in still:
                p=pieces[idx]
                w,h=p["width"],p["height"]
                if h>w: w,h=h,w; poly=_rotate90(p["polygon"])
                else: poly=p["polygon"]
                placed.append({"idx":idx,"x":0,"y":0,"width":min(w,W),"height":h,
                    "tipo":p["tipo"],"label":p["label"],"area":p["area"],"polygon":poly,"rot":0})
            sheets.append(placed); break
        sheets.append(placed); remaining=still

    elapsed_ms=(time.perf_counter()-t0)*1000
    result=_bl_metrics(sheets,pieces,W,L_max,elapsed_ms,"Pontos+RA (Roza)")
    result["gx"]=gx; result["gy"]=gy
    # Adiciona contagem de restrições
    cc = count_constraints(pieces, W, L_max, gap_mm, gx, gy)
    result["n_restricoes"]    = cc["n_restricoes_total"]
    result["n_vars_binarias"] = cc["n_vars_binarias"]
    result["n_restricoes_detalhe"] = cc
    return result

def _find_filtered(ifp_list,valid_fn):
    for pt in ifp_list:
        if valid_fn(pt): return pt
    return None


# ──────────────────────────────────────────────────────────────────
# CONTAGEM DE RESTRIÇÕES (Roza 2025 = Toledo 2013 + Rest. 7 e 8)
# ──────────────────────────────────────────────────────────────────
def count_constraints(pieces, W, L_max=None, gap_mm=2.0, gx=GX, gy=GY):
    """
    Restrições do Modelo dos Pontos + Adicionais de Roza (2025).

    Herdadas de Toledo 2013:
      (2) z-comprimento:  Σ_t |IFP_t|
      (3) demanda:        |T|
      (4) não-sobreposição: Σ_{t≤u} Σ_{d∈IFP_t} |NFP^d_{t,u}|

    Adicionadas por Roza 2025:
      (7) incompatibilidade: |I| × |D|
            I = pares incompatíveis (todos os pares de tipos distintos)
            D = total de pontos válidos da malha
      (8) empilhamento por coluna: |T| × n_colunas
            n_colunas = int(W / gx) + 1
    """
    from algoritmos.modelo_pontos import (
        count_constraints as _base_count,
        _build_tipos_with_rot, _build_ifp, _upper_bound
    )
    import math

    # Herda todas as restrições do modelo base
    base = _base_count(pieces, W, L_max, gap_mm, gx, gy)

    L_bound   = L_max or _upper_bound(pieces, W)
    tipos     = _build_tipos_with_rot(pieces, list(range(len(pieces))))
    tipo_list = list(tipos.keys())
    tipos_base = {k.rsplit('_', 1)[0] for k in tipo_list}
    T = len(tipos_base)

    # Pontos totais da malha (D)
    n_cols = int(W / gx) + 1
    n_rows = int(L_bound / gy) + 1
    D = n_cols * n_rows

    # Restrição (7): incompatibilidade — todos os pares de tipos distintos
    # |I| = T*(T-1)/2 pares; cada par proibido em D pontos
    n_incomp_pairs = T * (T - 1) // 2
    r7 = n_incomp_pairs * D

    # Restrição (8): limite de empilhamento por coluna
    # Uma restrição por (tipo, coluna)
    r8 = T * n_cols

    r_add = r7 + r8
    r_total = base["n_restricoes_total"] + r_add

    result = dict(base)
    result.update({
        "n_restricoes_total":         r_total,
        "n_restricoes_incomp_r7":     r7,
        "n_restricoes_empilh_r8":     r8,
        "n_restricoes_adicionais":    r_add,
        "n_pares_incompativeis":      n_incomp_pairs,
    })
    return result
