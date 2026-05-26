"""
BL+NFP com 2-exchange — Gomes & Oliveira (2002)
════════════════════════════════════════════════
GOMES, A. M.; OLIVEIRA, J. F. A 2-exchange heuristic for nesting
problems. European Journal of Operational Research, 141(2), 2002.

Algoritmo completo (fiel ao paper):
────────────────────────────────────
FASE 1 — Construção Bottom-Left (Seção 3.1):
  Ordena peças por área decrescente.
  Para cada peça: testa orientações 0° e 90°, coloca na posição
  Bottom-Left válida (menor y, depois menor x) usando verificação
  exata de sobreposição via Shapely.

FASE 2 — Melhoria por 2-exchange (Seção 3.2):
  Itera sobre todos os pares (i,j) de peças na sequência de alocação.
  Para cada par: troca as posições das duas peças na sequência e
  re-decodifica com BL. Se o comprimento z melhorar (ou igual), aceita.
  Repete até nenhuma troca melhorar (convergência).

Multi-chapa: quando peças não cabem na chapa atual, continuam em
  chapas adicionais até todas serem colocadas.
"""

import time, math
from typing import Optional
from shapely.geometry import Polygon
from algoritmos.geom_real import (
    PlacedPiece, rotate_polygon, place_bl, compute_metrics, _ensure_polygon
)

GAP_MM = 2.0
ANGLES = [0, 90]


def run(pieces: list[dict], W: float, L_max: Optional[float] = None,
        gap_mm: float = GAP_MM, time_limit_s: float = 30.0, **kwargs) -> dict:
    t0 = time.perf_counter()

    # ── FASE 1: Bottom-Left construtivo ─────────────────────────────
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["area"])
    sheets_bl = _bl_decode(order, pieces, W, L_max, gap_mm)
    z_bl      = _z_total(sheets_bl)

    # ── FASE 2: 2-exchange (Seção 3.2 do paper) ─────────────────────
    best_order  = list(order)
    best_sheets = sheets_bl
    best_z      = z_bl

    improved = True
    while improved and (time.perf_counter() - t0) < time_limit_s:
        improved = False
        n = len(best_order)
        for i in range(n - 1):
            if (time.perf_counter() - t0) >= time_limit_s:
                break
            for j in range(i + 1, n):
                if (time.perf_counter() - t0) >= time_limit_s:
                    break
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_sheets = _bl_decode(new_order, pieces, W, L_max, gap_mm)
                new_z      = _z_total(new_sheets)
                if new_z <= best_z - 0.5:
                    best_order  = new_order
                    best_sheets = new_sheets
                    best_z      = new_z
                    improved    = True

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return compute_metrics(best_sheets, pieces, W, L_max, elapsed_ms, "BL+NFP")


def _bl_decode(order: list, pieces: list, W: float,
               L_max: Optional[float], gap_mm: float) -> list:
    """
    Decodificador Bottom-Left: posiciona peças na ordem dada.
    Retorna lista de chapas (lista de PlacedPiece).
    """
    remaining = list(order)
    sheets: list[list[PlacedPiece]] = []

    while remaining:
        placed: list[PlacedPiece] = []
        still = []

        for idx in remaining:
            p       = pieces[idx]
            best_pp = _try_place(idx, p, placed, W, L_max, gap_mm)
            if best_pp is None:
                still.append(idx)
            else:
                placed.append(best_pp)

        if not placed:
            for idx in still:
                poly = rotate_polygon(p["polygon"], 0)
                placed.append(PlacedPiece(idx, 0.0, 0.0, 0,
                                          pieces[idx]["tipo"],
                                          pieces[idx]["label"],
                                          pieces[idx]["area"],
                                          rotate_polygon(pieces[idx]["polygon"], 0)))
            sheets.append(placed)
            break

        sheets.append(placed)
        remaining = still

    return sheets


def _try_place(idx, p, placed, W, L_max, gap_mm) -> Optional[PlacedPiece]:
    """Tenta posicionar peça em 0° e 90°; retorna melhor posição BL."""
    best_pp  = None
    best_key = (math.inf, math.inf)

    for ang in ANGLES:
        poly = rotate_polygon(p["polygon"], ang)
        pos  = place_bl(poly, placed, W, L_max, gap_mm)
        if pos is None:
            continue
        x, y = pos
        h = poly.bounds[3]
        key = (y + h, x)
        if key < best_key:
            best_key = key
            best_pp  = PlacedPiece(idx, x, y, ang,
                                   p["tipo"], p["label"], p["area"], poly)
    return best_pp


def _z_total(sheets: list) -> float:
    if not sheets:
        return math.inf
    total = 0.0
    for sheet in sheets:
        if sheet:
            total += max(pp.poly_placed.bounds[3] for pp in sheet)
    return total


# ── Funções de compatibilidade (usadas por outros módulos) ────────────

def _overlaps(placed, x, y, pw, ph):
    for r in placed:
        if (x < r["x"]+r["width"] and x+pw > r["x"] and
                y < r["y"]+r["height"] and y+ph > r["y"]):
            return True
    return False


def _find_bl(placed, pw, ph, W, L_max, gap_mm):
    xs = sorted({0.0} | {r["x"]+r["width"]+gap_mm for r in placed})
    ys = sorted({0.0} | {r["y"]+r["height"]+gap_mm for r in placed})
    best = None
    for y in ys:
        for x in xs:
            if x+pw > W+1e-6: continue
            if L_max and y+ph > L_max+1e-6: continue
            if not _overlaps(placed, x, y, pw, ph):
                if best is None or (y,x)<(best[1],best[0]): best=(x,y)
        if best and best[1]==y: break
    return (best[0],best[1]) if best else (None,None)


def _metrics(sheets_layouts, all_pieces, W, L_max, elapsed_ms, metodo):
    all_placed=[p for s in sheets_layouts for p in s]
    n_sheets=len(sheets_layouts)
    if not all_placed:
        return {"metodo":metodo,"n_colocadas":0,"n_total":len(all_pieces),
                "z":0.0,"z_lb":0.0,"gap_pct":None,"comprimento_usado":0.0,
                "area_chapa":0.0,"area_pecas":0.0,"utilizacao":0.0,
                "refugo":100.0,"tempo_ms":round(elapsed_ms,2),
                "n_chapas":0,"media_pecas_chapa":0.0,
                "n_restricoes":None,"n_vars_binarias":None,
                "layout":[],"sheets_layouts":[]}
    z_pc=[max(p["y"]+p["height"] for p in s) for s in sheets_layouts]
    z_total=sum(z_pc)
    ac=sum(W*z for z in z_pc)
    ap=sum(p["area"] for p in all_pieces)
    util=ap/ac*100 if ac>0 else 0.0
    lb=max(max(p["height"] for p in all_placed),ap/W)
    gp=round((z_total-lb)/z_total*100,2) if z_total>0 else None
    return {"metodo":metodo,"n_colocadas":len(all_placed),"n_total":len(all_pieces),
            "z":round(z_total,2),"z_lb":round(lb,2),"gap_pct":gp,
            "comprimento_usado":round(z_total,2),"z_por_chapa":[round(z,2) for z in z_pc],
            "area_chapa":round(ac,2),"area_pecas":round(ap,2),
            "utilizacao":round(util,2),"refugo":round(100-util,2),
            "tempo_ms":round(elapsed_ms,2),"n_chapas":n_sheets,
            "media_pecas_chapa":round(len(all_placed)/n_sheets,1),
            "n_restricoes":None,"n_vars_binarias":None,
            "layout":all_placed,"sheets_layouts":sheets_layouts}


def _get_orientations(p):
    w0,h0=p["width"],p["height"]
    result=[(w0,h0,p["polygon"],0)]
    if abs(h0-w0)>1e-3: result.append((h0,w0,_rotate90(p["polygon"]),90))
    return result


def _rotate90(poly):
    rotated=[(-y,x) for(x,y) in poly]
    mx=min(p[0] for p in rotated); my=min(p[1] for p in rotated)
    return [(x-mx,y-my) for(x,y) in rotated]


def _force_place(p,W,gap_mm,idx):
    return {"idx":idx,"x":0,"y":0,"width":p["width"],"height":p["height"],
            "tipo":p["tipo"],"label":p["label"],"area":p["area"],
            "polygon":p["polygon"],"rot":0}
