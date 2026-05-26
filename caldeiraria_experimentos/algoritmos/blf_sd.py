"""
BLF-SD — Bottom-Left-Fill Semi-Discreto (Chehrazad, Roose & Wauters, 2021)
Com verificação de sobreposição por polígonos reais via Shapely.

A semi-discretização é usada para GERAR POSIÇÕES CANDIDATAS eficientemente
(eventos de resolução), mas a verificação final de sobreposição é exacta.
Isso corresponde ao espírito do paper: a representação semi-discreta serve
para tornar rápida a busca de candidatos, não para aproximar a geometria.
"""
import time, math
from typing import Optional
from shapely.geometry import Polygon
from algoritmos.geom_real import (
    PlacedPiece, rotate_polygon, can_place, compute_metrics
)

R_FACTOR = 2.0    # R = R_b × R_FACTOR (< 1 = mais posições candidatas)
ANGLES   = [0, 90, 180, 270]

def run(pieces, W, L_max=None, gap_mm=2.0, seed=42, **kwargs):
    t0 = time.perf_counter()

    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["area"])
    R = _compute_resolution(pieces, R_FACTOR)

    remaining = list(order)
    sheets: list[list[PlacedPiece]] = []

    while remaining:
        placed: list[PlacedPiece] = []
        still = []

        for idx in remaining:
            p    = pieces[idx]
            best_pp  = None
            best_key = (math.inf, math.inf)

            for ang in ANGLES:
                poly = rotate_polygon(p["polygon"], ang)
                pos  = _place_blf_sd(poly, placed, W, L_max, gap_mm, R)
                if pos is None:
                    continue
                x, y = pos
                h = poly.bounds[3]
                key = (y + h, x)
                if key < best_key:
                    best_key = key
                    best_pp  = PlacedPiece(idx, x, y, ang,
                                           p["tipo"], p["label"], p["area"], poly)

            if best_pp is None:
                still.append(idx)
            else:
                placed.append(best_pp)

        if not placed:
            for idx in still:
                p    = pieces[idx]
                poly = rotate_polygon(p["polygon"], 0)
                placed.append(PlacedPiece(idx, 0, 0, 0,
                                          p["tipo"], p["label"], p["area"], poly))
            sheets.append(placed)
            break

        sheets.append(placed)
        remaining = still

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return compute_metrics(sheets, pieces, W, L_max, elapsed_ms, "BLF-SD")


def _place_blf_sd(poly: Polygon, placed: list[PlacedPiece],
                  W: float, L_max, gap_mm: float, R: float):
    """
    Gera posições candidatas em múltiplos de R (linhas de resolução),
    e verifica cada candidato com polígono real.
    Corresponde a: x_t = m×R (Seção 2 do paper), y_t = mínimo válido.
    """
    b = poly.bounds         # (0, 0, pw, ph) após normalização
    pw = b[2]; ph = b[3]

    # Colunas candidatas: x = m×R onde a peça cabe na largura
    max_m = int((W - pw) / R) + 1
    if max_m < 0:
        return None

    # Linhas y candidatas: 0 + topos das peças colocadas (Seção 4)
    y_events = sorted({0.0} | {
        pp.poly_placed.bounds[3] + gap_mm
        for pp in placed
    })

    # Ordem de colunas: interleaved {0, max, max/2, ...} (Seção 4)
    cols = list(range(0, max_m + 1))
    cols_interleaved = _interleave(cols)

    best = None
    for m in cols_interleaved:
        x = m * R
        if x + pw > W + 1e-6:
            continue
        for y in y_events:
            if L_max and y + ph > L_max + 1e-6:
                continue
            if can_place(poly, x, y, placed, W, L_max, gap_mm):
                key = (y + ph, x)
                if best is None or key < best[0]:
                    best = (key, (x, y))
                break   # menor y válido para este x — próxima coluna

    return best[1] if best else None


def _interleave(lst):
    """Gera ordem {0, n-1, n//2, n//4, 3n//4, ...}."""
    if not lst: return []
    result = []; visited = set()
    def add(i):
        if 0 <= i < len(lst) and i not in visited:
            result.append(lst[i]); visited.add(i)
    add(0); add(len(lst)-1)
    step = len(lst)
    while step > 1:
        step //= 2
        for i in range(step, len(lst), step*2): add(i)
    for i in range(len(lst)): add(i)
    return result


def _compute_resolution(pieces, factor):
    min_pe = math.inf; min_pp = math.inf; min_ne = 1
    for p in pieces:
        poly = p["polygon"]; n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
            proj = abs(x2-x1)
            if proj > 1e-9: min_pe = min(min_pe, proj)
        xs = [pt[0] for pt in poly]
        pp = max(xs)-min(xs); ne = n
        if pp < min_pp: min_pp = pp; min_ne = ne
    if min_pe == math.inf: min_pe = 10.0
    if min_pp == math.inf: min_pp = 50.0
    return max(5.0, max(min_pe, min_pp/max(1,min_ne)) * factor)
