"""
polylabel — Implementação do algoritmo de Agafonkin (2016)
══════════════════════════════════════════════════════════
Referência: AGAFONKIN, V. Polylabel: a fast algorithm for finding
the pole of inaccessibility of a polygon.
https://github.com/mapbox/polylabel, 2016.

Citado em: GARDEYN et al. (2026), Seção 6 — os poles de inacessibilidade
são usados como proxy de detecção de colisão no sparrow/jagua-rs.

Algoritmo:
  Pole of Inaccessibility = ponto interior mais distante da borda.
  Usa busca em quadtree (priority queue de células):
  1. Cobre o bbox com células quadradas.
  2. Para cada célula: calcula distância do centro à borda do polígono.
  3. Expande células promissoras (upper bound = dist + metade do lado).
  4. Para quando upper_bound - best < precision.

Extensão para múltiplos poles (Gardeyn et al. 2026, Seção 6):
  Após encontrar o pole principal (maior círculo inscrito), remove a
  região coberta e repete para encontrar os próximos poles, até que
  o raio seja menor que min_radius. Gera um conjunto de P = {(cx,cy,r)}
  que cobre bem o interior da forma.
"""

import math
import heapq
from shapely.geometry import Polygon, Point


# ── Distância do ponto ao polígono (signed: + = dentro, - = fora) ────

def _point_to_polygon_dist(px: float, py: float, polygon) -> float:
    """
    Distância com sinal do ponto (px,py) ao polígono.
    Positiva se dentro, negativa se fora.
    Suporta Polygon e MultiPolygon.
    """
    from shapely.geometry import MultiPolygon as MP
    if isinstance(polygon, MP):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    pt = Point(px, py)
    dist_to_boundary = polygon.exterior.distance(pt)
    if polygon.contains(pt):
        return dist_to_boundary
    return -dist_to_boundary


# ── Célula da quadtree ────────────────────────────────────────────────

class _Cell:
    __slots__ = ('cx', 'cy', 'h', 'd', 'max_d')

    def __init__(self, cx, cy, h, polygon):
        self.cx = cx
        self.cy = cy
        self.h  = h
        try:
            self.d = _point_to_polygon_dist(cx, cy, polygon)
        except Exception:
            self.d = -h
        self.max_d = self.d + h * math.sqrt(2)

    def __lt__(self, other):
        # Heap de máximos: negativo para simular max-heap com heapq (min-heap)
        return self.max_d > other.max_d


# ── Algoritmo principal (pole único) ─────────────────────────────────

def polylabel(polygon: Polygon, precision: float = 1.0) -> tuple:
    """
    Encontra o pole de inacessibilidade (maior círculo inscrito) do polígono.

    Parâmetros:
        polygon   : Shapely Polygon
        precision : precisão desejada em unidades do polígono

    Retorna (cx, cy, radius).
    """
    minx, miny, maxx, maxy = polygon.bounds
    width  = maxx - minx
    height = maxy - miny
    cell_size = min(width, height)

    if cell_size == 0:
        return (minx, miny, 0.0)

    h = cell_size / 2.0

    # Candidato inicial: centroide
    centroid = polygon.centroid
    best_cx, best_cy = centroid.x, centroid.y
    best_d = _point_to_polygon_dist(best_cx, best_cy, polygon)

    # Candidato pela bbox center
    bbox_cx = (minx + maxx) / 2
    bbox_cy = (miny + maxy) / 2
    bbox_d  = _point_to_polygon_dist(bbox_cx, bbox_cy, polygon)
    if bbox_d > best_d:
        best_cx, best_cy, best_d = bbox_cx, bbox_cy, bbox_d

    # Priority queue (max-heap por max_d)
    pq = []

    # Preenche grid inicial com células de tamanho h
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = _Cell(x + h, y + h, h, polygon)
            heapq.heappush(pq, cell)
            y += cell_size
        x += cell_size

    # Busca
    while pq:
        cell = heapq.heappop(pq)

        # Atualiza melhor
        if cell.d > best_d:
            best_cx, best_cy, best_d = cell.cx, cell.cy, cell.d

        # Poda: se o upper bound não melhora dentro da precisão, descarta
        if cell.max_d - best_d <= precision:
            continue

        # Subdivide em 4 células menores
        h2 = cell.h / 2
        for dcx, dcy in [(-h2, -h2), (h2, -h2), (-h2, h2), (h2, h2)]:
            child = _Cell(cell.cx + dcx, cell.cy + dcy, h2, polygon)
            heapq.heappush(pq, child)

    return (best_cx, best_cy, max(0.0, best_d))


# ── Múltiplos poles (Gardeyn et al. 2026, Seção 6) ───────────────────

def compute_poles(polygon: Polygon,
                  n_poles: int = 12,
                  precision: float = None,
                  min_radius_frac: float = 0.02) -> list:
    """
    Calcula um conjunto de poles de inacessibilidade para o polígono,
    como descrito em Gardeyn et al. (2026) e Agafonkin (2016).

    O pole principal é o maior círculo inscrito. Poles subsequentes são
    encontrados na geometria residual após remover a área do círculo anterior.

    Parâmetros:
        polygon        : Shapely Polygon
        n_poles        : número máximo de poles a calcular
        precision      : precisão da busca (default: 1% da menor dimensão)
        min_radius_frac: raio mínimo como fração da maior dimensão
                         (poles menores que isso são ignorados)

    Retorna lista de (cx, cy, radius) ordenada por raio decrescente.
    """
    if not polygon.is_valid or polygon.is_empty:
        return []

    minx, miny, maxx, maxy = polygon.bounds
    size = max(maxx - minx, maxy - miny)

    if precision is None:
        precision = size * 0.01   # 1% da maior dimensão

    min_radius = size * min_radius_frac

    poles = []
    remaining = polygon

    for _ in range(n_poles):
        if remaining.is_empty or remaining.area < 1e-6:
            break

        # Garante Polygon simples
        if remaining.geom_type == 'MultiPolygon':
            remaining = max(remaining.geoms, key=lambda g: g.area)
        if not remaining.is_valid:
            remaining = remaining.buffer(0)
            if remaining.geom_type == 'MultiPolygon':
                remaining = max(remaining.geoms, key=lambda g: g.area)

        cx, cy, r = polylabel(remaining, precision)

        if r < min_radius:
            break

        poles.append((cx, cy, r))

        # Remove círculo do polo da geometria residual
        try:
            circle = Point(cx, cy).buffer(r)
            remaining = remaining.difference(circle)
            if remaining.is_empty:
                break
        except Exception:
            break

    return poles


# ── Cache de poles por polígono ───────────────────────────────────────

_poles_cache: dict = {}

def get_poles(polygon: Polygon, n_poles: int = 12) -> list:
    """
    Retorna poles com cache (evita recomputar para o mesmo polígono).
    A chave é o hash das coordenadas do exterior.
    """
    key = hash(polygon.wkt[:200])   # hash do WKT truncado
    if key not in _poles_cache:
        _poles_cache[key] = compute_poles(polygon, n_poles=n_poles)
    return _poles_cache[key]
