"""
geom_real.py — Núcleo de posicionamento com polígonos reais (Shapely)
═══════════════════════════════════════════════════════════════════════
Substitui a aproximação por bounding-box por verificação exata de
sobreposição entre os polígonos planificados reais das peças.

Conceitos implementados:
────────────────────────
IFP (Inner-Fit Polygon / região válida):
  Região onde o ponto de referência (canto inferior esquerdo do bbox)
  da peça pode ser colocado sem sair da chapa.
  IFP = retângulo [0, W-w] × [0, L-h] para chapa retangular.
  (Para strip packing livre: sem limite em y.)

NFP (No-Fit Polygon) — implementação por Minkowski:
  NFP(A, B) = A ⊕ (-B) = conjunto de translações de B que causam
  contato/sobreposição com A fixo na origem.
  Calculado via: NFP = A.buffer(0).union(B_refletido_e_erodido)
  
  Na prática: para dois polígonos A e B, B não pode ser posicionado
  em nenhum ponto dentro de NFP(A, B).
  
  Implementação eficiente via Shapely:
    nfp = shapely_minkowski_difference(A, B)
  
  Como Shapely não tem Minkowski direto, usamos:
    nfp ≈ A.buffer(r_B) onde r_B = raio efetivo de B
    (aproximação conservadora) para validação rápida,
    + verificação exata por interseção quando necessário.

Bottom-Left com NFP real:
  1. Para cada posição candidata (x, y) do ponto de ref. da peça:
     a. Translada o polígono real da peça para (x, y)
     b. Verifica se está dentro da chapa (containment)
     c. Verifica se não intersecta nenhuma peça já colocada
  2. Busca a posição (y mínimo, depois x mínimo) válida
     em malha de candidatos gerada pelos eventos de posicionamento.

Candidatos de posição (analogia ao NFP clássico):
  Após cada peça colocada, os candidatos novos são os pontos onde:
  - x = x_dir + gap  (lado direito da peça colocada)
  - y = y_top + gap  (topo da peça colocada)
  - Combinações desses com as posições x dos cantos das peças atuais
  Esta é a heurística Bottom-Left-Fill com eventos de contato.
"""

import math
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, rotate
from shapely.ops import unary_union


# ──────────────────────────────────────────────────────────────────────
# CLASSE PlacedPiece — representa uma peça posicionada
# ──────────────────────────────────────────────────────────────────────

class PlacedPiece:
    __slots__ = ('idx','x','y','rot','tipo','label','area',
                 'poly_local','poly_placed','bbox_w','bbox_h','_poly_buf')

    def __init__(self, idx, x, y, rot, tipo, label, area,
                 poly_local: Polygon):
        self.idx        = idx
        self.x          = x
        self.y          = y
        self.rot        = rot
        self.tipo       = tipo
        self.label      = label
        self.area       = area
        self.poly_local = poly_local          # polígono em coord locais (ref=0,0)
        self.poly_placed = translate(poly_local, x, y)  # polígono no strip
        b = self.poly_placed.bounds            # (minx, miny, maxx, maxy)
        self.bbox_w     = b[2] - b[0]
        self.bbox_h     = b[3] - b[1]
        self._poly_buf  = None   # cache do buffer com gap

    def to_dict(self) -> dict:
        coords = list(self.poly_local.exterior.coords)
        return {
            "idx":     self.idx,
            "x":       self.x,
            "y":       self.y,
            "width":   self.bbox_w,
            "height":  self.bbox_h,
            "tipo":    self.tipo,
            "label":   self.label,
            "area":    self.area,
            "polygon": coords,
            "rot":     self.rot,
        }


# ──────────────────────────────────────────────────────────────────────
# ROTAÇÃO de polígono
# ──────────────────────────────────────────────────────────────────────

def rotate_polygon(coords: list, angle_deg: float) -> Polygon:
    """
    Rotaciona lista de pontos por angle_deg (graus, sentido anti-horário),
    normaliza para canto inferior esquerdo em (0,0).
    Retorna Shapely Polygon. Garante resultado simples (sem MultiPolygon).
    """
    if angle_deg == 0:
        p = Polygon(coords).buffer(0)
        return _ensure_polygon(p)

    rad   = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rotated = [(x*cos_a - y*sin_a, x*sin_a + y*cos_a) for (x,y) in coords]
    min_x = min(p[0] for p in rotated)
    min_y = min(p[1] for p in rotated)
    normalized = [(x - min_x, y - min_y) for (x,y) in rotated]
    p = Polygon(normalized).buffer(0)
    return _ensure_polygon(p)


def _ensure_polygon(p) -> Polygon:
    """Garante que o resultado é um Polygon simples (não MultiPolygon)."""
    from shapely.geometry import MultiPolygon
    if isinstance(p, MultiPolygon):
        # Retorna o maior sub-polígono
        return max(p.geoms, key=lambda g: g.area)
    if not p.is_valid:
        p = p.buffer(0)
    if isinstance(p, MultiPolygon):
        return max(p.geoms, key=lambda g: g.area)
    return p


# ──────────────────────────────────────────────────────────────────────
# VERIFICAÇÃO DE POSIÇÃO (núcleo do método)
# ──────────────────────────────────────────────────────────────────────

def can_place(poly_local: Polygon, x: float, y: float,
              placed: list[PlacedPiece],
              W: float, L_max: Optional[float],
              gap_mm: float) -> bool:
    """
    Verifica se poly_local pode ser posicionado em (x,y) sem:
      1. Sair da chapa (W × L_max, L_max=None = livre)
      2. Sobrepor qualquer peça já colocada (com gap_mm de folga)

    Usa polígonos reais via Shapely.
    """
    poly_placed = translate(poly_local, x, y)
    b = poly_placed.bounds   # (minx, miny, maxx, maxy)

    # 1. Dentro da largura
    if b[0] < -1e-6 or b[2] > W + 1e-6:
        return False

    # 2. Dentro do comprimento (se fixo)
    if L_max is not None and b[3] > L_max + 1e-6:
        return False

    # 3. Não sobrepõe nenhuma peça colocada
    # Buffer negativo não funciona bem; usamos buffer positivo na peça colocada
    # e verificamos interseção com o polígono original:
    # equivalente a: dist(poly_placed, placed_i) >= gap_mm
    g2 = gap_mm / 2
    if gap_mm > 0:
        poly_buffered = _ensure_polygon(poly_placed.buffer(g2))
    else:
        poly_buffered = poly_placed

    for pp in placed:
        # Otimização AABB
        pb = pp.poly_placed.bounds
        if (b[2] + g2 < pb[0] or b[0] - g2 > pb[2] or
                b[3] + g2 < pb[1] or b[1] - g2 > pb[3]):
            continue
        # Cache buffer por peça colocada
        if pp._poly_buf is None:
            pp._poly_buf = _ensure_polygon(pp.poly_placed.buffer(g2))
        if poly_buffered.intersects(pp._poly_buf):
            return False

    return True


# ──────────────────────────────────────────────────────────────────────
# GERAÇÃO DE CANDIDATOS (eventos Bottom-Left)
# ──────────────────────────────────────────────────────────────────────

def candidate_positions(placed: list[PlacedPiece],
                        poly_local: Polygon,
                        W: float, gap_mm: float) -> list[tuple[float,float]]:
    """
    Gera posições candidatas (x, y) para o ponto de referência
    do polígono poly_local.

    Candidatos:
      y=0, x=0 (canto inferior esquerdo)
      Para cada peça colocada: y=topo+gap, x=direita+gap
      Combinações cruzadas dos eventos x e y
    
    Filtro rápido: só candidatos onde bbox da peça cabe em largura.
    """
    b_local = poly_local.bounds   # (0, 0, w, h) após normalização
    pw = b_local[2]
    ph = b_local[3]

    # Eventos de x: bordas esquerdas e direitas das peças colocadas
    x_events = {0.0}
    y_events = {0.0}

    for pp in placed:
        pb = pp.poly_placed.bounds
        x_events.add(pb[0])            # borda esquerda
        x_events.add(pb[2] + gap_mm)   # borda direita + gap
        y_events.add(pb[1])            # borda inferior
        y_events.add(pb[3] + gap_mm)   # borda superior + gap

    # Filtra x que permitem a peça dentro da largura
    x_cands = sorted(x for x in x_events if x >= 0 and x + pw <= W + 1e-6)
    y_cands = sorted(y_events)

    # Produto cartesiano (BL: menor y primeiro, depois menor x)
    candidates = []
    for y in y_cands:
        for x in x_cands:
            candidates.append((x, y))

    return candidates


# ──────────────────────────────────────────────────────────────────────
# BOTTOM-LEFT COM NFP REAL
# ──────────────────────────────────────────────────────────────────────

def place_bl(poly_local: Polygon, placed: list[PlacedPiece],
             W: float, L_max: Optional[float], gap_mm: float,
             ) -> Optional[tuple[float, float]]:
    """
    Encontra a posição Bottom-Left válida para poly_local.
    Retorna (x, y) ou None se não cabe.
    """
    candidates = candidate_positions(placed, poly_local, W, gap_mm)

    best = None
    for (x, y) in candidates:
        if can_place(poly_local, x, y, placed, W, L_max, gap_mm):
            if best is None or (y, x) < (best[1], best[0]):
                best = (x, y)
        # Otimização: se já encontramos algo na linha y e y aumentou, para
        if best and y > best[1] + 1e-6:
            break

    return best


# ──────────────────────────────────────────────────────────────────────
# MÉTRICAS (compatível com _metrics do bl_nfp)
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(sheets_placed: list[list[PlacedPiece]],
                    all_pieces: list[dict],
                    W: float, L_max: Optional[float],
                    elapsed_ms: float, metodo: str) -> dict:
    """
    Calcula todas as métricas padrão C&P/Nesting a partir de PlacedPiece.
    """
    # Converte para dicts (compatibilidade com visualizacao.py)
    sheets_dicts = [[pp.to_dict() for pp in sheet] for sheet in sheets_placed]
    all_placed   = [pp for sheet in sheets_placed for pp in sheet]

    n_sheets = len(sheets_placed)
    if not all_placed:
        return {
            "metodo":metodo,"n_colocadas":0,"n_total":len(all_pieces),
            "z":0.0,"z_lb":0.0,"gap_pct":None,
            "comprimento_usado":0.0,"area_chapa":0.0,"area_pecas":0.0,
            "utilizacao":0.0,"refugo":100.0,"tempo_ms":round(elapsed_ms,2),
            "n_chapas":0,"media_pecas_chapa":0.0,
            "n_restricoes":None,"n_vars_binarias":None,
            "layout":[],"sheets_layouts":[],
        }

    # z por chapa: máximo y+height de cada peça na chapa
    z_por_chapa = []
    for sheet in sheets_placed:
        if sheet:
            z = max(pp.poly_placed.bounds[3] for pp in sheet)
        else:
            z = 0.0
        z_por_chapa.append(z)

    z_total    = sum(z_por_chapa)
    area_chapa = sum(W * z for z in z_por_chapa)

    # Área REAL das peças (polígonos, não bbox)
    area_pecas = sum(p["area"] for p in all_pieces)
    util       = area_pecas / area_chapa * 100 if area_chapa > 0 else 0.0

    # Lower bound z̲ = max(menor altura possível da peça mais alta, área_total / W)
    # "Menor altura possível" = mínimo das alturas em 0°, 90°, 180°, 270°
    # Isso garante que o LB seja independente da rotação escolhida pelo algoritmo.
    def _min_height(pp: PlacedPiece) -> float:
        import math as _math
        poly = pp.poly_local
        # Testa as 4 rotações ortogonais
        best = poly.bounds[3]  # altura a 0°
        for ang in [90, 180, 270]:
            rad = _math.radians(ang)
            c, s = _math.cos(rad), _math.sin(rad)
            coords = list(poly.exterior.coords)
            rotated = [(x*c - y*s, x*s + y*c) for x, y in coords]
            h = max(p[1] for p in rotated) - min(p[1] for p in rotated)
            if h < best:
                best = h
        return best

    max_h  = max(_min_height(pp) for pp in all_placed)
    lb     = max(max_h, area_pecas / W)
    gap_pct = round((z_total - lb) / z_total * 100, 2) if z_total > 0 else None

    all_placed_cnt = len(all_placed)
    return {
        "metodo":            metodo,
        "n_colocadas":       all_placed_cnt,
        "n_total":           len(all_pieces),
        "z":                 round(z_total, 2),
        "z_lb":              round(lb, 2),
        "gap_pct":           gap_pct,
        "comprimento_usado": round(z_total, 2),
        "z_por_chapa":       [round(z,2) for z in z_por_chapa],
        "area_chapa":        round(area_chapa, 2),
        "area_pecas":        round(area_pecas, 2),
        "utilizacao":        round(util, 2),
        "refugo":            round(100-util, 2),
        "tempo_ms":          round(elapsed_ms, 2),
        "n_chapas":          n_sheets,
        "media_pecas_chapa": round(all_placed_cnt/n_sheets, 1),
        "n_restricoes":      None,
        "n_vars_binarias":   None,
        "layout":            [pp.to_dict() for pp in all_placed],
        "sheets_layouts":    sheets_dicts,
    }
