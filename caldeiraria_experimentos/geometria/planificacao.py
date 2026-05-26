"""
Planificação geométrica das 4 peças de caldeiraria.
Gera polígonos 2D (listas de pontos) que representam a planificação
(desenvolvimento plano) de cada peça tubular.

Série ISO 4200 / ASME B36.10 — diâmetros externos nominais:
  DN25=33.7, DN50=60.3, DN80=88.9, DN100=114.3,
  DN150=168.3, DN200=219.1, DN300=323.9  (mm)
"""

import math
import numpy as np
from shapely.geometry import Polygon


# ──────────────────────────────────────────────────────────────────
# 1. CORTE INCLINADO EM TUBO
#    Planificação: retângulo com topo senoidal
#    D       = diâmetro externo do tubo (mm)
#    Alfa    = ângulo de corte em graus (ex: 45)
#    n       = número de pontos de discretização (ex: 24)
# ──────────────────────────────────────────────────────────────────
def corte_inclinado(D: float, Alfa: float = 45.0, n: int = 24) -> list[tuple[float, float]]:
    R = D / 2.0
    alfa_rad = math.radians(Alfa)
    perim = math.pi * D          # perímetro da seção circular
    H_min = 0.0                  # borda inferior plana
    H_max = D * math.tan(alfa_rad)  # altura máxima do corte

    pts = []
    # borda superior (senoidal)
    for i in range(n + 1):
        t = i / n                          # 0..1
        x = t * perim
        # desenvolvimento: ângulo θ varia de 0 a 2π ao longo do perímetro
        theta = 2 * math.pi * t
        y = H_min + (H_max / 2) * (1 - math.cos(theta))
        pts.append((x, y))
    # borda inferior (plana, de volta)
    for i in range(n, -1, -1):
        t = i / n
        x = t * perim
        pts.append((x, H_min))

    return _normalize(pts)


# ──────────────────────────────────────────────────────────────────
# 2. REDUÇÃO CONCÊNTRICA
#    Planificação: trapézio com arestas laterais curvas (cônicas)
#    D_maior = diâmetro da extremidade maior (mm)
#    D_menor = diâmetro da extremidade menor (mm)
#    H_red   = altura (comprimento) da redução (mm)
#    pts     = pontos de discretização do arco (ex: 64)
# ──────────────────────────────────────────────────────────────────
def reducao_concentrica(D_maior: float, D_menor: float,
                        H_red: float, pts: int = 64) -> list[tuple[float, float]]:
    R1 = D_maior / 2.0
    R2 = D_menor / 2.0

    # Comprimento da geratriz (slant height)
    L_slant = math.sqrt(H_red ** 2 + (R1 - R2) ** 2)

    # Ângulo total do setor no plano (cone desenvolvido)
    # Raio do setor R_sec = L_slant * R1 / (R1 - R2) se tronco
    if abs(R1 - R2) < 1e-6:
        # cilindro: retângulo simples
        perim = math.pi * D_maior
        return _normalize([(0, 0), (perim, 0), (perim, H_red), (0, H_red)])

    apex_dist1 = L_slant * R1 / (R1 - R2)   # raio externo do setor
    apex_dist2 = L_slant * R2 / (R1 - R2)   # raio interno do setor

    # Ângulo subtendido pelo setor completo (2π * R1 / apex_dist1)
    theta_total = 2 * math.pi * R1 / apex_dist1

    coords = []
    # arco externo (raio apex_dist1)
    for i in range(pts + 1):
        a = -theta_total / 2 + i * theta_total / pts
        coords.append((apex_dist1 * math.cos(a), apex_dist1 * math.sin(a)))
    # arco interno (raio apex_dist2), sentido inverso
    for i in range(pts, -1, -1):
        a = -theta_total / 2 + i * theta_total / pts
        coords.append((apex_dist2 * math.cos(a), apex_dist2 * math.sin(a)))

    return _normalize(coords)


# ──────────────────────────────────────────────────────────────────
# 3. BOCA DE LOBO TANGENTE
#    Gera DUAS peças que compõem a junta (menor e maior).
#    Para nesting retorna a peça MENOR — a peça do tubo derivado
#    (que tem a curva de intersecção mais complexa e é cortada
#    separadamente da peça maior).
#
#    Fórmulas do desenvolvimento plano (ref: app boca_de_lobo.html):
#
#    Peça Menor (tubo derivado, diâm. dp):
#      X = crp/n * i          (crp = π·dp, desenvolvimento do cilindro menor)
#      y = y1 + y2  onde:
#        db = sqrt(RG² - (rp·sin β)²)
#        y1 = (RG - db) / sin(Alfa)
#        y2 = rp·(1 - cos β)·tan(π/2 - Alfa)
#      Polígono: borda superior = perfil y(i), borda inferior = 0 (reta)
#
#    Peça Maior (tubo principal, diâm. DG):
#      X = RG·arcsin(rp·sin β / RG)
#      Y = rp·(1 - cos β)/sin(Alfa) + (RG - db)/tan(Alfa)
#      Polígono: desenvolvimento com recorte elíptico no centro
#
#    DG   = diâmetro do tubo principal (galho)
#    dp   = diâmetro do tubo derivado (ramal)
#    Alfa = ângulo de intersecção (graus, 90° = perpendicular)
#    n    = pontos de discretização
#    peca = 'menor' | 'maior'
# ──────────────────────────────────────────────────────────────────
def boca_de_lobo(DG: float, dp: float, L: float = None,
                 Alfa: float = 90.0, H: float = None,
                 n: int = 36, peca: str = 'menor') -> list[tuple[float, float]]:
    RG      = DG / 2.0
    rp      = dp / 2.0
    alfa_r  = math.radians(Alfa)
    crp     = math.pi * dp          # desenvolvimento do cilindro menor
    Beta    = 2 * math.pi / n       # incremento angular

    if peca == 'menor':
        # ── Peça Menor ──────────────────────────────────────────
        # X varia de 0 a crp (perímetro do tubo derivado)
        # y = perfil de intersecção no plano desenvolvido
        xs, ys = [], []
        for i in range(n + 1):
            beta = Beta * i
            db   = math.sqrt(max(0.0, RG**2 - (rp * math.sin(beta))**2))
            y1   = (RG - db) / math.sin(alfa_r)
            y2   = rp * (1 - math.cos(beta)) * math.tan(math.pi/2 - alfa_r)
            y    = y1 + y2
            xs.append(crp / n * i)
            ys.append(y)

        # Polígono: perfil superior + base reta
        pts_list = list(zip(xs, ys))            # borda superior (perfil)
        pts_list += [(xs[-1], 0.0), (xs[0], 0.0)]  # borda inferior plana
        return _normalize(pts_list)

    else:
        # ── Peça Maior ──────────────────────────────────────────
        # Desenvolvimento do tubo principal com recorte elíptico
        # O recorte percorre -X_max..+X_max no eixo X (centrado)
        # e tem altura Y variável
        xs_top, ys_top = [], []
        xs_bot, ys_bot = [], []
        for i in range(n + 1):
            beta = Beta * i
            db   = math.sqrt(max(0.0, RG**2 - (rp * math.sin(beta))**2))
            arg  = max(-1.0, min(1.0, rp * math.sin(beta) / RG))
            X    = math.asin(arg) * RG
            Y    = rp * (1 - math.cos(beta)) / math.sin(alfa_r) + (RG - db) / math.tan(alfa_r)
            xs_top.append(X)
            ys_top.append(Y)

        # Espelha: a peça maior é simétrica — percorre ida e volta
        # Borda externa: y = Y_max (topo plano) + perfil interno recortado
        Y_max = max(ys_top)
        # Contorno: topo plano (esquerda→direita), depois perfil (direita→esquerda)
        x_lo, x_hi = min(xs_top), max(xs_top)
        pts_list  = [(x_lo, Y_max), (x_hi, Y_max)]   # topo
        # Perfil do recorte (de volta, i decrescente → x decrescente)
        for i in range(n, -1, -1):
            pts_list.append((xs_top[i], ys_top[i]))
        pts_list.append((x_lo, Y_max))
        return _normalize(pts_list)


# ──────────────────────────────────────────────────────────────────
# 4. CURVA DE GOMOS (segmented elbow)
#    Gera DOIS tipos de gomo que compõem a curva:
#      - Gomo Parcial  (extremidades): perfil senoidal unilateral
#      - Gomo Intermediário (interno): perfil senoidal bilateral (simétrico)
#    Para nesting retorna o GOMO PARCIAL por padrão (forma mais restritiva).
#
#    Fórmulas (ref: app curva_de_gomos.html):
#      Gama    = Alfa / (2·(NG-1))  em radianos   (ângulo de corte por gomo)
#      Beta_i  = i · (2π/n)                        (ângulo da seção circular)
#      fb(i)   = tan(Gama) · (RC - RT·cos(Beta_i)) (altura no ponto i)
#      fixo    = fb(0) = tan(Gama) · (RC - RT)     (base reta)
#
#      Gomo Parcial:
#        X = Desenv·i/n,   Y = fb(i)
#        Polígono: borda superior = perfil fb, borda inferior = fb(0) (reta)
#
#      Gomo Intermediário:
#        X = Desenv·i/n,   Y_sup = fb(i),  Y_inf = -fb(i)
#        Polígono: borda superior = +fb, borda inferior = -fb (simétrico)
#
#    DT   = diâmetro do tubo (mm)
#    RC   = raio de curvatura do eixo (mm)
#    Alfa = ângulo total da curva (graus)
#    NG   = número de gomos
#    n    = pontos de discretização por gomo
#    gomo = 'parcial' | 'inter'
# ──────────────────────────────────────────────────────────────────
def curva_de_gomos(DT: float, RC: float = None, Alfa: float = 90.0,
                   NG: int = 5, n: int = 36,
                   gomo: str = 'parcial') -> list[tuple[float, float]]:
    RT = DT / 2.0
    if RC is None:
        RC = 1.5 * DT

    Gama       = math.radians(Alfa / (2 * (NG - 1)))  # ângulo de corte
    Beta_step  = 2 * math.pi / n                       # incremento angular
    Desenv     = math.pi * DT                          # perímetro = desenvolvimento

    # fb(i) = tan(Gama) · (RC - RT·cos(Beta_i))
    def fb(i):
        return math.tan(Gama) * (RC - RT * math.cos(i * Beta_step))

    fixo = fb(0)   # valor em β=0: borda inferior do gomo parcial

    if gomo == 'parcial':
        # ── Gomo Parcial ────────────────────────────────────────
        # X: 0..Desenv,  Y_sup: fb(i) (senoidal),  Y_inf: fixo (reta)
        pts_top = [(Desenv / n * i, fb(i)) for i in range(n + 1)]
        pts_bot = [(Desenv / n * i, fixo)  for i in range(n, -1, -1)]
        return _normalize(pts_top + pts_bot)

    else:
        # ── Gomo Intermediário ───────────────────────────────────
        # Y_sup = +fb(i),  Y_inf = -fb(i)  (simétrico em torno de 0)
        pts_top = [(Desenv / n * i,  fb(i)) for i in range(n + 1)]
        pts_bot = [(Desenv / n * i, -fb(i)) for i in range(n, -1, -1)]
        return _normalize(pts_top + pts_bot)


# ──────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ──────────────────────────────────────────────────────────────────
def _normalize(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Translada o polígono para que seu bounding box comece em (0,0)."""
    if not pts:
        return pts
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, min_y = min(xs), min(ys)
    return [(x - min_x, y - min_y) for x, y in pts]


def area_bbox(pts: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def to_shapely(pts: list[tuple[float, float]]) -> Polygon:
    return Polygon(pts).buffer(0)   # buffer(0) = fix invalid geometry


def poly_area(pts: list[tuple[float, float]]) -> float:
    """Área real do polígono (fórmula do shoelace)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
