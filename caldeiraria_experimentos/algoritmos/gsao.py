"""
GSAO — Genetic Simulated Annealing Adaptativo (Qin, Jin & Zheng, 2021)
Com polígonos reais via Shapely.

Adaptações para chapas retangulares (vs. chapas irregulares de couro do paper):
- Ângulos restringidos a múltiplos de 90° (0°, 90°, 180°, 270°) em vez de
  múltiplos de 9° (40 ângulos). Para chapas retangulares, ângulos oblíquos
  aumentam a bounding box e desperdiçam espaço.
- Cache de fitness por cromossomo para evitar redecodificações redundantes.
- Decodificador BL com verificação Shapely exacta.
"""
import time, random, math
from typing import Optional
from algoritmos.geom_real import PlacedPiece, rotate_polygon, place_bl, compute_metrics

# ── Parâmetros ───────────────────────────────────────────────────────
POP_SIZE   = 20
K1         = 1.0     # constante crossover (Eq. 3)
K2         = 0.2     # constante mutação (Eq. 4, ∈ [0.1, 0.3])
ALPHA      = 0.7     # peso da função fitness (chapas) (Eq. 1)
BETA       = -0.3    # peso da função fitness (utilização última chapa) (Eq. 1)
N_GEN      = 60
T0         = 1.0     # temperatura inicial SA
COOL       = 0.95    # fator de resfriamento
CONV_DELTA = 0.0005  # critério de convergência (Seção 4.6)

# Ângulos para chapas retangulares: múltiplos de 90°
# (paper usa θ_g=9° para chapas irregulares de couro)
ANGLES = [0, 90, 180, 270]


def run(pieces, W, L_max=None, gap_mm=2.0, seed=42, time_limit_s=10.0, **kwargs):
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = len(pieces)

    # Inicializa população:
    # Metade com ordem por área (como BL) para ter bom ponto de partida
    # Metade aleatória para diversidade
    order_by_area = sorted(range(n), key=lambda i: -pieces[i]["area"])
    def bl_chrom():
        return [(idx, rng.choice(ANGLES)) for idx in order_by_area]

    pop = [bl_chrom()] + [_random_chrom(n, rng) for _ in range(POP_SIZE - 1)]
    best_chrom = None
    best_fit   = -math.inf
    best_sheets = []
    prev_avg   = None
    T_k        = T0

    # Cache: chave=(seq,angs) → (fit, sheets)
    cache = {}

    def eval_chrom(chrom):
        key = tuple((idx, ang) for idx, ang in chrom)
        if key not in cache:
            sheets = _decode(chrom, pieces, W, L_max, gap_mm)
            fit    = _fitness(sheets, W, pieces)
            cache[key] = (fit, sheets)
        return cache[key]

    for gen in range(N_GEN):
        if time.perf_counter() - t0 > time_limit_s:
            break

        # Avalia
        scored = []
        for chrom in pop:
            fit, sheets = eval_chrom(chrom)
            scored.append((fit, chrom, sheets))
            if fit > best_fit:
                best_fit = fit
                best_chrom = chrom[:]
                best_sheets = sheets

        scored.sort(key=lambda x: -x[0])
        fits    = [s[0] for s in scored]
        avg_fit = sum(fits) / len(fits)
        f_max   = fits[0]

        # Critério de convergência (Seção 4.6)
        if prev_avg is not None and abs(avg_fit - prev_avg) < CONV_DELTA:
            break
        prev_avg = avg_fit

        # Seleção: elitismo + roulette (Eq. 2)
        elites = [scored[0][1]]
        pool   = _roulette(scored, rng, POP_SIZE - 1)

        # Crossover 2-pontos adaptativo (Eq. 3)
        children = []
        for i in range(0, len(pool) - 1, 2):
            p1, p2 = pool[i], pool[i+1]
            f1, _ = eval_chrom(p1)
            f2, _ = eval_chrom(p2)
            f_prime = max(f1, f2)
            if f_max > avg_fit + 1e-9:
                p_c = min(1.0, K1 * (f_prime - avg_fit) / (f_max - avg_fit))
            else:
                p_c = K1
            if rng.random() < p_c:
                c1, c2 = _two_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]
            children.extend([c1, c2])
        if len(pool) % 2 == 1:
            children.append(pool[-1])

        # Mutação adaptativa (Eq. 4)
        mutated = []
        for chrom in children:
            f_ind, _ = eval_chrom(chrom)
            if f_max > avg_fit + 1e-9:
                p_m = min(0.3, max(0.1, K2 * (f_max - f_ind) / (f_max - avg_fit)))
            else:
                p_m = K2
            if rng.random() < p_m:
                chrom = _mutate_seq(chrom, rng)
                chrom = _mutate_angle(chrom, rng)
            mutated.append(chrom)

        # SA: aceita/rejeita (Seção 4.5, Eq. 5-6)
        new_pop = elites[:]
        for i, new_c in enumerate(mutated[:POP_SIZE - 1]):
            old_c = scored[i+1][1] if i+1 < len(scored) else scored[-1][1]
            f_new, _ = eval_chrom(new_c)
            f_old, _ = eval_chrom(old_c)
            # ΔF = 1/F_new - 1/F_old (Eq. 5)
            df = (1 / (f_new + 1e-9)) - (1 / (f_old + 1e-9))
            if df <= 0:
                new_pop.append(new_c)
            else:
                p_acc = math.exp(-df / (T_k + 1e-9))  # Eq. 6
                new_pop.append(new_c if rng.random() < p_acc else old_c)

        pop = new_pop[:POP_SIZE]
        T_k *= COOL

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return compute_metrics(best_sheets, pieces, W, L_max, elapsed_ms, "GSAO")


# ── Funções auxiliares ───────────────────────────────────────────────

def _random_chrom(n, rng):
    """Cromossomo: lista de (idx_peça, ângulo°)."""
    idxs = list(range(n))
    rng.shuffle(idxs)
    return [(idx, rng.choice(ANGLES)) for idx in idxs]


def _fitness(sheets, W, pieces):
    """
    Fitness (Eq. 1 adaptada):
    Para chapas fixas (multi-sheet): F = α*(1/M) + β*R  (paper original)
    Para strip packing (comprimento livre): F = utilização total
    Maximizar fitness = minimizar comprimento usado / maximizar aproveitamento.
    """
    if not sheets:
        return -1e9
    all_placed = [pp for sh in sheets for pp in sh]
    if not all_placed:
        return -1e9
    M = len(sheets)
    z_total = sum(max(pp.poly_placed.bounds[3] for pp in sh) for sh in sheets if sh)
    area_total = sum(pp.area for pp in all_placed)
    utilizacao = area_total / (W * z_total + 1e-9)
    # Fitness composto: premia alta utilização e poucas chapas
    last = sheets[-1]
    z_last = max(pp.poly_placed.bounds[3] for pp in last) if last else 1e9
    area_last = sum(pp.area for pp in last)
    R = area_last / (W * z_last + 1e-9)
    return ALPHA * (1.0 / (M + 1e-9)) + BETA * (1.0 - utilizacao)


def _decode(chrom, pieces, W, L_max, gap_mm):
    """
    Decodifica cromossomo usando Bottom-Left com Shapely.
    Peças são posicionadas na ordem do cromossomo com a rotação especificada.
    """
    remaining = list(chrom)
    sheets    = []

    while remaining:
        placed: list[PlacedPiece] = []
        still  = []

        for (idx, ang) in remaining:
            p = pieces[idx]
            # Testa o ângulo do cromossomo primeiro; se não couber,
            # tenta as outras rotações ortogonais antes de desistir
            best_pp = None
            best_key = (float('inf'), float('inf'))
            for a in ([ang] + [a for a in ANGLES if a != ang]):
                poly = rotate_polygon(p["polygon"], a)
                pos  = place_bl(poly, placed, W, L_max, gap_mm)
                if pos is not None:
                    x, y = pos
                    h = poly.bounds[3]
                    key = (y + h, x)
                    if key < best_key:
                        best_key = key
                        best_pp  = PlacedPiece(idx, x, y, a,
                                               p["tipo"], p["label"],
                                               p["area"], poly)
                    break  # aceita a primeira rotação válida (BL já é greedy)
            if best_pp is None:
                still.append((idx, ang))
            else:
                placed.append(best_pp)

        if not placed:
            # Força restantes em y=0 (fallback — não deveria acontecer com L=None)
            for (idx, ang) in still:
                p    = pieces[idx]
                poly = rotate_polygon(p["polygon"], ang)
                placed.append(PlacedPiece(idx, 0.0, 0.0, ang,
                                          p["tipo"], p["label"],
                                          p["area"], poly))
            sheets.append(placed)
            break

        sheets.append(placed)
        remaining = still

    return sheets


def _roulette(scored, rng, k):
    """Seleção proporcional ao fitness (Eq. 2)."""
    fits  = [max(0.0, s[0]) for s in scored]
    total = sum(fits) + 1e-9
    probs = [f / total for f in fits]
    chosen = []
    for _ in range(k):
        r = rng.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen.append(scored[i][1][:])
                break
        else:
            chosen.append(scored[-1][1][:])
    return chosen


def _two_point_crossover(p1, p2, rng):
    """
    Crossover 2-pontos (Seção 4.4.1):
    Segmento [b1,b2) de X1 vai para X1_new;
    restante copiado na ordem em que aparece em X2.
    """
    n = len(p1)
    if n < 3:
        return p1[:], p2[:]
    b1 = rng.randint(1, n - 2)
    b2 = rng.randint(b1 + 1, n - 1)

    seg1  = p1[b1:b2]
    used1 = {g[0] for g in seg1}
    rest1 = [g for g in p2 if g[0] not in used1]
    c1    = (rest1[:b1] + seg1 + rest1[b1:])[:n]

    seg2  = p2[b1:b2]
    used2 = {g[0] for g in seg2}
    rest2 = [g for g in p1 if g[0] not in used2]
    c2    = (rest2[:b1] + seg2 + rest2[b1:])[:n]

    return c1, c2


def _mutate_seq(chrom, rng):
    """Mutação sequencial: troca dois genes de posição (Seção 4.4.2)."""
    n = len(chrom)
    if n < 2:
        return chrom
    c1 = rng.randint(0, n - 2)
    c2 = rng.randint(c1 + 1, n - 1)
    chrom = chrom[:]
    chrom[c1], chrom[c2] = chrom[c2], chrom[c1]
    return chrom


def _mutate_angle(chrom, rng):
    """Mutação de ângulo: substitui ângulo de um gene aleatório (Seção 4.4.2)."""
    n = len(chrom)
    d = rng.randint(0, n - 1)
    chrom = chrom[:]
    idx, _ = chrom[d]
    chrom[d] = (idx, rng.choice(ANGLES))
    return chrom
