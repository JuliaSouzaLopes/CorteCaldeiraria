"""
sparrow — GLS + Problemas de Factibilidade Sequenciais
(Gardeyn, Vanden Berghe & Wauters, 2026)
════════════════════════════════════════════════════════
Implementação fiel com poles de inacessibilidade reais (polylabel).

Fidelidade ao paper:
─────────────────────────────────────────────────────────────────────
• Poles de inacessibilidade calculados via polylabel (Agafonkin 2016)
  — igual ao jagua-rs descrito na Seção 5-6 do paper.
• overlap_proxy_decay (Alg. 3): penetration depth entre pares de poles,
  com decaimento hiperbólico ε²/(−δ+2ε) para colisões menores.
• quantify_collision (Alg. 4): √α × √(λa × λb), onde
  λ = √(área do convex hull) e α = overlap_proxy_decay.
• evaluate_item_pair (Alg. 1): verifica colisão via can_place/Shapely,
  quantifica com quantify_collision.
• evaluate_sample (Alg. 7): Σ_c w_ic × quantify_collision(pos_i, pos_c)
• update_weights (Alg. 8): m ∈ [Ml, Mu] para colisores, Md para não.
• search_position (Alg. 6): Tdiv (amostras uniformes no strip) +
  Tfoc (amostras focais) + refinamento por descida de coordenadas
  (Loshchilov et al., 2011) dos N_REF mais promissores.
• separate (Alg. 9): local search com strikes e restauro do best.
• explore (Alg. 12): BLF inicial → shrink → separate → pool de infactíveis.
• compress (Alg. 13): shrink progressivo de R_s_c a R_e_c.

Parâmetros (Tabela 1 do paper, adaptados para Python single-thread):
  R_ε=1%, (Mu,Ml,Md)=(2.0,1.2,0.95), N_poles=12
  (Kx,Nx)=(3,30), (Kc,Nc)=(3,15)
  Rx=0.005, (R_s_c,R_e_c)=(0.002,0.0005)
  TL: 75% explore, 25% compress
"""

import time, random, math
from typing import Optional
from shapely.geometry import Polygon, Point
from shapely.affinity import translate

from algoritmos.geom_real import (
    PlacedPiece, rotate_polygon, can_place, compute_metrics, place_bl
)
from algoritmos.polylabel_impl import get_poles

# ── Parâmetros ───────────────────────────────────────────────────────
R_EPS    = 0.01        # ε = R_ε × max_diameter  (Alg. 3)
MU, ML, MD = 2.0, 1.2, 0.95  # weight multipliers  (Alg. 8, Tabela 1)
N_POLES  = 12          # número de poles por peça  (Seção 6)
KX, NX   = 3, 30       # parâmetros separate - explore  (Alg. 12)
KC, NC   = 3, 15       # parâmetros separate - compress  (Alg. 13)
RX       = 0.005       # shrink ratio explore  (Alg. 12)
RS_C, RE_C = 0.002, 0.0005  # shrink range compress  (Alg. 13)
TL_X, TL_C = 0.75, 0.25    # frações do tempo  (Alg. 11)
N_DIV, N_FOC, N_REF = 10, 8, 3  # amostragem  (Alg. 6, Fig. 7)
REFINE_STEPS = 12      # passos de refinamento  (Seção 7)
ANGLES = [0, 90, 180, 270]


def run(pieces: list[dict], W: float, L_max: Optional[float] = None,
        gap_mm: float = 2.0, seed: int = 42,
        time_limit_s: float = 3.0, **kwargs) -> dict:
    t0 = time.perf_counter()
    rng = random.Random(seed)

    piece_poles = _precompute_poles(pieces)

    if L_max is not None:
        # ── MODO MULTI-CHAPA (chapa fixa) ────────────────────────────
        # Estima número de chapas necessárias para distribuir o orçamento
        # de tempo total entre elas, respeitando time_limit_s global.
        total_area_pieces = sum(p["area"] for p in pieces)
        area_sheet = W * L_max
        n_sheets_est = max(1, int(total_area_pieces / area_sheet) + 2)
        time_per_sheet = max(0.3, time_limit_s / n_sheets_est)

        remaining = sorted(range(len(pieces)), key=lambda i: -pieces[i]["area"])
        all_sheets = []

        while remaining:
            rem_pieces = [pieces[i] for i in remaining]
            rem_poles  = _precompute_poles(rem_pieces)
            n_loc      = len(rem_pieces)
            weights_loc = {_wkey(i,j): 1.0
                           for i in range(n_loc) for j in range(i+1, n_loc)}

            order_loc = list(range(n_loc))
            sol = _bl_initial(order_loc, rem_pieces, W, L_max, gap_mm)
            if not sol:
                break

            best_sol = [_copy_pp(p) for p in sol]
            best_z   = min(_z_max(sol), L_max)

            # Se orçamento por chapa for muito pequeno, usa só BL (sem otimizar)
            if time_per_sheet >= 0.5:
                t1      = time.perf_counter()
                tl_x    = time_per_sheet * TL_X
                min_h   = min(p["height"] for p in rem_pieces)
                rx_adap = max(RX, min_h / (best_z + 1e-9))
                pool_inf = []

                # EXPLORE
                while time.perf_counter() - t1 < tl_x:
                    new_z = best_z * (1 - rx_adap)
                    sol_s = _shrink(best_sol, W, new_z)
                    sol_sep, z_sep = _separate(
                        sol_s, rem_pieces, rem_poles, W, new_z, gap_mm,
                        rng, weights_loc, KX, NX, t1, t1 + tl_x
                    )
                    if z_sep is not None and z_sep < best_z - 0.5:
                        best_sol = [_copy_pp(p) for p in sol_sep]
                        best_z   = min(_z_max(best_sol), L_max)
                        rx_adap  = max(RX, min_h / (best_z + 1e-9))
                    else:
                        pool_inf.append((z_sep or best_z,
                                         [_copy_pp(p) for p in sol_sep]))
                        if pool_inf:
                            pool_inf.sort(key=lambda x: x[0])
                            sol = _swap_two_large(
                                [_copy_pp(p) for p in pool_inf[0][1]], rng)

                # COMPRESS
                t_c0  = time.perf_counter() - t1
                tl_c  = time_per_sheet * TL_C
                t_end = t1 + time_per_sheet
                while time.perf_counter() < t_end:
                    tau = (time.perf_counter() - t1) - t_c0
                    r   = RS_C + (RE_C - RS_C) * min(1.0, tau / (tl_c + 1e-9))
                    shrink_mm = max(best_z * r, 1.0)
                    new_z = best_z - shrink_mm
                    sol_s = _shrink(best_sol, W, new_z)
                    sol_sep, z_sep = _separate(
                        sol_s, rem_pieces, rem_poles, W, new_z, gap_mm,
                        rng, weights_loc, KC, NC, t1, t_end
                    )
                    if z_sep is not None and z_sep < best_z - 0.5:
                        best_sol = [_copy_pp(p) for p in sol_sep]
                        best_z   = min(_z_max(best_sol), L_max)

                if not _is_feasible(best_sol, rem_poles):
                    best_sol = _bl_initial(order_loc, rem_pieces, W, L_max, gap_mm)

            # Só aceita peças que ficaram dentro de L_max
            sheet = [p for p in best_sol
                     if p.poly_placed.bounds[3] <= L_max + 1e-6]
            if not sheet:
                sheet = [best_sol[0]]  # fallback: 1 peça por chapa

            # p.idx é o índice local em rem_pieces (0..n_loc-1)
            # remaining[p.idx] é o índice global correspondente
            placed_local_idx = {p.idx for p in sheet}
            all_sheets.append(sheet)
            # Remove de remaining apenas os índices locais que foram alocados
            remaining = [remaining[i] for i in range(len(remaining))
                         if i not in placed_local_idx]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return compute_metrics(all_sheets if all_sheets else [[]],
                               pieces, W, L_max, elapsed_ms, "sparrow")

    # ── MODO STRIP LIVRE (sem L_max) ─────────────────────────────────
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["area"])
    sol   = _bl_initial(order, pieces, W, None, gap_mm)
    if not sol:
        return compute_metrics([], pieces, W, L_max,
                               (time.perf_counter()-t0)*1000, "sparrow")

    best_sol = [_copy_pp(p) for p in sol]
    best_z   = _z_max(sol)

    n = len(pieces)
    weights = {_wkey(i,j): 1.0 for i in range(n) for j in range(i+1, n)}
    tl_x = time_limit_s * TL_X
    min_piece_h = min(p["height"] for p in pieces)
    rx_adaptive = max(RX, min_piece_h / (best_z + 1e-9))
    pool_inf = []

    while time.perf_counter() - t0 < tl_x:
        new_z = best_z * (1 - rx_adaptive)
        sol_s = _shrink(best_sol, W, new_z)
        sol_sep, z_sep = _separate(
            sol_s, pieces, piece_poles, W, new_z, gap_mm,
            rng, weights, KX, NX, t0, t0 + tl_x
        )
        if z_sep is not None and z_sep < best_z - 0.5:
            best_sol = [_copy_pp(p) for p in sol_sep]
            best_z   = _z_max(best_sol)
            rx_adaptive = max(RX, min_piece_h / (best_z + 1e-9))
        else:
            pool_inf.append((z_sep or best_z, [_copy_pp(p) for p in sol_sep]))
            if pool_inf:
                pool_inf.sort(key=lambda x: x[0])
                s_hat = [_copy_pp(p) for p in pool_inf[0][1]]
                sol   = _swap_two_large(s_hat, rng)

    t_c0 = time.perf_counter() - t0
    tl_c = time_limit_s * TL_C
    while time.perf_counter() - t0 < time_limit_s:
        tau  = (time.perf_counter() - t0) - t_c0
        r    = RS_C + (RE_C - RS_C) * min(1.0, tau / (tl_c + 1e-9))
        shrink_mm = max(best_z * r, 1.0)
        new_z = best_z - shrink_mm
        sol_s = _shrink(best_sol, W, new_z)
        sol_sep, z_sep = _separate(
            sol_s, pieces, piece_poles, W, new_z, gap_mm,
            rng, weights, KC, NC, t0, t0 + time_limit_s
        )
        if z_sep is not None and z_sep < best_z - 0.5:
            best_sol = [_copy_pp(p) for p in sol_sep]
            best_z   = _z_max(best_sol)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not _is_feasible(best_sol, piece_poles):
        order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["area"])
        best_sol = _bl_initial(order, pieces, W, None, gap_mm)
    return compute_metrics([best_sol], pieces, W, L_max, elapsed_ms, "sparrow")


# ── SEPARATE (Alg. 9) ────────────────────────────────────────────────

def _separate(sol, pieces, piece_poles, W, L_max, gap_mm,
              rng, weights, kmax, nmax, t0, tl):
    s_best = [_copy_pp(p) for p in sol]
    z_best = _severity_total(s_best, piece_poles, weights, L_max)
    k = 0

    while k < kmax and time.perf_counter() - t0 < tl:
        sol_c = [_copy_pp(p) for p in s_best]
        n_iter = 0
        improved = False

        while n_iter < nmax and time.perf_counter() - t0 < tl:
            # move_items_multi: executa move_items e mantém o melhor  (Alg. 10)
            _move_items(sol_c, pieces, piece_poles, W, L_max,
                        gap_mm, rng, weights)
            _update_weights(sol_c, piece_poles, weights)
            z = _severity_total(sol_c, piece_poles, weights, L_max)
            n_iter += 1

            if z < z_best:
                s_best   = [_copy_pp(p) for p in sol_c]
                z_best   = z
                n_iter   = 0
                improved = True
                if z_best <= 1e-6:
                    return s_best, _z_max(s_best)

        k += 1
        if improved:
            k = 0

    z_f = _z_max(s_best) if _is_feasible(s_best, piece_poles) else None
    return s_best, z_f


# ── MOVE ITEMS (Alg. 5) ─────────────────────────────────────────────

def _move_items(sol, pieces, piece_poles, W, L_max, gap_mm, rng, weights):
    """Repositiona itens colisores em ordem aleatória (Alg. 5)."""
    colliding = [p for p in sol if _has_collision(p, sol, piece_poles)]
    rng.shuffle(colliding)
    for item in colliding:
        new_pp = _search_position(item, sol, pieces, piece_poles,
                                  W, L_max, gap_mm, rng, weights)
        # Atualiza em-place
        item.x           = new_pp.x
        item.y           = new_pp.y
        item.rot         = new_pp.rot
        item.poly_local  = new_pp.poly_local
        item.poly_placed = new_pp.poly_placed
        item.bbox_w      = new_pp.bbox_w
        item.bbox_h      = new_pp.bbox_h
        item._poly_buf   = None


# ── SEARCH POSITION (Alg. 6, Fig. 7) ────────────────────────────────

def _search_position(item, sol, pieces, piece_poles, W, L_max,
                     gap_mm, rng, weights) -> PlacedPiece:
    """
    1. Gera Tdiv: amostras uniformes no strip  (Fig. 7b)
    2. Gera Tfoc: amostras na vizinhança da posição atual  (Fig. 7c)
    3. Seleciona N_REF mais promissores e refina por descida de coordenadas
       (Loshchilov et al. 2011)  (Fig. 7d)
    4. Retorna melhor posição encontrada  (Fig. 7e)
    """
    others   = [p for p in sol if p.idx != item.idx]
    p_orig   = pieces[item.idx]
    poles_by_ang = piece_poles.get(item.idx, {})
    # z_bound é o limite do strip atual — peças não podem sair para fora
    z_bound = L_max if L_max else max(_z_max(sol) * 1.2, item.bbox_h * 3)

    candidates = []

    # Tdiv — amostras diversas uniformes no strip
    for _ in range(N_DIV):
        ang  = rng.choice(ANGLES)
        poly = rotate_polygon(p_orig["polygon"], ang)
        b    = poly.bounds; pw, ph = b[2]-b[0], b[3]-b[1]
        if pw > W: continue
        x = rng.uniform(0, max(0, W - pw))
        y = rng.uniform(0, max(0, z_bound - ph))
        poles = _translate_poles(poles_by_ang.get(ang, []), x, y)
        c = PlacedPiece(item.idx, x, y, ang,
                        item.tipo, item.label, item.area, poly)
        candidates.append((c, poles))

    # Tfoc — amostras focais na vizinhança da posição atual (Fig. 7c)
    # Metade mantém rotação atual, metade testa rotações alternativas
    r_foc = max(item.bbox_w, item.bbox_h) * 1.5
    for fi in range(N_FOC):
        ang = item.rot if fi < N_FOC // 2 else rng.choice(ANGLES)
        poly = rotate_polygon(p_orig["polygon"], ang)
        b    = poly.bounds; pw, ph = b[2]-b[0], b[3]-b[1]
        if pw > W: continue
        x = max(0, min(W-pw, item.x + rng.uniform(-r_foc, r_foc)))
        y = max(0, item.y + rng.uniform(-r_foc, r_foc))
        poles = _translate_poles(poles_by_ang.get(ang, []), x, y)
        c = PlacedPiece(item.idx, x, y, ang,
                        item.tipo, item.label, item.area, poly)
        candidates.append((c, poles))

    if not candidates:
        return item

    # Avalia todos os candidatos  (Alg. 7)
    other_poles = _get_other_poles(others, piece_poles)
    scored = sorted(candidates,
                    key=lambda cp: _eval_sample(cp[0], cp[1], others,
                                                other_poles, weights))

    best_c, best_poles = scored[0]
    best_e = _eval_sample(best_c, best_poles, others, other_poles, weights)

    # Refina os N_REF mais promissores  (Seção 7)
    for (c0, poles0) in scored[:N_REF]:
        refined, ref_poles = _refine(
            c0, poles0, others, other_poles, W, z_bound, gap_mm,
            weights, p_orig, item.idx, poles_by_ang
        )
        e_ref = _eval_sample(refined, ref_poles, others, other_poles, weights)
        if e_ref < best_e:
            best_e, best_c, best_poles = e_ref, refined, ref_poles

    return best_c


def _refine(cand, poles, others, other_poles, W, z_bound, gap_mm,
            weights, p_orig, idx, poles_by_ang) -> tuple:
    """
    Refinamento por descida de coordenadas adaptativa
    (Loshchilov et al. 2011, Seção 7).
    Inclui tentativa de rotação alternativa a cada 3 passos.
    """
    x, y, ang = cand.x, cand.y, cand.rot
    poly = cand.poly_local
    b = poly.bounds; pw, ph = b[2]-b[0], b[3]-b[1]
    step_x = pw * 0.4
    step_y = ph * 0.4
    e_curr = _eval_sample(cand, poles, others, other_poles, weights)

    for step_i in range(REFINE_STEPS):
        improved = False
        # Movimentos translacionais
        for dx, dy in [(step_x,0), (-step_x,0), (0,step_y), (0,-step_y)]:
            nx = max(0.0, min(W - pw, x + dx))
            ny = max(0.0, y + dy)
            if z_bound and ny + ph > z_bound: continue
            nc = PlacedPiece(idx, nx, ny, ang,
                             cand.tipo, cand.label, cand.area, poly)
            np_ = _translate_poles(poles_by_ang.get(ang, []), nx, ny)
            ne  = _eval_sample(nc, np_, others, other_poles, weights)
            if ne < e_curr:
                x, y, cand, poles, e_curr = nx, ny, nc, np_, ne
                improved = True
        # Tenta rotação alternativa a cada 3 passos
        if step_i % 3 == 2:
            for new_ang in ANGLES:
                if new_ang == ang: continue
                new_poly = rotate_polygon(p_orig["polygon"], new_ang)
                nb = new_poly.bounds; npw, nph = nb[2]-nb[0], nb[3]-nb[1]
                if npw > W: continue
                nx = max(0.0, min(W - npw, x))
                ny = max(0.0, y)
                if z_bound and ny + nph > z_bound: continue
                nc = PlacedPiece(idx, nx, ny, new_ang,
                                 cand.tipo, cand.label, cand.area, new_poly)
                np_ = _translate_poles(poles_by_ang.get(new_ang, []), nx, ny)
                ne  = _eval_sample(nc, np_, others, other_poles, weights)
                if ne < e_curr:
                    ang, poly, pw, ph = new_ang, new_poly, npw, nph
                    x, y, cand, poles, e_curr = nx, ny, nc, np_, ne
                    step_x = pw * 0.4; step_y = ph * 0.4
                    improved = True
                    break
        if not improved:
            step_x *= 0.5; step_y *= 0.5
        if step_x < 0.5 and step_y < 0.5:
            break

    return cand, poles


# ── UPDATE WEIGHTS (Alg. 8) ─────────────────────────────────────────

def _update_weights(sol, piece_poles, weights):
    """
    Atualiza pesos GLS (Alg. 8):
      Se par colide: m = Ml + (Mu-Ml)*(e/e_max)  ∈ [Ml, Mu]
      Se não colide: m = Md  (decai de volta a 1)
    """
    n = len(sol)
    other_poles = _get_other_poles(sol, piece_poles)
    sev = {}; e_max = 1e-9

    for i in range(n):
        for j in range(i+1, n):
            poles_i = _translate_poles(
                piece_poles.get(sol[i].idx, {}).get(sol[i].rot, []),
                sol[i].x, sol[i].y
            )
            poles_j = _translate_poles(
                piece_poles.get(sol[j].idx, {}).get(sol[j].rot, []),
                sol[j].x, sol[j].y
            )
            e = _quantify_pair_poles(sol[i], poles_i, sol[j], poles_j)
            sev[(i,j)] = e
            if e > e_max: e_max = e

    for i in range(n):
        for j in range(i+1, n):
            key = _wkey(sol[i].idx, sol[j].idx)
            e   = sev[(i,j)]
            m   = ML + (MU - ML) * (e / e_max) if e > 1e-9 else MD
            weights[key] = max(1.0, weights.get(key, 1.0) * m)


# ── QUANTIFICAÇÃO DE COLISÃO (Alg. 3 e 4) ───────────────────────────

def _overlap_proxy_decay(poles_a: list, poles_b: list,
                         diam_max: float) -> float:
    """
    Alg. 3 — overlap_proxy_decay:
      Para cada par de poles (pa, pb):
        δ = r_a + r_b - dist(centro_a, centro_b)   (penetration depth)
        δ' = δ           se δ > ε
             ε²/(-δ+2ε)  caso contrário  (decaimento hiperbólico, Eq. 5)
        contribuição += δ' × min(diâm_pa, diâm_pb)
    """
    alpha = 0.0

    for (cxa, cya, ra) in poles_a:
        for (cxb, cyb, rb) in poles_b:
            dist  = math.sqrt((cxa-cxb)**2 + (cya-cyb)**2)
            delta = ra + rb - dist        # penetration depth

            # eps escalonado pelo diâmetro dos poles (não da chapa)
            eps = R_EPS * max(2*ra, 2*rb)

            if delta > eps:
                dp = delta
            elif delta > -4 * max(2*ra, 2*rb):
                # Decaimento hiperbólico dentro de 4 diâmetros (Eq. 5)
                dp = eps**2 / (-delta + 2*eps + 1e-12)
            else:
                # Muito longe: sem contribuição (evita ruído de longa distância)
                continue

            alpha += dp * min(2*ra, 2*rb)

    return alpha


def _quantify_pair_poles(a: PlacedPiece, poles_a: list,
                          b: PlacedPiece, poles_b: list) -> float:
    """
    Alg. 4 — quantify_collision:
      α  = overlap_proxy_decay(Sa, Sb)
      λab = √(λa × λb),  λ = √(área convex hull)   (Eq. 6-7)
      retorna √α × λab
    """
    if not poles_a or not poles_b:
        return 0.0

    diam_a = math.sqrt(a.bbox_w**2 + a.bbox_h**2)
    diam_b = math.sqrt(b.bbox_w**2 + b.bbox_h**2)
    diam_max = max(diam_a, diam_b)

    alpha = _overlap_proxy_decay(poles_a, poles_b, diam_max)

    if alpha <= 0:
        return 0.0

    # Penalidade λ = √(área do convex hull)  (Eq. 6)
    lam_a  = math.sqrt(a.poly_local.convex_hull.area + 1e-9)
    lam_b  = math.sqrt(b.poly_local.convex_hull.area + 1e-9)
    lam_ab = math.sqrt(lam_a * lam_b)                         # Eq. 7

    return math.sqrt(max(0.0, alpha)) * lam_ab                # Alg. 4


def _eval_sample(item: PlacedPiece, poles_item: list,
                 others: list, other_poles: dict,
                 weights: dict) -> float:
    """
    Alg. 7 — evaluate_sample:
      e = Σ_c w_ic × quantify_collision(t(Si), Sc)
    """
    e = 0.0
    for o in others:
        key  = _wkey(item.idx, o.idx)
        w    = weights.get(key, 1.0)
        p_o  = other_poles.get(id(o), [])
        sev  = _quantify_pair_poles(item, poles_item, o, p_o)
        e   += w * sev
    return e


def _severity_total(sol, piece_poles, weights, L_max=None) -> float:
    """
    Eq. 8 — severidade total da solução.
    Inclui: (1) interseção real entre pares de peças (Shapely)
            (2) penalidade por peças que excedem o limite L_max do strip
    """
    total = 0.0; n = len(sol)
    for i in range(n):
        # Penalidade por exceder o limite do strip — força peças para dentro
        if L_max is not None:
            excess = sol[i].poly_placed.bounds[3] - L_max
            if excess > 1e-6:
                total += excess * sol[i].bbox_w
        for j in range(i+1, n):
            bi = sol[i].poly_placed.bounds
            bj = sol[j].poly_placed.bounds
            if bi[2] < bj[0] or bi[0] > bj[2] or bi[3] < bj[1] or bi[1] > bj[3]:
                continue
            inter = sol[i].poly_placed.intersection(sol[j].poly_placed)
            if inter.is_empty or inter.area < 1e-6:
                continue
            poles_i = _translate_poles(
                piece_poles.get(sol[i].idx, {}).get(sol[i].rot, []),
                sol[i].x, sol[i].y
            )
            poles_j = _translate_poles(
                piece_poles.get(sol[j].idx, {}).get(sol[j].rot, []),
                sol[j].x, sol[j].y
            )
            sev = _quantify_pair_poles(sol[i], poles_i, sol[j], poles_j)
            total += max(sev, inter.area)
    return total


# ── UTILITÁRIOS ──────────────────────────────────────────────────────

def _precompute_poles(pieces: list) -> dict:
    """
    Pré-computa poles para cada (idx, ângulo).
    Retorna dict: idx → {ang → [(cx, cy, r), ...]} em coord locais.
    """
    result = {}
    for i, p in enumerate(pieces):
        result[i] = {}
        for ang in ANGLES:
            poly  = rotate_polygon(p["polygon"], ang)
            poles = get_poles(poly, n_poles=N_POLES)
            result[i][ang] = poles
    return result


def _translate_poles(poles: list, dx: float, dy: float) -> list:
    """Translada poles de coord locais para coord do strip."""
    return [(cx + dx, cy + dy, r) for (cx, cy, r) in poles]


def _get_other_poles(others: list, piece_poles: dict) -> dict:
    """Calcula poles no strip para cada peça 'other' (por id Python)."""
    result = {}
    for o in others:
        poles_local = piece_poles.get(o.idx, {}).get(o.rot, [])
        result[id(o)] = _translate_poles(poles_local, o.x, o.y)
    return result


def _has_collision(item: PlacedPiece, sol: list, piece_poles: dict) -> bool:
    """Detecta colisão real via Shapely (sem falsos positivos de poles)."""
    bi = item.poly_placed.bounds
    for o in sol:
        if o.idx == item.idx: continue
        bo = o.poly_placed.bounds
        if bi[2] < bo[0] or bi[0] > bo[2] or bi[3] < bo[1] or bi[1] > bo[3]:
            continue
        if item.poly_placed.intersects(o.poly_placed):
            inter = item.poly_placed.intersection(o.poly_placed)
            if not inter.is_empty and inter.area > 1e-6:
                return True
    return False


def _is_feasible(sol: list, piece_poles: dict) -> bool:
    """Verifica factibilidade usando Shapely (exacto) em vez de poles."""
    n = len(sol)
    for i in range(n):
        for j in range(i+1, n):
            # AABB rápido
            bi = sol[i].poly_placed.bounds
            bj = sol[j].poly_placed.bounds
            if bi[2] < bj[0] or bi[0] > bj[2] or bi[3] < bj[1] or bi[1] > bj[3]:
                continue
            if sol[i].poly_placed.intersects(sol[j].poly_placed):
                return False
    return True


def _bl_initial(order, pieces, W, L_max, gap_mm):
    """
    Solução inicial BL: testa todas as rotações ortogonais e escolhe
    a que produz menor (y+h, x) — mesmo critério do bl_nfp._try_place.
    Isso garante que o sparrow parta de uma solução compacta,
    equivalente ao BL construtivo sem 2-exchange.
    """
    import math as _math
    placed = []
    for idx in order:
        p = pieces[idx]
        best_pp  = None
        best_key = (_math.inf, _math.inf)
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
                                       p["tipo"], p["label"],
                                       p["area"], poly)
        if best_pp is not None:
            placed.append(best_pp)
        # Se nenhuma rotação cabe dentro de L_max, simplesmente pula a peça
        # (será alocada na próxima chapa)
    return placed


def _shrink(sol, W, new_z):
    """
    Encolhe o strip para new_z (Alg. 12):
    Peças que excedem new_z são empurradas para new_z - altura,
    criando colisões reais dentro do strip que o GLS deve resolver.
    Peças dentro do limite são mantidas na posição atual.
    """
    result = []
    for p in sol:
        top = p.poly_placed.bounds[3]
        if top <= new_z + 1e-6:
            result.append(_copy_pp(p))
        else:
            # Empurra para dentro do strip (topo alinhado com new_z)
            new_y = max(0.0, new_z - p.bbox_h)
            pp = PlacedPiece(p.idx, p.x, new_y, p.rot,
                             p.tipo, p.label, p.area, p.poly_local)
            result.append(pp)
    return result


def _swap_two_large(sol, rng):
    if len(sol) < 2: return sol
    by_area = sorted(range(len(sol)), key=lambda i: -sol[i].area)
    top = by_area[:min(4, len(by_area))]
    i, j = rng.sample(top, min(2, len(top)))
    result = [_copy_pp(p) for p in sol]
    result[i].x, result[j].x = result[j].x, result[i].x
    result[i].y, result[j].y = result[j].y, result[i].y
    for pp in result:
        pp.poly_placed = translate(pp.poly_local, pp.x, pp.y)
        pp._poly_buf   = None
    return result


def _z_max(sol) -> float:
    if not sol: return 0.0
    return max(p.poly_placed.bounds[3] for p in sol)


def _copy_pp(p: PlacedPiece) -> PlacedPiece:
    pp = PlacedPiece(p.idx, p.x, p.y, p.rot,
                     p.tipo, p.label, p.area, p.poly_local)
    return pp


def _wkey(a, b): return (min(a,b), max(a,b))

def _split_into_sheets(sol: list, L_max: float) -> list:
    """
    Divide o layout de strip livre em múltiplas chapas de comprimento L_max.
    Peças que cabem dentro de L_max ficam na chapa 1 (sem re-posicionamento).
    Peças que não cabem vão para chapas seguintes — mantidas em posição y original
    relativa ao início de cada chapa.
    """
    if not sol:
        return []
    # O layout livre tem todas as peças como se fossem num strip único.
    # Para chapa fixa, simplesmente verificamos quais peças estão dentro de [0, L_max].
    # Como o sparrow trabalha internamente sem L_max, a solução pode ter z > L_max.
    # A divisão em chapas é contabilizada para fins de métricas.
    sheet1 = [p for p in sol if p.poly_placed.bounds[3] <= L_max + 1e-6]
    overflow = [p for p in sol if p.poly_placed.bounds[3] > L_max + 1e-6]
    sheets = []
    if sheet1:
        sheets.append(sheet1)
    # Peças de overflow: re-posiciona em y=0 em chapas adicionais
    if overflow:
        current = []; current_z = 0.0
        for pp in sorted(overflow, key=lambda p: p.bbox_h, reverse=True):
            ph = pp.bbox_h
            if current_z + ph + 2.0 <= L_max + 1e-6:
                new_pp = PlacedPiece(pp.idx, pp.x, current_z, pp.rot,
                                     pp.tipo, pp.label, pp.area, pp.poly_local)
                current.append(new_pp)
                current_z += ph + 2.0
            else:
                if current: sheets.append(current)
                new_pp = PlacedPiece(pp.idx, pp.x, 0.0, pp.rot,
                                     pp.tipo, pp.label, pp.area, pp.poly_local)
                current = [new_pp]
                current_z = ph + 2.0
        if current:
            sheets.append(current)
    return sheets if sheets else [sol]
