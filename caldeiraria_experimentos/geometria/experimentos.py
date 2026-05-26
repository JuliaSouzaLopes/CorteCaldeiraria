"""
Definição dos experimentos E1–E6.
Série ISO 4200 / ASME B36.10 com restrição: poucas peças grandes OU muitas peças pequenas.

Cada experimento é um dict com:
  id, label, desc, pieces: list de {tipo, qty, params}
"""

from geometria.planificacao import (
    corte_inclinado, reducao_concentrica, boca_de_lobo, curva_de_gomos,
    poly_area, bbox
)

# ──────────────────────────────────────────────────────────────────
# Diâmetros externos ISO 4200 / ASME B36.10
# ──────────────────────────────────────────────────────────────────
# DN25=33.7, DN50=60.3, DN80=88.9, DN100=114.3,
# DN150=168.3, DN200=219.1, DN300=323.9  (mm)

EXPERIMENTOS_RAW = [
    {
        "id": "E1",
        "label": "E1 — Poucas peças · pouca variedade",
        "desc": "6 peças · 2 tipos · DN50/DN80 · peças pequenas",
        "pieces": [
            {"tipo": "corte_inclinado", "qty": 4, "params": {"D": 60.3,  "Alfa": 45.0, "n": 24}},
            {"tipo": "reducao",         "qty": 2, "params": {"D_maior": 88.9, "D_menor": 60.3, "H_red": 80.0, "pts": 64}},
        ]
    },
    {
        "id": "E2",
        "label": "E2 — Poucas peças · alta variedade",
        "desc": "11 peças · 4 tipos · DN50/DN80/DN100",
        "pieces": [
            {"tipo": "boca_de_lobo",    "qty": 2, "params": {"DG": 114.3, "dp": 88.9,  "Alfa": 90.0, "n": 36, "peca": "menor"}},
            {"tipo": "reducao",         "qty": 2, "params": {"D_maior": 88.9,  "D_menor": 60.3,  "H_red": 80.0,  "pts": 64}},
            {"tipo": "curva_gomos",     "qty": 5, "params": {"DT": 60.3,  "RC": 90.5,  "Alfa": 90.0, "NG": 5, "n": 36, "gomo": "inter"}},
            {"tipo": "corte_inclinado", "qty": 2, "params": {"D": 88.9,  "Alfa": 45.0, "n": 24}},
        ]
    },
    {
        "id": "E3",
        "label": "E3 — Qtd. intermediária · pouca variedade",
        "desc": "18 peças · 2 tipos · DN150 · repetição",
        "pieces": [
            {"tipo": "corte_inclinado", "qty": 12, "params": {"D": 168.3, "Alfa": 45.0, "n": 24}},
            {"tipo": "reducao",     "qty": 6,  "params": {"D_maior": 168.3,  "D_menor": 114.3,  "H_red": 150.0,  "pts": 64}},
        ]
    },
    {
        "id": "E4",
        "label": "E4 — Qtd. intermediária · alta variedade",
        "desc": "20 peças · 4 tipos · DN100/DN150/DN200 · lote misto",
        "pieces": [
            {"tipo": "corte_inclinado", "qty": 7, "params": {"D": 114.3,  "Alfa": 45.0, "n": 24}},
            {"tipo": "boca_de_lobo",    "qty": 5, "params": {"DG": 219.1, "dp": 168.3, "Alfa": 90.0, "n": 36, "peca": "menor"}},
            {"tipo": "reducao",         "qty": 3, "params": {"D_maior": 168.3, "D_menor": 114.3, "H_red": 150.0, "pts": 64}},
            {"tipo": "curva_gomos",     "qty": 5, "params": {"DT": 168.3, "RC": 252.5, "Alfa": 90.0, "NG": 5, "n": 36, "gomo": "inter"}},
        ]
    },
    {
        "id": "E5",
        "label": "E5 — Muitas peças · pouca variedade",
        "desc": "40 peças · 2 tipos · DN300 · peças grandes",
        "pieces": [
            {"tipo": "reducao",     "qty": 22, "params": {"D_maior": 323.9, "D_menor": 219.1, "H_red": 280.0, "pts": 64}},
            {"tipo": "curva_gomos", "qty": 18, "params": {"DT": 323.9, "RC": 485.9, "Alfa": 90.0, "NG": 6, "n": 36, "gomo": "inter"}},
        ]
    },
    {
        "id": "E6",
        "label": "E6 — Muitas peças · alta variedade",
        "desc": "50 peças · 4 tipos · DN200/DN300 · máxima complexidade",
        "pieces": [
            {"tipo": "corte_inclinado", "qty": 14, "params": {"D": 219.1,  "Alfa": 45.0, "n": 24}},
            {"tipo": "reducao",         "qty": 12, "params": {"D_maior": 323.9, "D_menor": 219.1, "H_red": 280.0, "pts": 64}},
            {"tipo": "boca_de_lobo",    "qty": 12, "params": {"DG": 323.9, "dp": 219.1, "Alfa": 90.0, "n": 36, "peca": "menor"}},
            {"tipo": "curva_gomos",     "qty": 12, "params": {"DT": 323.9, "RC": 485.9, "Alfa": 90.0, "NG": 6, "n": 36, "gomo": "inter"}},
        ]
    },
]


_GENERATORS = {
    "corte_inclinado": corte_inclinado,
    "reducao":         reducao_concentrica,
    "boca_de_lobo":    boca_de_lobo,
    "curva_gomos":     curva_de_gomos,
}

_LABELS = {
    "corte_inclinado": "Corte Inclinado",
    "reducao":         "Redução Concêntrica",
    "boca_de_lobo":    "Boca de Lobo",
    "curva_gomos":     "Curva de Gomos",
}


def build_piece_list(exp_raw: dict) -> list[dict]:
    """
    Retorna lista de dicts, cada um representando UMA peça:
      {tipo, label, polygon, width, height, area}
    """
    pieces = []
    for item in exp_raw["pieces"]:
        gen = _GENERATORS[item["tipo"]]
        poly = gen(**item["params"])
        x0, y0, x1, y1 = bbox(poly)
        w, h = x1 - x0, y1 - y0
        a = poly_area(poly)
        for _ in range(item["qty"]):
            pieces.append({
                "tipo":    item["tipo"],
                "label":   _LABELS[item["tipo"]],
                "polygon": poly,
                "width":   w,
                "height":  h,
                "area":    a,
            })
    return pieces


def get_experiments() -> list[dict]:
    """Retorna lista de experimentos com piece_list já resolvida."""
    result = []
    for raw in EXPERIMENTOS_RAW:
        pieces = build_piece_list(raw)
        result.append({
            "id":         raw["id"],
            "label":      raw["label"],
            "desc":       raw["desc"],
            "pieces":     pieces,
            "n_pecas":    len(pieces),
            "n_tipos":    len(raw["pieces"]),
            "area_total": sum(p["area"] for p in pieces),
        })
    return result
