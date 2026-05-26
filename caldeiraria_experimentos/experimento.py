"""
experimento.py — Orquestrador v12
===================================
Uso:
  python experimento.py                      # todos os métodos, todos os experimentos
  python experimento.py --metodo BL+NFP      # só BL+NFP
  python experimento.py --metodo sparrow     # só sparrow
  python experimento.py --metodo BLF-SD --exp E1 E2   # BLF-SD nos experimentos E1 e E2
  python experimento.py --metodo GSAO --chapa 1000x2000mm  # uma chapa específica

Métodos disponíveis: BL+NFP, mu-BRKGA, BS+TS, GSAO, BLF-SD, sparrow
Chapas disponíveis:  1000x2000mm, 1500x3000mm, 1000xliv, 1500xliv
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

from geometria.experimentos import get_experiments
from algoritmos import bl_nfp, mu_brkga, bsts, gsao, blf_sd, sparrow
from visualizacao import (
    plot_layout, plot_metricas_experimento,
    plot_comparativo_geral, save_csv
)

# ── Configuração das chapas ──────────────────────────────────────────
ALL_CHAPAS = [
    {"W": 1000., "L": 2000., "label": "1000x2000mm", "fixed": True},
    {"W": 1500., "L": 3000., "label": "1500x3000mm", "fixed": True},
    {"W": 1000., "L": None,  "label": "1000xliv",    "fixed": False},
    {"W": 1500., "L": None,  "label": "1500xliv",    "fixed": False},
]

# ── Configuração dos métodos ─────────────────────────────────────────
ALL_ALGS = {
    "BL+NFP":   (bl_nfp.run,   {"time_limit_s": 8}),
    "mu-BRKGA": (mu_brkga.run, {"time_limit_s": 5}),
    "BS+TS":    (bsts.run,     {"time_limit_s": 5}),
    "GSAO":     (gsao.run,     {"time_limit_s": 5}),
    "BLF-SD":   (blf_sd.run,   {}),
    "sparrow":  (sparrow.run,  {"time_limit_s": 3}),
}

# E4-E6 usam subconjunto de métodos por padrão (mais lentos)
FAST_ALGS = {"BL+NFP", "BLF-SD", "sparrow"}

GAP_MM = 2.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "resultados")


def main():
    parser = argparse.ArgumentParser(
        description="Experimentos de corte irregular para caldeiraria"
    )
    parser.add_argument(
        "--metodo", "-m",
        help="Método a executar. Se omitido, executa todos.",
        choices=list(ALL_ALGS.keys()),
        default=None,
    )
    parser.add_argument(
        "--exp", "-e",
        nargs="+",
        help="IDs dos experimentos (ex: E1 E2). Se omitido, executa todos.",
        default=None,
    )
    parser.add_argument(
        "--chapa", "-c",
        nargs="+",
        help="Labels das chapas (ex: 1000x2000mm 1000xliv). Se omitido, todas.",
        choices=[ch["label"] for ch in ALL_CHAPAS],
        default=None,
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Força re-execução mesmo que já exista no CSV.",
    )
    args = parser.parse_args()

    # ── Filtra experimentos, chapas e métodos ────────────────────────
    all_exps = get_experiments()
    if args.exp:
        all_exps = [e for e in all_exps if e["id"] in args.exp]
        if not all_exps:
            print(f"Nenhum experimento encontrado para: {args.exp}")
            return

    chapas = ALL_CHAPAS
    if args.chapa:
        chapas = [ch for ch in ALL_CHAPAS if ch["label"] in args.chapa]

    if args.metodo:
        algs_to_run = {args.metodo: ALL_ALGS[args.metodo]}
    else:
        algs_to_run = ALL_ALGS

    # ── Carrega CSV existente para checkpoint ────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "metricas_consolidado.csv")
    all_rows, done_keys = _load_existing(csv_path)
    all_summ = _rows_to_summ(all_rows)

    total = len(all_exps) * len(chapas) * len(algs_to_run)
    done = 0

    print("=" * 68)
    print(f"  Métodos : {', '.join(algs_to_run.keys())}")
    print(f"  Experim.: {', '.join(e['id'] for e in all_exps)}")
    print(f"  Chapas  : {', '.join(ch['label'] for ch in chapas)}")
    print(f"  Total   : {total} runs  (--force={args.force})")
    print("=" * 68)

    t0 = time.perf_counter()

    for exp in all_exps:
        exp_dir = os.path.join(OUT_DIR, exp["id"])
        os.makedirs(exp_dir, exist_ok=True)
        exp_changed = False

        for ch in chapas:
            W, L, sl, fixed = ch["W"], ch["L"], ch["label"], ch["fixed"]

            # Determina quais métodos rodar nesta chapa
            if args.metodo is None and exp["id"] in ("E4", "E5", "E6"):
                run_algs = {k: v for k, v in algs_to_run.items() if k in FAST_ALGS}
            else:
                run_algs = algs_to_run

            for mn, (mf, kw) in run_algs.items():
                done += 1
                key = (exp["id"], sl, mn)

                if key in done_keys and not args.force:
                    print(f"  SKIP {exp['id']} {sl} {mn}")
                    continue

                print(f"  [{done:>3}/{total}] {exp['id']} · {sl} · {mn} ...",
                      end="", flush=True)
                try:
                    r = mf(exp["pieces"], W, L, gap_mm=GAP_MM, **kw)
                except Exception as e:
                    print(f"  ERRO: {e}")
                    continue

                elapsed = r["tempo_ms"]
                print(f" z={r['z']:.0f} util={r['utilizacao']:.1f}%"
                      f" t={elapsed:.0f}ms")

                # Salva layout PNG
                sm = mn.replace("+", "_").replace("µ", "u").replace("-", "_")\
                       .replace(".", "").replace(" ", "_")
                layout_path = os.path.join(exp_dir, f"layout_{sm}_{sl}.png")
                try:
                    plot_layout(r, W, exp["id"], sl, layout_path, L_max=L)
                except Exception as e:
                    print(f"    [layout err: {e}]")

                # Acumula métricas
                excl = ("layout", "sheets_layouts", "z_por_chapa",
                        "n_restricoes_detalhe", "comprimento_usado")
                row = {
                    "exp_id": exp["id"], "exp_desc": exp["desc"],
                    "n_pecas": exp["n_pecas"], "n_tipos": exp["n_tipos"],
                    "sheet_label": sl, "sheet_W": W,
                    "sheet_L": L if L else "livre", "fixed": fixed,
                    **{k: v for k, v in r.items() if k not in excl}
                }
                # Remove linha antiga se force
                if args.force:
                    all_rows = [rr for rr in all_rows
                                if not (rr["exp_id"] == exp["id"]
                                        and rr["sheet_label"] == sl
                                        and rr["metodo"] == mn)]
                all_rows.append(row)
                all_summ.append(_row_to_summ(row))
                done_keys.add(key)
                exp_changed = True

            # Tabela de métricas por chapa (agrega todos os métodos presentes)
            all_for_sheet = [rr for rr in all_rows
                             if rr["exp_id"] == exp["id"]
                             and rr["sheet_label"] == sl]
            if all_for_sheet:
                tab_path = os.path.join(exp_dir, f"metricas_{sl}.png")
                try:
                    rp = _rows_to_plot_dicts(all_for_sheet)
                    plot_metricas_experimento(
                        rp, exp["id"], exp["desc"], sl, tab_path,
                        fixed_length=fixed
                    )
                except Exception as e:
                    print(f"    [tabela err: {e}]")

        if exp_changed:
            save_csv(all_rows, csv_path)
            print(f"  ✓ {exp['id']} — CSV salvo ({len(all_rows)} linhas,"
                  f" {time.perf_counter() - t0:.0f}s)")

    # Comparativo geral
    try:
        plot_comparativo_geral(all_summ, OUT_DIR)
    except Exception as e:
        print(f"  [comparativo err: {e}]")

    print("=" * 68)
    print(f"  Concluído em {time.perf_counter() - t0:.1f}s"
          f" · {len(all_rows)} linhas no CSV")
    print("=" * 68)


# ── Helpers ──────────────────────────────────────────────────────────

def _load_existing(csv_path):
    if not os.path.exists(csv_path):
        return [], set()
    import pandas as pd
    rows = pd.read_csv(csv_path).to_dict("records")
    keys = set((r["exp_id"], r["sheet_label"], r["metodo"]) for r in rows)
    return rows, keys


def _rows_to_summ(rows):
    keys = ["exp_id", "sheet_label", "metodo", "utilizacao", "refugo",
            "tempo_ms", "gap_pct", "z", "z_lb", "n_colocadas",
            "n_total", "n_chapas"]
    return [{k: r.get(k) for k in keys} for r in rows]


def _row_to_summ(r):
    return {k: r.get(k) for k in
            ["exp_id", "sheet_label", "metodo", "utilizacao", "refugo",
             "tempo_ms", "gap_pct", "z", "z_lb", "n_colocadas",
             "n_total", "n_chapas"]}


def _rows_to_plot_dicts(rows):
    return [{
        "metodo":            r["metodo"],
        "n_colocadas":       r["n_colocadas"],
        "n_total":           r["n_total"],
        "z":                 r["z"],
        "z_lb":              r.get("z_lb", r["z"]),
        "gap_pct":           r.get("gap_pct"),
        "comprimento_usado": r["z"],
        "area_chapa":        r.get("area_chapa", 0),
        "area_pecas":        r.get("area_pecas", 0),
        "utilizacao":        r["utilizacao"],
        "refugo":            r["refugo"],
        "n_chapas":          r.get("n_chapas", 1),
        "media_pecas_chapa": r.get("media_pecas_chapa", r["n_colocadas"]),
        "tempo_ms":          r["tempo_ms"],
        "n_restricoes":      r.get("n_restricoes"),
        "n_vars_binarias":   r.get("n_vars_binarias"),
    } for r in rows]


if __name__ == "__main__":
    main()
