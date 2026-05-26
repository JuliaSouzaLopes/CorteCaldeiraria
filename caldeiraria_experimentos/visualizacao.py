"""
Módulo de visualização — multi-chapa, tempo em ms, fatiamento de layouts altos.
"""
import os
import math as _math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd

CORES = {
    "corte_inclinado": "#4E9AF1",
    "reducao":         "#F4A259",
    "boca_de_lobo":    "#6FCF97",
    "curva_gomos":     "#BB6BD9",
}
CORE_DEFAULT = "#AAAAAA"

METODO_CORES = {
    "BL+NFP":   "#4E9AF1",   # azul
    "µ-BRKGA":  "#F4A259",   # laranja
    "BS+TS":    "#BB6BD9",   # roxo
    "GSAO":     "#E74C3C",   # vermelho
    "BLF-SD":   "#27AE60",   # verde escuro
    "sparrow":  "#1ABC9C",   # turquesa
}

# Altura máxima de um painel antes de fatiar (mm)
_MAX_PANEL_MM = 3000.0


def plot_layout(result: dict, W: float, exp_id: str,
                sheet_label: str, out_path: str,
                L_max: float = None) -> None:
    """
    Gera PNG do layout.
    - N chapas → N colunas.
    - Comprimento livre alto: cada chapa é fatiada em painéis de
      _MAX_PANEL_MM mm, empilhados verticalmente.  O título informa
      que z é a soma dos painéis.
    """
    sheets = result.get("sheets_layouts") or []
    if not sheets and result.get("layout"):
        sheets = [result["layout"]]
    if not sheets:
        return

    metodo = result["metodo"]
    util   = result["utilizacao"]
    n_sh   = len(sheets)

    z_por_chapa = [
        max(p["y"] + p["height"] for p in s) if s else 0.0
        for s in sheets
    ]

    # Altura do painel: para chapa fixa = L_max; para livre = min(z, _MAX_PANEL_MM)
    if L_max:
        panel_h = float(L_max)
        sliced  = False
    else:
        panel_h = _MAX_PANEL_MM
        sliced  = any(z > panel_h for z in z_por_chapa)

    n_panels    = [max(1, _math.ceil(z / panel_h)) for z in z_por_chapa]
    total_cols  = sum(n_panels)
    col_w = 4.0
    row_h = max(2.0, min(8.0, col_w * panel_h / W))

    fig, axes_flat = plt.subplots(1, total_cols,
                                  figsize=(col_w * total_cols, row_h),
                                  squeeze=False)
    axes_row = axes_flat[0]
    fig.patch.set_facecolor("#FFFFFF")
    seen_tipos = {}

    col_offset = 0
    for si, (sheet, z) in enumerate(zip(sheets, z_por_chapa)):
        np_this = n_panels[si]
        for pi in range(np_this):
            ax = axes_row[col_offset + pi]
            ax.set_facecolor("#F8F8F8")
            ax.tick_params(labelsize=5)

            y_lo = pi * panel_h
            y_hi = y_lo + panel_h

            ax.set_xlim(0, W)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal")

            # Fundo da chapa
            rect_h = min(panel_h, max(0.0, z - y_lo))
            ax.add_patch(mpatches.Rectangle((0, y_lo), W, rect_h, lw=1.5,
                         edgecolor="#888", facecolor="#EEEEEE", zorder=0))

            # Linha de fim-de-uso (comprimento fixo com sobra)
            if L_max and z < L_max:
                ax.axhline(z, color="#CC4444", lw=0.9, ls="--", alpha=0.7)

            # Linha de separação entre painéis
            if sliced and pi > 0:
                ax.axhline(y_lo, color="#888888", lw=0.8, ls=":", alpha=0.6)

            # Peças
            for item in sheet:
                i_y0 = item["y"]
                i_y1 = item["y"] + item["height"]
                if i_y1 <= y_lo or i_y0 >= y_hi:
                    continue
                cor = CORES.get(item["tipo"], CORE_DEFAULT)
                pts = [(item["x"] + px, item["y"] + py) for px, py in item["polygon"]]
                ax.add_patch(MplPolygon(pts, closed=True, facecolor=cor,
                                        edgecolor="#333", lw=0.6, alpha=0.85, zorder=2))
                cx = item["x"] + item["width"] / 2
                cy = item["y"] + item["height"] / 2
                if y_lo <= cy < y_hi:
                    ax.text(cx, cy, _abbrev(item["tipo"]), ha="center", va="center",
                            fontsize=5, color="#111", fontweight="bold", zorder=3)
                seen_tipos.setdefault(item["tipo"], item["label"])

            # Título
            if np_this == 1:
                ax.set_title("Chapa %d | %d pcs | z=%.0fmm" % (si+1,len(sheet),z),
                             fontsize=7, pad=3)
            elif pi == 0:
                ax.set_title(
                    ("Chapa %d (%d partes) | %d pcs | z=%.0fmm / parte 1: 0..%.0fmm" % (si+1,np_this,len(sheet),z,panel_h)),
                    fontsize=6, pad=2)
            else:
                ax.set_title(
                    "Chapa %d parte %d/%d | y=%.0f..%.0fmm" % (si+1,pi+1,np_this,y_lo,min(y_hi,z)),
                    fontsize=6, pad=2)

            ax.set_xlabel("Larg. (mm)", fontsize=6)
            if pi == 0:
                lbl = "Comp. (mm)" + (" [fatiado →]" if np_this > 1 else "")
                ax.set_ylabel(lbl, fontsize=6)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        col_offset += np_this

    # Legenda global
    handles = [mpatches.Patch(facecolor=CORES.get(t, CORE_DEFAULT),
                              edgecolor="#333", label=lbl)
               for t, lbl in seen_tipos.items()]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)),
               fontsize=6, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    n_total  = result["n_colocadas"]
    n_chapas = result["n_chapas"]
    tempo_ms = result["tempo_ms"]
    z_total  = result["comprimento_usado"]

    suptitle = (
        f"{exp_id} · {metodo} · Chapa {sheet_label}  |  "
        f"{n_total}/{result['n_total']} peças · {n_chapas} chapa(s) · "
        f"z total = {z_total:.0f} mm · Util={util:.1f}% · {tempo_ms:.1f} ms"
    )
    if sliced:
        suptitle += f"\n⚠ Layout fatiado em partes de {panel_h:.0f} mm para visualização"

    fig.suptitle(suptitle, fontsize=8, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metricas_experimento(results: list[dict], exp_id: str,
                              exp_desc: str, sheet_label: str,
                              out_path: str, fixed_length: bool = False) -> None:
    """
    Tabela comparativa completa com todas as métricas de C&P/Nesting:
      Qualidade   : z, z̲, GAP%
      Material    : Utilização%, Refugo%, Nº chapas*, Média pç/chapa*
      Computacional: Tempo(ms), Nº Restr.*, Nº Vars.Bin.*
    * só para modelos MIP (Modelo dos Pontos, Pontos+RA)
    """
    # Colunas fixas + fixas-only
    if fixed_length:
        colunas = ["Metodo","Pecas\ncolocadas","z (mm)","z LB\n(mm)","GAP %",
                   "Utilizacao\n(%)","Refugo\n(%)","N chapas","Media\npç/chapa",
                   "Tempo\n(ms)","N Restricoes","N Vars\nBinarias"]
    else:
        colunas = ["Metodo","Pecas\ncolocadas","z (mm)","z LB\n(mm)","GAP %",
                   "Utilizacao\n(%)","Refugo\n(%)","Area chapa\n(mm2)","Area pecas\n(mm2)",
                   "Tempo\n(ms)","N Restricoes","N Vars\nBinarias"]

    def _fmt_nr(v):
        if v is None: return "—"
        if v >= 1_000_000: return f"{v/1e6:.1f}M"
        if v >= 1_000:     return f"{v/1e3:.1f}k"
        return str(v)

    rows = []
    for r in results:
        gap_str = f"{r['gap_pct']:.1f}" if r.get('gap_pct') is not None else "—"
        if fixed_length:
            rows.append([
                r["metodo"],
                f"{r['n_colocadas']}/{r['n_total']}",
                f"{r['z']:.0f}",
                f"{r.get('z_lb', r['z']):.0f}",
                gap_str,
                f"{r['utilizacao']:.1f}",
                f"{r['refugo']:.1f}",
                str(r.get("n_chapas", 1)),
                f"{r.get('media_pecas_chapa', r['n_colocadas']):.1f}",
                f"{r['tempo_ms']:.1f}",
                _fmt_nr(r.get("n_restricoes")),
                _fmt_nr(r.get("n_vars_binarias")),
            ])
        else:
            rows.append([
                r["metodo"],
                f"{r['n_colocadas']}/{r['n_total']}",
                f"{r['z']:.0f}",
                f"{r.get('z_lb', r['z']):.0f}",
                gap_str,
                f"{r['utilizacao']:.1f}",
                f"{r['refugo']:.1f}",
                f"{r['area_chapa']:,.0f}",
                f"{r['area_pecas']:,.0f}",
                f"{r['tempo_ms']:.1f}",
                _fmt_nr(r.get("n_restricoes")),
                _fmt_nr(r.get("n_vars_binarias")),
            ])

    ncols = len(colunas)
    fig_w = max(14.0, 1.2 * ncols)
    fig_h = 1.2 + 0.5 * len(rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=colunas,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    # Grupos de colunas por cor de cabeçalho
    if fixed_length:
        groups = {
            "qualidade":    (0, 5,  "#1A5276"),   # método..GAP
            "material":     (5, 9,  "#145A32"),   # util..media
            "computacional":(9, 12, "#4A235A"),   # tempo..vars
        }
    else:
        groups = {
            "qualidade":    (0, 5,  "#1A5276"),
            "material":     (5, 10, "#145A32"),
            "computacional":(10, 12,"#4A235A"),
        }

    for _, (c_lo, c_hi, cor) in groups.items():
        for j in range(c_lo, c_hi):
            cell = tbl[0, j]
            cell.set_facecolor(cor)
            cell.set_text_props(color="white", fontweight="bold")

    # Linhas alternadas por método
    for i, row in enumerate(rows):
        met = row[0]
        cor = METODO_CORES.get(met, "#EEEEEE")
        for j in range(ncols):
            tbl[i+1, j].set_facecolor(cor + "28")

    tipo_chapa = "Comprimento fixo" if fixed_length else "Comprimento livre"
    fig.suptitle(
        f"{exp_id} · {exp_desc}  |  Chapa {sheet_label} · {tipo_chapa}",
        fontsize=10, fontweight="bold", y=0.98
    )

    # Legenda dos grupos de colunas
    import matplotlib.patches as mp
    legend_patches = [
        mp.Patch(color="#1A5276", label="Qualidade da solução"),
        mp.Patch(color="#145A32", label="Eficiência de material"),
        mp.Patch(color="#4A235A", label="Desempenho computacional"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparativo_geral(all_results: list[dict], out_dir: str) -> None:
    df = pd.DataFrame(all_results)
    metodos = df["metodo"].unique().tolist()
    exp_ids = df["exp_id"].unique().tolist()
    sheets  = df["sheet_label"].unique().tolist()

    for sheet in sheets:
        sub = df[df["sheet_label"] == sheet]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Comparativo Geral — Chapa {sheet}", fontsize=12, fontweight="bold")
        x = np.arange(len(exp_ids)); w = 0.13

        for k, met in enumerate(metodos):
            vu = [sub[(sub["exp_id"]==e)&(sub["metodo"]==met)]["utilizacao"].values for e in exp_ids]
            vu = [v[0] if len(v) > 0 else 0 for v in vu]
            vt = [sub[(sub["exp_id"]==e)&(sub["metodo"]==met)]["tempo_ms"].values for e in exp_ids]
            vt = [v[0] if len(v) > 0 else 0 for v in vt]
            axes[0].bar(x+k*w, vu, w, label=met, color=METODO_CORES.get(met, "#AAAAAA"), alpha=0.85)
            axes[1].bar(x+k*w, vt, w, label=met, color=METODO_CORES.get(met, "#AAAAAA"), alpha=0.85)

        for ax, ylabel, title, ylim in [
            (axes[0], "Utilização (%)", "Taxa de Utilização", (0, 105)),
            (axes[1], "Tempo (ms)",     "Tempo de Execução",  None)
        ]:
            ax.set_xticks(x + w*(len(metodos)-1)/2)
            ax.set_xticklabels(exp_ids, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            if ylim: ax.set_ylim(*ylim)
            ax.legend(fontsize=7)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        safe = sheet.replace("×", "x").replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"comparativo_{safe}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_csv(all_rows, out_path):
    pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def _abbrev(tipo):
    return {"corte_inclinado":"CI","reducao":"RC",
            "boca_de_lobo":"BL","curva_gomos":"CG"}.get(tipo, "?")
