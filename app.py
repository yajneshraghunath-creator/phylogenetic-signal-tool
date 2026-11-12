"""Protein Sequence Alignment-Based Phylogenetic Signal Tool - Streamlit interface."""
from __future__ import annotations

import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from Bio import Phylo

import core

st.set_page_config(page_title="Phylogenetic Signal Tool", page_icon="🧬", layout="wide")

SAMPLE_REAL_NOTE = (
    "Demo data is **simulated** (a random ancestral protein evolved down a fixed 7-taxon tree, "
    "with a trait evolving by Brownian motion) - it is not real protein data."
)

# ------------------------------- sidebar ----------------------------------- #
with st.sidebar:
    st.title("🧬 Settings")
    st.subheader("Alignment")
    matrix = st.selectbox("Substitution matrix", ["BLOSUM62", "BLOSUM45", "BLOSUM80", "PAM250"])
    gap_open = st.number_input("Gap open penalty", value=-10.0, max_value=0.0, step=1.0)
    gap_ext = st.number_input("Gap extend penalty", value=-0.5, max_value=0.0, step=0.5)
    mode = st.radio("Alignment mode", ["global", "local"], horizontal=True)
    st.subheader("Phylogeny")
    dist_method = st.radio("Distance", ["p-distance", "Poisson-corrected"],
                           help="p = 1 - identity. Poisson correction accounts for multiple substitutions.")
    tree_method = st.radio("Tree method", ["NJ", "UPGMA"], horizontal=True,
                           help="Neighbor-Joining (no clock assumption) or UPGMA (assumes a molecular clock).")
    st.subheader("Phylogenetic signal")
    n_perm = st.slider("Permutations", 99, 4999, 999, step=100)

# ------------------------------- header ------------------------------------ #
st.title("Protein Sequence Alignment-Based Phylogenetic Signal Tool")
st.caption("FASTA → validation → pairwise alignment → distance matrix → tree → phylogenetic signal (Blomberg's K)")

if "fasta" not in st.session_state:
    st.session_state.fasta = ""
    st.session_state.traits = ""

col_in, col_tr = st.columns([3, 2])
with col_in:
    st.subheader("1 · Protein sequences (FASTA)")
    b1, b2 = st.columns(2)
    if b1.button("Load simulated demo", width="stretch"):
        st.session_state.fasta, st.session_state.traits = core.simulated_demo()
    if b2.button("Clear", width="stretch"):
        st.session_state.fasta = st.session_state.traits = ""
    up = st.file_uploader("…or upload a FASTA file", type=["fasta", "fa", "faa", "txt"])
    if up is not None:
        st.session_state.fasta = up.getvalue().decode("utf-8", errors="replace")
    fasta_text = st.text_area("FASTA", key="fasta", height=220,
                              placeholder=">Seq1\nMKTAYIAKQRQISFVKSHFSRQ\n>Seq2\nMKTAYIAKQRQISFVKSHFTRQ\n>Seq3\n…")
with col_tr:
    st.subheader("2 · Trait values (optional)")
    st.caption("One `name,value` per line, names matching the FASTA headers. Needed for the signal test.")
    tup = st.file_uploader("…or upload a CSV", type=["csv", "txt", "tsv"], key="trait_up")
    if tup is not None:
        st.session_state.traits = tup.getvalue().decode("utf-8", errors="replace")
    trait_text = st.text_area("Traits", key="traits", height=270, placeholder="name,value\nSeq1,1.2\nSeq2,1.4")

st.caption(SAMPLE_REAL_NOTE)

# ------------------------------- parse ------------------------------------- #
parsed = core.parse_fasta(fasta_text) if fasta_text.strip() else None
if parsed is None:
    st.info("Paste or upload protein sequences, or click **Load simulated demo**, to begin.")
    st.stop()

for e in parsed.errors:
    st.error(e)
for w in parsed.warnings:
    st.warning(w)
if parsed.errors or len(parsed.records) < 2:
    st.stop()

records = parsed.records
if len(records) > 60:
    st.error("Please limit input to 60 sequences (all-vs-all alignment is quadratic).")
    st.stop()


@st.cache_data(show_spinner="Aligning…")
def run_alignments(fasta: str, matrix: str, gap_open: float, gap_ext: float, mode: str, dist_method: str):
    recs = core.parse_fasta(fasta).records
    al = core.make_aligner(matrix, gap_open, gap_ext, mode)
    pairs = core.all_pairs(recs, al)
    dist, ident = core.build_matrices(recs, pairs, dist_method)
    return recs, pairs, dist, ident


records, pairs, dist_df, ident_df = run_alignments(fasta_text, matrix, gap_open, gap_ext, mode, dist_method)
names = [r.id for r in records]

st.success(f"{len(records)} valid sequences · lengths {min(len(r.seq) for r in records)}-"
           f"{max(len(r.seq) for r in records)} aa · {len(pairs)} pairwise alignments")

tab_aln, tab_mat, tab_tree, tab_sig, tab_sum = st.tabs(
    ["Alignment", "Distance matrix", "Phylogenetic tree", "Phylogenetic signal", "Interpretation"]
)

# ------------------------------- alignment --------------------------------- #
with tab_aln:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Sequence A", names, index=0)
    b = c2.selectbox("Sequence B", names, index=1)
    if a == b:
        st.info("Pick two different sequences.")
    else:
        i, j = sorted((names.index(a), names.index(b)))
        res = pairs[(i, j)]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{res.score:.1f}")
        m2.metric("Identity", f"{res.identity:.1%}")
        m3.metric("Similarity", f"{res.similarity:.1%}")
        m4.metric("Gap columns", res.gaps)

        mat = core.make_aligner(matrix).substitution_matrix
        width = 60
        rows = []
        for s in range(0, res.length, width):
            ca, cb = res.aligned_a[s:s + width], res.aligned_b[s:s + width]
            mid, ha, hb = [], [], []
            for x, y in zip(ca, cb):
                if x == y:
                    mid.append("|"); cls = "m"
                elif "-" in (x, y):
                    mid.append(" "); cls = "g"
                else:
                    try:
                        pos = mat[x, y] > 0
                    except Exception:
                        pos = False
                    mid.append(":" if pos else "."); cls = "s" if pos else "x"
                ha.append(f'<span class="{cls}">{html.escape(x)}</span>')
                hb.append(f'<span class="{cls}">{html.escape(y)}</span>')
            rows.append(
                f'<div class="blk"><b>{html.escape(res.a[:10]):<10}</b> {s + 1:>4} {"".join(ha)}\n'
                f'<b>{"":<10}</b>      <span class="mid">{html.escape("".join(mid))}</span>\n'
                f'<b>{html.escape(res.b[:10]):<10}</b> {s + 1:>4} {"".join(hb)}</div>'
            )
        st.markdown(
            """<style>
            .aln{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;white-space:pre;overflow-x:auto}
            .blk{margin-bottom:14px}
            .aln .m{background:#2e7d3233;color:inherit} .aln .s{background:#f9a82544}
            .aln .x{background:#c6282833} .aln .g{background:#78909c44} .aln .mid{opacity:.7}
            </style><div class="aln">""" + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Green = identical · amber = conservative substitution (| identity, : similar, . different) · "
                   "red = non-conservative · grey = gap")

# ------------------------------- matrix ------------------------------------ #
with tab_mat:
    which = st.radio("Show", ["Distance", "% Identity"], horizontal=True)
    df = dist_df if which == "Distance" else ident_df * 100
    fig = px.imshow(df, text_auto=".2f" if which == "Distance" else ".1f", aspect="auto",
                    color_continuous_scale="Viridis_r" if which == "Distance" else "Viridis",
                    labels=dict(color=which))
    fig.update_layout(height=max(420, 60 * len(names)), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.round(4), width="stretch")
    st.download_button("Download matrix (CSV)", df.to_csv().encode(), f"{which.lower().replace(' ', '_')}_matrix.csv")

# ------------------------------- tree -------------------------------------- #
tree = None
if len(records) >= 3:
    tree = core.build_tree(dist_df, tree_method)

with tab_tree:
    if tree is None:
        st.info("A tree needs at least 3 sequences.")
    else:
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(names))))
        Phylo.draw(tree, axes=ax, do_show=False,
                   label_func=lambda c: c.name if c.is_terminal() else "")
        ax.set_xlabel(f"Branch length ({dist_method}, substitutions/site-equivalent)")
        ax.set_ylabel("")
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        st.pyplot(fig, width="stretch")
        newick = core.tree_to_newick(tree)
        st.code(newick, language=None)
        d1, d2 = st.columns(2)
        d1.download_button("Download Newick", newick, "tree.nwk")
        buf = __import__("io").BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        d2.download_button("Download PNG", buf.getvalue(), "tree.png", mime="image/png")
        st.caption(f"{tree_method} tree, midpoint-rooted" if tree_method == "NJ" else "UPGMA tree (rooted, ultrametric)")

# ------------------------------- signal ------------------------------------ #
with tab_sig:
    if tree is None:
        st.info("Phylogenetic signal needs a tree (3+ sequences).")
    elif not trait_text.strip():
        st.info("Add trait values in step 2 (or load the demo) to test phylogenetic signal.")
    else:
        traits, problems = core.parse_traits(trait_text)
        for p in problems:
            st.warning(p)
        try:
            sig = core.phylogenetic_signal(tree, names, traits, n_perm)
        except ValueError as e:
            st.error(str(e))
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Blomberg's K", f"{sig['K']:.3f}")
            k2.metric("p-value (permutation)", f"{sig['p']:.4f}")
            k3.metric("Taxa", sig["n"])
            (st.success if sig["p"] < 0.05 else st.info)(core.interpret_k(sig["K"], sig["p"]))
            h = go.Figure(go.Histogram(x=sig["perm"], nbinsx=40, name="Permuted K"))
            h.add_vline(x=sig["K"], line_color="crimson", annotation_text="observed K")
            h.update_layout(title="Null distribution of K (trait values shuffled across tips)",
                            xaxis_title="K", yaxis_title="Count", height=340, showlegend=False)
            st.plotly_chart(h, width="stretch")
            st.caption("K compares observed trait variance among tips to that expected under Brownian "
                       "motion on this tree (K≈1). Significance comes from shuffling trait values across tips. "
                       "Interpret cautiously with few taxa: power is low below ~20.")

# ------------------------------- summary ----------------------------------- #
with tab_sum:
    for line in core.summarize(records, pairs, dist_df, ident_df):
        st.markdown("- " + line)
    st.caption("Distances come from pairwise alignments, so they depend on the matrix and gap settings. "
               "Phylogenetic signal assumes the tree reflects true relatedness.")
