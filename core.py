"""Core bioinformatics logic for the Protein Alignment-Based Phylogenetic Signal Tool.

No Streamlit code lives here, so everything can be unit-tested on its own.
Pipeline: FASTA -> validation -> pairwise alignment -> distance matrix
          -> tree (NJ / UPGMA) -> phylogenetic signal (Blomberg's K).
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AA = set("XBZJUO")  # accepted, but flagged
ALLOWED_AA = STANDARD_AA | AMBIGUOUS_AA


# --------------------------------------------------------------------------- #
# FASTA parsing & validation
# --------------------------------------------------------------------------- #
@dataclass
class ParseResult:
    records: list[SeqRecord]
    warnings: list[str]
    errors: list[str]


def parse_fasta(text: str) -> ParseResult:
    """Parse FASTA text, clean and validate protein sequences."""
    warnings: list[str] = []
    errors: list[str] = []
    records: list[SeqRecord] = []

    text = (text or "").strip()
    if not text:
        return ParseResult([], [], ["No input provided."])
    if not text.startswith(">"):
        # Tolerate a single bare sequence by giving it a header.
        text = ">Sequence_1\n" + text

    seen: dict[str, int] = {}
    for rec in SeqIO.parse(io.StringIO(text), "fasta"):
        name = rec.id.strip() or "unnamed"
        raw = str(rec.seq).upper().replace(" ", "")
        cleaned = raw.replace("*", "").replace("-", "")
        if cleaned != raw:
            warnings.append(f"{name}: removed stop codon (*) / gap (-) characters.")
        if not cleaned:
            errors.append(f"{name}: empty sequence.")
            continue
        bad = sorted(set(cleaned) - ALLOWED_AA)
        if bad:
            errors.append(
                f"{name}: invalid amino-acid characters {', '.join(bad)} "
                "(looks like DNA/RNA or a typo?)."
            )
            continue
        amb = sorted(set(cleaned) & AMBIGUOUS_AA)
        if amb:
            warnings.append(f"{name}: contains ambiguous residues {', '.join(amb)}.")
        if set(cleaned) <= set("ACGTUN") and len(cleaned) > 10:
            warnings.append(f"{name}: composed only of A/C/G/T/U/N - is this a nucleotide sequence?")
        if name in seen:  # keep names unique for matrix / tree labels
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        records.append(SeqRecord(Seq(cleaned), id=name, description=""))

    if len(records) < 2 and not errors:
        errors.append("At least 2 valid sequences are required (3+ for a tree).")
    elif len(records) < 3 and records and not errors:
        warnings.append("Only 2 sequences: a tree and phylogenetic signal need at least 3.")
    return ParseResult(records, warnings, errors)


# --------------------------------------------------------------------------- #
# Pairwise alignment & distances
# --------------------------------------------------------------------------- #
def make_aligner(matrix: str = "BLOSUM62", gap_open: float = -10.0, gap_extend: float = -0.5,
                 mode: str = "global") -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load(matrix)
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    return aligner


@dataclass
class PairResult:
    a: str
    b: str
    aligned_a: str
    aligned_b: str
    score: float
    identity: float      # matches / alignment columns
    similarity: float    # positive-scoring columns / alignment columns
    gaps: int
    length: int


def align_pair(aligner: PairwiseAligner, a: SeqRecord, b: SeqRecord) -> PairResult:
    aln = aligner.align(str(a.seq), str(b.seq))[0]
    aligned_a, aligned_b = _gapped(aln)
    L = len(aligned_a)
    matches = sim = gaps = 0
    mat = aligner.substitution_matrix
    for x, y in zip(aligned_a, aligned_b):
        if x == "-" or y == "-":
            gaps += 1
        elif x == y:
            matches += 1
            sim += 1
        else:
            try:
                if mat[x, y] > 0:
                    sim += 1
            except (KeyError, IndexError):
                pass
    return PairResult(a.id, b.id, aligned_a, aligned_b, float(aln.score),
                      matches / L, sim / L, gaps, L)


def _gapped(aln) -> tuple[str, str]:
    """Return the two gapped strings of a Biopython pairwise alignment."""
    seq_a, seq_b = aln.sequences[0], aln.sequences[1]
    out_a, out_b = [], []
    pa = pb = 0
    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
        gap_a, gap_b = a0 - pa, b0 - pb
        if gap_a:
            out_a.append(seq_a[pa:a0]); out_b.append("-" * gap_a)
        if gap_b:
            out_a.append("-" * gap_b); out_b.append(seq_b[pb:b0])
        out_a.append(seq_a[a0:a1]); out_b.append(seq_b[b0:b1])
        pa, pb = a1, b1
    if len(seq_a) - pa:
        out_a.append(seq_a[pa:]); out_b.append("-" * (len(seq_a) - pa))
    if len(seq_b) - pb:
        out_a.append("-" * (len(seq_b) - pb)); out_b.append(seq_b[pb:])
    return "".join(out_a), "".join(out_b)


def all_pairs(records: list[SeqRecord], aligner: PairwiseAligner) -> dict[tuple[int, int], PairResult]:
    out = {}
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            out[(i, j)] = align_pair(aligner, records[i], records[j])
    return out


def distance_from_identity(identity: float, method: str = "p-distance") -> float:
    p = 1.0 - identity
    if method == "Poisson-corrected":
        # d = -ln(1 - p); saturates as p -> 1, so cap to keep it finite
        return -math.log(max(1.0 - p, 1e-3)) if p < 1 else -math.log(1e-3)
    return p


def build_matrices(records, pairs, method="p-distance") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (distance_df, identity_df) indexed by sequence id."""
    names = [r.id for r in records]
    n = len(names)
    dist = np.zeros((n, n))
    ident = np.eye(n)
    for (i, j), res in pairs.items():
        d = distance_from_identity(res.identity, method)
        dist[i, j] = dist[j, i] = d
        ident[i, j] = ident[j, i] = res.identity
    return (pd.DataFrame(dist, index=names, columns=names),
            pd.DataFrame(ident, index=names, columns=names))


# --------------------------------------------------------------------------- #
# Trees
# --------------------------------------------------------------------------- #
def build_tree(dist_df: pd.DataFrame, method: str = "NJ"):
    names = list(dist_df.index)
    n = len(names)
    lower = [[float(dist_df.iloc[i, j]) for j in range(i + 1)] for i in range(n)]
    dm = DistanceMatrix(names, lower)
    ctor = DistanceTreeConstructor()
    tree = ctor.nj(dm) if method == "NJ" else ctor.upgma(dm)
    # NJ can yield tiny negative branch lengths - clip to zero.
    for clade in tree.find_clades():
        if clade.branch_length is not None and clade.branch_length < 0:
            clade.branch_length = 0.0
    if method == "NJ":
        _midpoint_root(tree)
    return tree


def _midpoint_root(tree) -> None:
    try:
        tree.root_at_midpoint()
    except Exception:
        pass


def tree_to_newick(tree) -> str:
    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    return buf.getvalue().strip()


def shared_path_matrix(tree, names: list[str]) -> np.ndarray:
    """C[i, j] = root-to-MRCA distance (phylogenetic variance-covariance under Brownian motion)."""
    n = len(names)
    C = np.zeros((n, n))
    tips = {t.name: t for t in tree.get_terminals()}
    for i in range(n):
        C[i, i] = tree.distance(tips[names[i]])
        for j in range(i + 1, n):
            mrca = tree.common_ancestor(tips[names[i]], tips[names[j]])
            C[i, j] = C[j, i] = tree.distance(mrca)
    return C


# --------------------------------------------------------------------------- #
# Phylogenetic signal
# --------------------------------------------------------------------------- #
def parse_traits(text: str) -> tuple[dict[str, float], list[str]]:
    """Parse 'name,value' / 'name<TAB>value' / 'name value' lines. Header line is skipped."""
    traits: dict[str, float] = {}
    problems: list[str] = []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").replace(";", ",").split(",")]
        if len(parts) == 1:
            parts = line.split()
        if len(parts) < 2:
            problems.append(f"Could not read line: '{line}'")
            continue
        try:
            traits[parts[0]] = float(parts[1])
        except ValueError:
            if not traits:   # treat as a header
                continue
            problems.append(f"Non-numeric value for '{parts[0]}': '{parts[1]}'")
    return traits, problems


def blomberg_k(x: np.ndarray, C: np.ndarray) -> float:
    n = len(x)
    Cinv = np.linalg.pinv(C)
    one = np.ones(n)
    a = (one @ Cinv @ x) / (one @ Cinv @ one)
    e = x - a
    mse0 = (e @ e) / (n - 1)
    mse = (e @ Cinv @ e) / (n - 1)
    expected = (np.trace(C) - n / (one @ Cinv @ one)) / (n - 1)
    if mse == 0 or expected == 0:
        return float("nan")
    return float((mse0 / mse) / expected)


def phylogenetic_signal(tree, names: list[str], traits: dict[str, float],
                        n_perm: int = 999, seed: int = 42) -> dict:
    missing = [n for n in names if n not in traits]
    if missing:
        raise ValueError("Trait value missing for: " + ", ".join(missing))
    x = np.array([traits[n] for n in names], dtype=float)
    if len(names) < 3:
        raise ValueError("Need at least 3 taxa.")
    if np.ptp(x) == 0:
        raise ValueError("Trait has no variation.")
    C = shared_path_matrix(tree, names)
    k_obs = blomberg_k(x, C)
    rng = np.random.default_rng(seed)
    perm = np.array([blomberg_k(rng.permutation(x), C) for _ in range(n_perm)])
    p = (1 + np.sum(perm >= k_obs)) / (n_perm + 1)
    return {"K": k_obs, "p": float(p), "perm": perm, "n": len(names)}


def interpret_k(k: float, p: float) -> str:
    if p >= 0.05:
        return ("No significant phylogenetic signal: closely related sequences are not "
                "more similar in this trait than expected by chance.")
    if k < 0.9:
        return ("Significant but weaker-than-Brownian signal (K < 1): relatives resemble each "
                "other, but less than neutral drift predicts - suggests adaptive divergence or "
                "evolutionary lability.")
    if k <= 1.1:
        return ("Significant signal consistent with Brownian motion (K ≈ 1): the trait "
                "evolves along the tree as expected under neutral drift.")
    return ("Significant, stronger-than-Brownian signal (K > 1): close relatives are "
            "more similar than neutral drift predicts - suggests conservatism / stabilizing "
            "selection within clades.")


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def summarize(records, pairs, dist_df, ident_df) -> list[str]:
    lines = []
    best = max(pairs.items(), key=lambda kv: kv[1].identity)[1]
    worst = min(pairs.items(), key=lambda kv: kv[1].identity)[1]
    lines.append(f"**Most similar pair:** {best.a} and {best.b} - {best.identity:.1%} identity "
                 f"({best.similarity:.1%} similarity).")
    lines.append(f"**Most divergent pair:** {worst.a} and {worst.b} - {worst.identity:.1%} identity.")
    mean_id = np.mean([p.identity for p in pairs.values()])
    lines.append(f"**Mean pairwise identity:** {mean_id:.1%} across {len(pairs)} pairs.")
    if mean_id >= 0.9:
        lines.append("Sequences are highly conserved - likely orthologs from closely related taxa or recent paralogs.")
    elif mean_id >= 0.4:
        lines.append("Moderate identity - consistent with a shared protein family or homologs diverged over a long time.")
    else:
        lines.append("Low identity (<40%) - in the 'twilight zone'; homology is uncertain and distances "
                     "(and tree topology) should be interpreted cautiously.")
    # most 'isolated' sequence
    mean_d = dist_df.sum(axis=1) / (len(records) - 1)
    out = mean_d.idxmax()
    lines.append(f"**Most isolated sequence:** {out} (mean distance {mean_d.max():.3f}) - the longest-diverged lineage.")
    return lines


# --------------------------------------------------------------------------- #
# Simulated demo data (clearly synthetic - NOT real proteins)
# --------------------------------------------------------------------------- #
def simulated_demo(seed: int = 7, length: int = 140) -> tuple[str, str]:
    """Evolve a random 'ancestral' protein down a fixed 8-taxon tree, plus a trait under
    Brownian motion along the same tree. Returns (fasta_text, traits_text)."""
    rng = random.Random(seed)
    aas = "ACDEFGHIKLMNPQRSTVWY"
    root = [rng.choice(aas) for _ in range(length)]
    # (name | subtree, branch_length)
    topo = [
        ([("Alpha", 0.10), ("Beta", 0.12)], 0.15),
        ([("Gamma", 0.08), ("Delta", 0.10)], 0.25),
        ([([("Epsilon", 0.06), ("Zeta", 0.07)], 0.20), ("Eta", 0.30)], 0.35),
    ]
    out_seqs: dict[str, str] = {}
    out_traits: dict[str, float] = {}

    def mutate(seq, bl):
        s = seq[:]
        for i in range(len(s)):
            if rng.random() < bl:
                s[i] = rng.choice(aas)
        return s

    def walk(node, seq, trait):
        if isinstance(node, str):
            out_seqs[node] = "".join(seq)
            out_traits[node] = round(trait, 3)
            return
        for child, bl in node:
            walk_branch(child, bl, seq, trait)

    def walk_branch(child, bl, seq, trait):
        s2 = mutate(seq, bl)
        t2 = trait + rng.gauss(0, math.sqrt(bl))
        if isinstance(child, str):
            out_seqs[child] = "".join(s2)
            out_traits[child] = round(t2, 3)
        else:
            walk(child, s2, t2)

    walk(topo, root, 0.0)
    fasta = "\n".join(f">{n}\n{s}" for n, s in out_seqs.items())
    traits = "name,value\n" + "\n".join(f"{n},{v}" for n, v in out_traits.items())
    return fasta, traits
