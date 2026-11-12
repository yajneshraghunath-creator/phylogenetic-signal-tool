import core

def test_pipeline():
    f, t = core.simulated_demo()
    r = core.parse_fasta(f)
    assert not r.errors and len(r.records) == 7
    pairs = core.all_pairs(r.records, core.make_aligner())
    d, i = core.build_matrices(r.records, pairs)
    assert (d.values >= 0).all() and abs(i.values.diagonal() - 1).max() < 1e-9
    tree = core.build_tree(d, "NJ")
    sig = core.phylogenetic_signal(tree, list(d.index), core.parse_traits(t)[0], 199)
    assert sig["p"] < 0.05

def test_validation():
    assert core.parse_fasta(">a\nMK1\n>b\nMKT").errors
    assert core.parse_fasta("").errors

def test_identical_seqs():
    r = core.parse_fasta(">a\nMKTAYIAK\n>b\nMKTAYIAK")
    p = core.align_pair(core.make_aligner(), *r.records)
    assert p.identity == 1.0 and p.gaps == 0

if __name__ == "__main__":
    test_pipeline(); test_validation(); test_identical_seqs(); print("ok")
