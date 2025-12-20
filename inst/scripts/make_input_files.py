#! /usr/bin/env python3

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pandas import DataFrame, read_csv, Series
import numpy as np
import logging


DATA_SAVE_PATH = "/home/fm/Classes/cs84040/project/data/"

DIPEP = ["aa", "ac", "ad", "ae", "af", "ag", "ah", "ai", "ak", "al", "am",
         "an", "ap", "aq", "ar", "as", "at", "av", "aw", "ay", "ca", "cc",
         "cd", "ce", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn", "cp",
         "cq", "cr", "cs", "ct", "cv", "cw", "cy", "da", "dc", "dd", "de",
         "df", "dg", "dh", "di", "dk", "dl", "dm", "dn", "dp", "dq", "dr",
         "ds", "dt", "dv", "dw", "dy", "ea", "ec", "ed", "ee", "ef", "eg",
         "eh", "ei", "ek", "el", "em", "en", "ep", "eq", "er", "es", "et",
         "ev", "ew", "ey", "fa", "fc", "fd", "fe", "ff", "fg", "fh", "fi",
         "fk", "fl", "fm", "fn", "fp", "fq", "fr", "fs", "ft", "fv", "fw",
         "fy", "ga", "gc", "gd", "ge", "gf", "gg", "gh", "gi", "gk", "gl",
         "gm", "gn", "gp", "gq", "gr", "gs", "gt", "gv", "gw", "gy", "ha",
         "hc", "hd", "he", "hf", "hg", "hh", "hi", "hk", "hl", "hm", "hn",
         "hp", "hq", "hr", "hs", "ht", "hv", "hw", "hy", "ia", "ic", "id",
         "ie", "if", "ig", "ih", "ii", "ik", "il", "im", "in", "ip", "iq",
         "ir", "is", "it", "iv", "iw", "iy", "ka", "kc", "kd", "ke", "kf",
         "kg", "kh", "ki", "kk", "kl", "km", "kn", "kp", "kq", "kr", "ks",
         "kt", "kv", "kw", "ky", "la", "lc", "ld", "le", "lf", "lg", "lh",
         "li", "lk", "ll", "lm", "ln", "lp", "lq", "lr", "ls", "lt", "lv",
         "lw", "ly", "ma", "mc", "md", "me", "mf", "mg", "mh", "mi", "mk",
         "ml", "mm", "mn", "mp", "mq", "mr", "ms", "mt", "mv", "mw", "my",
         "na", "nc", "nd", "ne", "nf", "ng", "nh", "ni", "nk", "nl", "nm",
         "nn", "np", "nq", "nr", "ns", "nt", "nv", "nw", "ny", "pa", "pc",
         "pd", "pe", "pf", "pg", "ph", "pi", "pk", "pl", "pm", "pn", "pp",
         "pq", "pr", "ps", "pt", "pv", "pw", "py", "qa", "qc", "qd", "qe",
         "qf", "qg", "qh", "qi", "qk", "ql", "qm", "qn", "qp", "qq", "qr",
         "qs", "qt", "qv", "qw", "qy", "ra", "rc", "rd", "re", "rf", "rg",
         "rh", "ri", "rk", "rl", "rm", "rn", "rp", "rq", "rr", "rs", "rt",
         "rv", "rw", "ry", "sa", "sc", "sd", "se", "sf", "sg", "sh", "si",
         "sk", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sv", "sw",
         "sy", "ta", "tc", "td", "te", "tf", "tg", "th", "ti", "tk", "tl",
         "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tw", "ty", "va",
         "vc", "vd", "ve", "vf", "vg", "vh", "vi", "vk", "vl", "vm", "vn",
         "vp", "vq", "vr", "vs", "vt", "vv", "vw", "vy", "wa", "wc", "wd",
         "we", "wf", "wg", "wh", "wi", "wk", "wl", "wm", "wn", "wp", "wq",
         "wr", "ws", "wt", "wv", "ww", "wy", "ya", "yc", "yd", "ye", "yf",
         "yg", "yh", "yi", "yk", "yl", "ym", "yn", "yp", "yq", "yr", "ys",
         "yt", "yv", "yw", "yy"]

DI_REDUCE = ["aa", "ac", "ae", "af", "ag", "ah", "ak", "al", "ap", "as",
             "ca", "cc", "ce", "cf", "cg", "ch", "ck", "cl", "cp", "cs",
             "ea", "ec", "ee", "ef", "eg", "eh", "ek", "el", "ep", "es",
             "fa", "fc", "fe", "ff", "fg", "fh", "fk", "fl", "fp", "fs",
             "ga", "gc", "ge", "gf", "gg", "gh", "gk", "gl", "gp", "gs",
             "ha", "hc", "he", "hf", "hg", "hh", "hk", "hl", "hp", "hs",
             "ka", "kc", "ke", "kf", "kg", "kh", "kk", "kl", "kp", "ks",
             "la", "lc", "le", "lf", "lg", "lh", "lk", "ll", "lp", "ls",
             "pa", "pc", "pe", "pf", "pg", "ph", "pk", "pl", "pp", "ps",
             "sa", "sc", "se", "sf", "sg", "sh", "sk", "sl", "sp", "ss"]

TR_REDUCE = ["aaa", "aac", "aae", "aaf", "aag", "aah", "aak", "aal", "aap",
             "aas", "aca", "acc", "ace", "acf", "acg", "ach", "ack", "acl",
             "acp", "acs", "aea", "aec", "aee", "aef", "aeg", "aeh", "aek",
             "ael", "aep", "aes", "afa", "afc", "afe", "aff", "afg", "afh",
             "afk", "afl", "afp", "afs", "aga", "agc", "age", "agf", "agg",
             "agh", "agk", "agl", "agp", "ags", "aha", "ahc", "ahe", "ahf",
             "ahg", "ahh", "ahk", "ahl", "ahp", "ahs", "aka", "akc", "ake",
             "akf", "akg", "akh", "akk", "akl", "akp", "aks", "ala", "alc",
             "ale", "alf", "alg", "alh", "alk", "all", "alp", "als", "apa",
             "apc", "ape", "apf", "apg", "aph", "apk", "apl", "app", "aps",
             "asa", "asc", "ase", "asf", "asg", "ash", "ask", "asl", "asp",
             "ass", "caa", "cac", "cae", "caf", "cag", "cah", "cak", "cal",
             "cap", "cas", "cca", "ccc", "cce", "ccf", "ccg", "cch", "cck",
             "ccl", "ccp", "ccs", "cea", "cec", "cee", "cef", "ceg", "ceh",
             "cek", "cel", "cep", "ces", "cfa", "cfc", "cfe", "cff", "cfg",
             "cfh", "cfk", "cfl", "cfp", "cfs", "cga", "cgc", "cge", "cgf",
             "cgg", "cgh", "cgk", "cgl", "cgp", "cgs", "cha", "chc", "che",
             "chf", "chg", "chh", "chk", "chl", "chp", "chs", "cka", "ckc",
             "cke", "ckf", "ckg", "ckh", "ckk", "ckl", "ckp", "cks", "cla",
             "clc", "cle", "clf", "clg", "clh", "clk", "cll", "clp", "cls",
             "cpa", "cpc", "cpe", "cpf", "cpg", "cph", "cpk", "cpl", "cpp",
             "cps", "csa", "csc", "cse", "csf", "csg", "csh", "csk", "csl",
             "csp", "css", "eaa", "eac", "eae", "eaf", "eag", "eah", "eak",
             "eal", "eap", "eas", "eca", "ecc", "ece", "ecf", "ecg", "ech",
             "eck", "ecl", "ecp", "ecs", "eea", "eec", "eee", "eef", "eeg",
             "eeh", "eek", "eel", "eep", "ees", "efa", "efc", "efe", "eff",
             "efg", "efh", "efk", "efl", "efp", "efs", "ega", "egc", "ege",
             "egf", "egg", "egh", "egk", "egl", "egp", "egs", "eha", "ehc",
             "ehe", "ehf", "ehg", "ehh", "ehk", "ehl", "ehp", "ehs", "eka",
             "ekc", "eke", "ekf", "ekg", "ekh", "ekk", "ekl", "ekp", "eks",
             "ela", "elc", "ele", "elf", "elg", "elh", "elk", "ell", "elp",
             "els", "epa", "epc", "epe", "epf", "epg", "eph", "epk", "epl",
             "epp", "eps", "esa", "esc", "ese", "esf", "esg", "esh", "esk",
             "esl", "esp", "ess", "faa", "fac", "fae", "faf", "fag", "fah",
             "fak", "fal", "fap", "fas", "fca", "fcc", "fce", "fcf", "fcg",
             "fch", "fck", "fcl", "fcp", "fcs", "fea", "fec", "fee", "fef",
             "feg", "feh", "fek", "fel", "fep", "fes", "ffa", "ffc", "ffe",
             "fff", "ffg", "ffh", "ffk", "ffl", "ffp", "ffs", "fga", "fgc",
             "fge", "fgf", "fgg", "fgh", "fgk", "fgl", "fgp", "fgs", "fha",
             "fhc", "fhe", "fhf", "fhg", "fhh", "fhk", "fhl", "fhp", "fhs",
             "fka", "fkc", "fke", "fkf", "fkg", "fkh", "fkk", "fkl", "fkp",
             "fks", "fla", "flc", "fle", "flf", "flg", "flh", "flk", "fll",
             "flp", "fls", "fpa", "fpc", "fpe", "fpf", "fpg", "fph", "fpk",
             "fpl", "fpp", "fps", "fsa", "fsc", "fse", "fsf", "fsg", "fsh",
             "fsk", "fsl", "fsp", "fss", "gaa", "gac", "gae", "gaf", "gag",
             "gah", "gak", "gal", "gap", "gas", "gca", "gcc", "gce", "gcf",
             "gcg", "gch", "gck", "gcl", "gcp", "gcs", "gea", "gec", "gee",
             "gef", "geg", "geh", "gek", "gel", "gep", "ges", "gfa", "gfc",
             "gfe", "gff", "gfg", "gfh", "gfk", "gfl", "gfp", "gfs", "gga",
             "ggc", "gge", "ggf", "ggg", "ggh", "ggk", "ggl", "ggp", "ggs",
             "gha", "ghc", "ghe", "ghf", "ghg", "ghh", "ghk", "ghl", "ghp",
             "ghs", "gka", "gkc", "gke", "gkf", "gkg", "gkh", "gkk", "gkl",
             "gkp", "gks", "gla", "glc", "gle", "glf", "glg", "glh", "glk",
             "gll", "glp", "gls", "gpa", "gpc", "gpe", "gpf", "gpg", "gph",
             "gpk", "gpl", "gpp", "gps", "gsa", "gsc", "gse", "gsf", "gsg",
             "gsh", "gsk", "gsl", "gsp", "gss", "haa", "hac", "hae", "haf",
             "hag", "hah", "hak", "hal", "hap", "has", "hca", "hcc", "hce",
             "hcf", "hcg", "hch", "hck", "hcl", "hcp", "hcs", "hea", "hec",
             "hee", "hef", "heg", "heh", "hek", "hel", "hep", "hes", "hfa",
             "hfc", "hfe", "hff", "hfg", "hfh", "hfk", "hfl", "hfp", "hfs",
             "hga", "hgc", "hge", "hgf", "hgg", "hgh", "hgk", "hgl", "hgp",
             "hgs", "hha", "hhc", "hhe", "hhf", "hhg", "hhh", "hhk", "hhl",
             "hhp", "hhs", "hka", "hkc", "hke", "hkf", "hkg", "hkh", "hkk",
             "hkl", "hkp", "hks", "hla", "hlc", "hle", "hlf", "hlg", "hlh",
             "hlk", "hll", "hlp", "hls", "hpa", "hpc", "hpe", "hpf", "hpg",
             "hph", "hpk", "hpl", "hpp", "hps", "hsa", "hsc", "hse", "hsf",
             "hsg", "hsh", "hsk", "hsl", "hsp", "hss", "kaa", "kac", "kae",
             "kaf", "kag", "kah", "kak", "kal", "kap", "kas", "kca", "kcc",
             "kce", "kcf", "kcg", "kch", "kck", "kcl", "kcp", "kcs", "kea",
             "kec", "kee", "kef", "keg", "keh", "kek", "kel", "kep", "kes",
             "kfa", "kfc", "kfe", "kff", "kfg", "kfh", "kfk", "kfl", "kfp",
             "kfs", "kga", "kgc", "kge", "kgf", "kgg", "kgh", "kgk", "kgl",
             "kgp", "kgs", "kha", "khc", "khe", "khf", "khg", "khh", "khk",
             "khl", "khp", "khs", "kka", "kkc", "kke", "kkf", "kkg", "kkh",
             "kkk", "kkl", "kkp", "kks", "kla", "klc", "kle", "klf", "klg",
             "klh", "klk", "kll", "klp", "kls", "kpa", "kpc", "kpe", "kpf",
             "kpg", "kph", "kpk", "kpl", "kpp", "kps", "ksa", "ksc", "kse",
             "ksf", "ksg", "ksh", "ksk", "ksl", "ksp", "kss", "laa", "lac",
             "lae", "laf", "lag", "lah", "lak", "lal", "lap", "las", "lca",
             "lcc", "lce", "lcf", "lcg", "lch", "lck", "lcl", "lcp", "lcs",
             "lea", "lec", "lee", "lef", "leg", "leh", "lek", "lel", "lep",
             "les", "lfa", "lfc", "lfe", "lff", "lfg", "lfh", "lfk", "lfl",
             "lfp", "lfs", "lga", "lgc", "lge", "lgf", "lgg", "lgh", "lgk",
             "lgl", "lgp", "lgs", "lha", "lhc", "lhe", "lhf", "lhg", "lhh",
             "lhk", "lhl", "lhp", "lhs", "lka", "lkc", "lke", "lkf", "lkg",
             "lkh", "lkk", "lkl", "lkp", "lks", "lla", "llc", "lle", "llf",
             "llg", "llh", "llk", "lll", "llp", "lls", "lpa", "lpc", "lpe",
             "lpf", "lpg", "lph", "lpk", "lpl", "lpp", "lps", "lsa", "lsc",
             "lse", "lsf", "lsg", "lsh", "lsk", "lsl", "lsp", "lss", "paa",
             "pac", "pae", "paf", "pag", "pah", "pak", "pal", "pap", "pas",
             "pca", "pcc", "pce", "pcf", "pcg", "pch", "pck", "pcl", "pcp",
             "pcs", "pea", "pec", "pee", "pef", "peg", "peh", "pek", "pel",
             "pep", "pes", "pfa", "pfc", "pfe", "pff", "pfg", "pfh", "pfk",
             "pfl", "pfp", "pfs", "pga", "pgc", "pge", "pgf", "pgg", "pgh",
             "pgk", "pgl", "pgp", "pgs", "pha", "phc", "phe", "phf", "phg",
             "phh", "phk", "phl", "php", "phs", "pka", "pkc", "pke", "pkf",
             "pkg", "pkh", "pkk", "pkl", "pkp", "pks", "pla", "plc", "ple",
             "plf", "plg", "plh", "plk", "pll", "plp", "pls", "ppa", "ppc",
             "ppe", "ppf", "ppg", "pph", "ppk", "ppl", "ppp", "pps", "psa",
             "psc", "pse", "psf", "psg", "psh", "psk", "psl", "psp", "pss",
             "saa", "sac", "sae", "saf", "sag", "sah", "sak", "sal", "sap",
             "sas", "sca", "scc", "sce", "scf", "scg", "sch", "sck", "scl",
             "scp", "scs", "sea", "sec", "see", "sef", "seg", "seh", "sek",
             "sel", "sep", "ses", "sfa", "sfc", "sfe", "sff", "sfg", "sfh",
             "sfk", "sfl", "sfp", "sfs", "sga", "sgc", "sge", "sgf", "sgg",
             "sgh", "sgk", "sgl", "sgp", "sgs", "sha", "shc", "she", "shf",
             "shg", "shh", "shk", "shl", "shp", "shs", "ska", "skc", "ske",
             "skf", "skg", "skh", "skk", "skl", "skp", "sks", "sla", "slc",
             "sle", "slf", "slg", "slh", "slk", "sll", "slp", "sls", "spa",
             "spc", "spe", "spf", "spg", "sph", "spk", "spl", "spp", "sps",
             "ssa", "ssc", "sse", "ssf", "ssg", "ssh", "ssk", "ssl", "ssp",
             "sss"]

def calculate_peptide_frequencies(sequence: str, chain_length: int = 2,
                                  reduce_to: list = DIPEP,
                                  prefix: str | None = None) -> float:
    if chain_length not in [2, 3]:
        raise ValueError("chain_length must be 2 or 3")
    sequence = sequence.lower()
    chains = {}
    frequency = dict.fromkeys([prefix + k for k in reduce_to], 0.0)
    adjust = chain_length - 1
    length = len(sequence) - adjust
    for i in range(0, length):
        k = sequence[i:i+chain_length]
        label = k if prefix is None else prefix + k
        if k in chains.keys():
            chains[k] = chains[k] + 1
        else:
            chains[k] = 1
        if reduce_to is None or k in reduce_to:
            frequency[label] = chains[k] / length
    return frequency

def peptide_features(sequence: str) -> dict[str, float]:
    dipep = calculate_peptide_frequencies(sequence, prefix = "DIPEP.")
    red_dipep = calculate_peptide_frequencies(sequence, reduce_to = DI_REDUCE,
                                              prefix = "RED_DIPEP.")
    red_tripep = calculate_peptide_frequencies(sequence, chain_length = 3,
                                               reduce_to = TR_REDUCE,
                                               prefix = "RED_TRIPEP.")

    return dipep | red_dipep | red_tripep

def aa_grp(aa_group: str, percents: dict[str, float]) -> float:
    total = 0.0
    for aa in aa_group:
        if aa in percents.keys():
            total += percents[aa]
    return total

def bp_features(sequence: str) -> dict[str, float]:
    feat = {}
    X = ProteinAnalysis(sequence)

    feat["BP:molecular_weight"] = X.molecular_weight()
    feat["BP:instability_index"] = X.instability_index()
    feat["BP:isoelectric_point"] = X.isoelectric_point()
    feat["BP:molar_extinction_coefficient_reduced"] = \
        X.molar_extinction_coefficient()[0]
    feat["BP:molar_extinction_coefficient_cysteines"] = \
        X.molar_extinction_coefficient()[1]
    feat["BP:percent_helix_naive"] = X.secondary_structure_fraction()[0]
    feat["BP:percent_turn_naive"] = X.secondary_structure_fraction()[1]
    feat["BP:percent_strand_naive"] = X.secondary_structure_fraction()[2]
    feat["BP:percent:A"] = X.amino_acids_percent["A"]
    feat["BP:percent:C"] = X.amino_acids_percent["C"]
    feat["BP:percent:D"] = X.amino_acids_percent["D"]
    feat["BP:percent:E"] = X.amino_acids_percent["E"]
    feat["BP:percent:F"] = X.amino_acids_percent["F"]
    feat["BP:percent:G"] = X.amino_acids_percent["G"]
    feat["BP:percent:H"] = X.amino_acids_percent["H"]
    feat["BP:percent:I"] = X.amino_acids_percent["I"]
    feat["BP:percent:K"] = X.amino_acids_percent["K"]
    feat["BP:percent:L"] = X.amino_acids_percent["L"]
    feat["BP:percent:M"] = X.amino_acids_percent["M"]
    feat["BP:percent:N"] = X.amino_acids_percent["N"]
    feat["BP:percent:P"] = X.amino_acids_percent["P"]
    feat["BP:percent:Q"] = X.amino_acids_percent["Q"]
    feat["BP:percent:R"] = X.amino_acids_percent["R"]
    feat["BP:percent:S"] = X.amino_acids_percent["S"]
    feat["BP:percent:T"] = X.amino_acids_percent["T"]
    feat["BP:percent:V"] = X.amino_acids_percent["V"]
    feat["BP:percent:W"] = X.amino_acids_percent["W"]
    feat["BP:percent:Y"] = X.amino_acids_percent["Y"]
    feat["BP:flex:min"] = min(X.flexibility())
    feat["BP:flex:max"] = max(X.flexibility())
    feat["BP:flex:std"] = np.std(X.flexibility())
    feat["BP:gravy"] = X.gravy()
    feat["BP:prop_res_Polar"] = aa_grp("STCNQHY", X.amino_acids_percent)
    feat["BP:prop_res_Aliphatic"] = aa_grp("GAVLIP", X.amino_acids_percent)
    feat["BP:prop_res_Aromatic"] = aa_grp("FWY", X.amino_acids_percent)
    feat["BP:prop_res_Basic"] = aa_grp("KRH", X.amino_acids_percent)
    feat["BP:prop_res_Small"] = aa_grp("GASP", X.amino_acids_percent)
    feat["BP:prop_res_Acidic"] = aa_grp("DE", X.amino_acids_percent)
    feat["BP:prop_res_Charged"] = aa_grp("DEHKR", X.amino_acids_percent)
    feat["BP:prop_res_Tiny"] = aa_grp("GAS", X.amino_acids_percent)
    feat["BP:prop_res_Non.polar"] = aa_grp("GAVLIMP", X.amino_acids_percent)
    return feat

def generate_features(df: DataFrame) -> DataFrame:
    features = []
    for i, row in df.iterrows():
        print(f"Processing {i} {row["Meta"]}")
        try:
            bp_feats = bp_features(row["Sequence"])
        except ValueError as e:
            logger.info(f"{i} {row['Meta']} - {e}")
            continue
        pe_feats = peptide_features(row["Sequence"])
        if "Type_Phage" in df.columns:
            features.append({"Meta": row["Meta"]} | bp_feats | pe_feats | {"Type_Phage": row["Type_Phage"]})
        else:
            features.append({"Meta": row["Meta"]} | bp_feats | pe_feats)
    return DataFrame(features)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='make_input_files.log', encoding='utf-8', level=logging.DEBUG)

    logger.info(f"Processing labeled data . . .")

    df_labeled = read_csv("../data/complete_labeled_sequence.csv")

    labeled_features = generate_features(df_labeled)
    labeled_features.to_csv("../data/final_labeled_data.csv")

    logger.info(f"Processing unlabeled data . . .")

    df_unlabeled = read_csv("../data/complete_unlabeled_sequence.csv")

    unlabeled_features = generate_features(df_unlabeled)
    unlabeled_features.to_csv("../data/final_unlabeled_data.csv")
