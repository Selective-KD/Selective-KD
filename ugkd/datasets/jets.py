import os
import tarfile

import numpy as np

TREE = "tree"
JET_BRANCHES = ["jet_pt", "jet_eta", "jet_nparticles", "jet_sdmass",
                "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4"]
PART_BRANCHES = ["part_px", "part_py", "part_energy", "part_deta", "part_dphi",
                 "part_d0val", "part_d0err", "part_dzval", "part_dzerr", "part_charge",
                 "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
                 "part_isElectron", "part_isMuon"]
DERIVED = ["n_charged", "ptD", "girth",
           "efrac_chargedhad", "efrac_neutralhad", "efrac_photon", "n_lepton", "lepton_efrac",
           "n_ip_gt2", "n_ip_gt3", "log1p_ip_sig_max", "log1p_ip_sig_2nd", "log1p_ip_sig_3rd",
           "log1p_ip_sig_sum", "log1p_dz_sig_max", "ip_frac_gt2"]
FEATURES = JET_BRANCHES + DERIVED
LABEL_BRANCHES = ["label_QCD", "label_Hbb", "label_Hcc", "label_Hgg", "label_H4q",
                  "label_Hqql", "label_Zqq", "label_Wqq", "label_Tbqq", "label_Tbl"]


def features_from_root(path):
    import awkward as ak
    import uproot

    t = uproot.open(path)[TREE]
    jet = t.arrays(JET_BRANCHES + LABEL_BRANCHES, library="np")
    a = t.arrays(PART_BRANCHES, library="ak")

    onehot = np.stack([jet[b].astype(np.int64) for b in LABEL_BRANCHES], axis=1)
    if not np.all(onehot.sum(1) == 1):
        raise ValueError("%s: jets that are not one-hot over the label branches" % os.path.basename(path))
    y = onehot.argmax(1).astype(np.int64)
    base = np.stack([jet[f] for f in JET_BRANCHES], axis=1).astype(np.float64)

    ppt = np.sqrt(a["part_px"] ** 2 + a["part_py"] ** 2)
    dr = np.sqrt(a["part_deta"] ** 2 + a["part_dphi"] ** 2)
    sum_pt = np.maximum(ak.to_numpy(ak.sum(ppt, axis=1)), 1e-6)
    e_jet = np.maximum(ak.to_numpy(ak.sum(a["part_energy"], axis=1)), 1e-6)
    charged = a["part_charge"] != 0
    n_charged = ak.to_numpy(ak.sum(charged, axis=1)).astype(np.float64)
    d0sig = ak.where(charged, np.abs(a["part_d0val"]) / np.maximum(a["part_d0err"], 1e-6), 0.0)
    dzsig = ak.where(charged, np.abs(a["part_dzval"]) / np.maximum(a["part_dzerr"], 1e-6), 0.0)
    ranked = ak.sort(d0sig, axis=1, ascending=False)

    def nth(arr, i):
        return ak.to_numpy(ak.fill_none(ak.firsts(arr[:, i:]), 0.0)).astype(np.float64)

    def efrac(flag):
        return ak.to_numpy(ak.sum(a["part_energy"] * a[flag], axis=1)) / e_jet

    n_ip2 = ak.to_numpy(ak.sum(d0sig > 2, axis=1)).astype(np.float64)
    lepton = a["part_isElectron"] + a["part_isMuon"]
    derived = [
        n_charged,
        ak.to_numpy(np.sqrt(ak.sum(ppt ** 2, axis=1))) / sum_pt,
        ak.to_numpy(ak.sum(ppt * dr, axis=1)) / sum_pt,
        efrac("part_isChargedHadron"), efrac("part_isNeutralHadron"), efrac("part_isPhoton"),
        ak.to_numpy(ak.sum(lepton, axis=1)).astype(np.float64),
        ak.to_numpy(ak.sum(a["part_energy"] * lepton, axis=1)) / e_jet,
        n_ip2,
        ak.to_numpy(ak.sum(d0sig > 3, axis=1)).astype(np.float64),
        np.log1p(nth(ranked, 0)), np.log1p(nth(ranked, 1)), np.log1p(nth(ranked, 2)),
        np.log1p(ak.to_numpy(ak.sum(d0sig, axis=1))),
        np.log1p(ak.to_numpy(ak.max(dzsig, axis=1, initial=0.0))),
        n_ip2 / np.maximum(n_charged, 1.0),
    ]
    x = np.concatenate([base, np.stack(derived, axis=1)], axis=1)
    if not np.isfinite(x).all():
        raise ValueError("%s: non-finite feature values" % os.path.basename(path))
    return x.astype(np.float32), y


def extract(source, max_per_class=None, log=print):
    xs, ys, seen = [], [], {}

    def add(path, name):
        x, y = features_from_root(path)
        if max_per_class is not None:
            keep = np.zeros(len(y), dtype=bool)
            for c in np.unique(y):
                room = max_per_class - seen.get(int(c), 0)
                idx = np.flatnonzero(y == c)[:max(room, 0)]
                keep[idx] = True
                seen[int(c)] = seen.get(int(c), 0) + len(idx)
            x, y = x[keep], y[keep]
        xs.append(x)
        ys.append(y)
        log("  %s  %d jets" % (name, len(y)))

    if os.path.isdir(source):
        for name in sorted(os.listdir(source)):
            if name.endswith(".root"):
                add(os.path.join(source, name), name)
    else:
        import tempfile
        with tarfile.open(source) as tar, tempfile.TemporaryDirectory() as tmp:
            for m in tar.getmembers():
                if m.name.endswith(".root"):
                    tar.extract(m, path=tmp)
                    add(os.path.join(tmp, m.name), os.path.basename(m.name))
                    os.remove(os.path.join(tmp, m.name))
    return np.concatenate(xs), np.concatenate(ys)
