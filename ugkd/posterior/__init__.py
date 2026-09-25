from .laplace import LastLayerLaplace

KINDS = ("exact", "kfac", "library-full", "library-kron")


def fit(model, batches, kind, tau, device="cpu"):
    if kind.startswith("library-"):
        from .library import LibraryLaplace
        return LibraryLaplace.fit(model, batches, kind[len("library-"):], tau, device=device)
    return LastLayerLaplace.fit(model, batches, kind, tau, device=device)


def load(d, model):
    if d.get("backend") == "library":
        from .library import LibraryLaplace
        return LibraryLaplace.from_state_dict(d, model)
    return LastLayerLaplace.from_state_dict(d)
