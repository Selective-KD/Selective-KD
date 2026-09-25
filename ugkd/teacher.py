import torch

from .data.standardize import Standardizer
from .models import audio as audio_models, cifar as cifar_models, jets as jets_models
from . import posterior


def build_model(bundle):
    m, k, inp = bundle["modality"], bundle["num_classes"], bundle["input"]
    if m == "jets":
        return jets_models.build(bundle["model"], inp["in_dim"], k)
    if m == "cifar":
        return cifar_models.build(bundle["model"], k, dropout=inp.get("dropout", 0.0))
    if m == "audio":
        return audio_models.build(bundle["model"], k, inp["n_mels"], inp["n_frames"], dropout=inp.get("dropout", 0.0))
    raise ValueError("unknown modality %r" % m)


class Teacher:
    def __init__(self, bundle, device):
        self.bundle, self.device = bundle, device
        self.modality, self.num_classes = bundle["modality"], bundle["num_classes"]
        self._model = build_model(bundle)
        self._model.load_state_dict(bundle["state_dict"])
        self._model.to(device).eval()
        self._posterior = posterior.load(bundle["posterior"], self._model)
        self.standardizer = (Standardizer.from_state_dict(bundle["standardizer"])
                             if bundle.get("standardizer") else None)

    @torch.no_grad()
    def draws(self, x, n_draws, seed, temperature=1.0):
        return self._posterior.draws(self._model.features(x.to(self.device)), n_draws, seed, temperature)
