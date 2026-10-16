"""EXP-20261009-001: NVIDIA prompt-task-and-complexity-classifier (@fea11215), model-card formula.
``build_nvidia`` is copied verbatim from EXP-20261002-001 code/remote_score_routers.py (lines 87-148), which
scored this checkpoint on newsrv CPU with run-to-run max abs difference 0. CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def build_nvidia(ckpt: Path, deberta_cfg: Path):
    import torch.nn as nn
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = json.loads((ckpt / "config.json").read_text())

    class MeanPooling(nn.Module):
        def forward(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    class MulticlassHead(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    class CustomModel(nn.Module):
        def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
            super().__init__()
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(str(deberta_cfg)))
            self.target_sizes = list(target_sizes.values())
            self.task_type_map, self.weights_map, self.divisor_map = task_type_map, weights_map, divisor_map
            self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes]
            for i, head in enumerate(self.heads):
                self.add_module(f"head_{i}", head)
            self.pool = MeanPooling()

        def compute(self, preds, target, decimal=4):
            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            scores = np.sum(np.array(preds.detach().cpu()) * weights, axis=1) / self.divisor_map[target]
            scores = [round(float(v), decimal) for v in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

        def forward(self, batch):
            outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"])
            logits = [self.heads[k](pooled) for k in range(len(self.target_sizes))]
            r = {"creativity_scope": self.compute(logits[1], "creativity_scope"),
                 "reasoning": self.compute(logits[2], "reasoning"),
                 "contextual_knowledge": self.compute(logits[3], "contextual_knowledge"),
                 "number_of_few_shots": self.compute(logits[4], "number_of_few_shots"),
                 "domain_knowledge": self.compute(logits[5], "domain_knowledge"),
                 "constraint_ct": self.compute(logits[7], "constraint_ct")}
            r["prompt_complexity_score"] = [round(0.35 * c + 0.25 * rs + 0.15 * ct + 0.15 * dk + 0.05 * ck + 0.05 * fs, 5)
                                            for c, rs, ct, dk, ck, fs in zip(r["creativity_scope"], r["reasoning"], r["constraint_ct"],
                                                                             r["domain_knowledge"], r["contextual_knowledge"], r["number_of_few_shots"])]
            return r

    model = CustomModel(cfg["target_sizes"], cfg["task_type_map"], cfg["weights_map"], cfg["divisor_map"])
    sd = load_file(str(ckpt / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    return model, tok
