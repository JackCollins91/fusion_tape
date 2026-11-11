"""
The two token-classification models and the single scoring pass every domain pipe reads.

  bert_small  gravitee-io/bert-small-pii-detection   BERT-small ~29 M, Apache-2.0, 512-token context
  ettin       kalyan-ks/ettin-68m-nemotron-pii       Ettin 68 M, MIT, ~8k-token context

A model is run ONCE per piece of text.  That one forward pass yields, for every token,
a probability for every PII domain (the summed softmax mass of the domain's labels)
plus the model's top label.  ``TokenScores`` holds exactly that, and all nine domain
pipes take their decisions from it — no pipe ever triggers its own forward pass.

Weights are downloaded from the Hugging Face Hub the first time only, saved under the
models directory from ``settings.yaml``, and loaded from disk on every later run.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import ModelSpec

log = logging.getLogger('fusion_tape')

# ── Which model labels belong to which domain ─────────────────────────────────
# Each list is the union of several label vocabularies (ai4privacy, nym, Presidio,
# Nemotron-PII); it is intersected with a model's real labels when the model loads.
_ID_LABELS = [
    # ai4privacy
    'ACCOUNTNUMBER', 'CREDITCARDNUMBER', 'CREDITCARDCVV', 'SSN', 'IBAN', 'BIC', 'VEHICLEVIN',
    'VEHICLEVRM', 'BITCOINADDRESS', 'ETHEREUMADDRESS', 'LITECOINADDRESS', 'MASKEDNUMBER',
    'PHONEIMEI', 'IP', 'IPV4', 'IPV6', 'MAC',
    # nym
    'ACCOUNT_NUMBER', 'API_KEY', 'CREDIT_DEBIT_CARD', 'CUSTOMER_ID', 'CVV', 'DRIVERS_LICENSE',
    'EMPLOYEE_ID', 'GOVERNMENT_ID', 'LICENSE_PLATE', 'MAC_ADDRESS', 'MEDICAL_RECORD_NUMBER',
    'PASSPORT', 'ROUTING_NUMBER', 'SWIFT_BIC', 'TAX_ID',
    # Presidio
    'CREDIT_CARD', 'IBAN_CODE', 'FINANCIAL', 'US_BANK_NUMBER', 'US_SSN', 'US_DRIVER_LICENSE',
    'US_PASSPORT', 'US_ITIN', 'US_LICENSE_PLATE', 'IP_ADDRESS', 'IMEI',
    # Nemotron-PII
    'account_number', 'ssn', 'national_id', 'credit_debit_card', 'cvv', 'tax_id', 'employee_id',
    'customer_id', 'certificate_license_number', 'health_plan_beneficiary_number',
    'medical_record_number', 'unique_id', 'bank_routing_number', 'device_identifier', 'api_key',
    'mac_address', 'ipv4', 'ipv6', 'swift_bic', 'vehicle_identifier', 'license_plate',
]

DOMAIN_LABELS: dict[str, list[str]] = {
    'name': ['FIRSTNAME', 'LASTNAME', 'MIDDLENAME', 'PREFIX', 'GIVEN_NAME', 'SURNAME', 'PERSON',
             'HONORIFIC', 'first_name', 'last_name'],
    'username': ['USERNAME', 'user_name', 'username'],
    'telephone': ['PHONENUMBER', 'PHONE', 'PHONE_NUMBER', 'FAX_NUMBER', 'phone_number', 'fax_number'],
    'address': ['STREET', 'Street', 'STREETADDRESS', 'Streetaddress', 'Street Address',
                'BUILDINGNUMBER', 'Buildingnumber', 'Building Number', 'SECONDARYADDRESS',
                'Secondaryaddress', 'Secondary Address', 'STREET_ADDRESS', 'STREET_NAME',
                'BUILDING_NUMBER', 'SECONDARY_ADDRESS', 'LOCATION', 'street_address'],
    'organisation': ['COMPANYNAME', 'COMPANY_NAME', 'ORGANIZATION', 'company_name'],
    'id': _ID_LABELS,
    'password': ['PASSWORD', 'Password', 'password'],
}
MODEL_DOMAINS: tuple[str, ...] = tuple(DOMAIN_LABELS)


@dataclass
class TokenScores:
    """What one model says about one piece of text — one row per (non-special) token."""
    starts: np.ndarray         # [T] character offset where the token starts
    ends: np.ndarray           # [T] character offset where it ends
    top_type: np.ndarray       # [T] id of the top label's entity type (B-/I- stripped); -1 = 'O'
    top_begin: np.ndarray      # [T] True when the top label is a B- tag
    top_domain: np.ndarray     # [T] index into MODEL_DOMAINS of the top label's domain; -1 = none
    domain_prob: np.ndarray    # [T, D] summed probability of each domain's labels

    def __len__(self) -> int:
        return len(self.starts)


def _split_label(label: str) -> tuple[str, bool]:
    """'B-first_name' → ('first_name', True);  'PERSON' → ('PERSON', False)."""
    if label[:2] in ('B-', 'I-', 'E-', 'S-'):
        return label[2:], label[:2] == 'B-'
    return label, False


class PIIModel:
    """One loaded model: tokenizer, weights and its label → domain mapping."""

    def __init__(self, key: str, tokenizer, model, device: str = 'cpu'):
        import torch
        self.key = key
        self.tokenizer = tokenizer
        # float32 always: the Ettin checkpoint is stored as bfloat16, which is slow on CPU and
        # coarse enough (~3e-2 on a probability) that padding a batch could flip a label.
        self.model = model.float().to(device).eval()
        self.device = device
        self._torch = torch

        id2label = {int(i): lbl for i, lbl in model.config.id2label.items()}
        n_labels = len(id2label)
        self.entity_types: list[str] = sorted({_split_label(lbl)[0] for lbl in id2label.values()}
                                              - {'O', 'PAD'})
        type_id = {t: i for i, t in enumerate(self.entity_types)}
        domain_of_type = {t: d for d, domain in enumerate(MODEL_DOMAINS)
                          for t in DOMAIN_LABELS[domain]}

        self._label_type = np.full(n_labels, -1, dtype=np.int32)
        self._label_begin = np.zeros(n_labels, dtype=bool)
        self._label_domain = np.full(n_labels, -1, dtype=np.int32)
        label_to_domain = np.zeros((n_labels, len(MODEL_DOMAINS)), dtype=np.float32)
        for i, lbl in id2label.items():
            etype, begin = _split_label(lbl)
            if etype in type_id:
                self._label_type[i] = type_id[etype]
                self._label_begin[i] = begin
                d = domain_of_type.get(etype, -1)
                self._label_domain[i] = d
                if d >= 0:
                    label_to_domain[i, d] = 1.0
        self._label_to_domain = torch.from_numpy(label_to_domain).to(device)
        self.domains: set[str] = {MODEL_DOMAINS[d] for d in set(self._label_domain.tolist()) if d >= 0}
        self.context_limit = self._context_limit()

    def _context_limit(self) -> int:
        limits = [getattr(self.model.config, 'max_position_embeddings', None)]
        mml = getattr(self.tokenizer, 'model_max_length', None)
        if mml and mml < 1_000_000:                 # unset tokenizers report ~1e30
            limits.append(mml)
        limits = [int(l) for l in limits if l]
        return min(limits) if limits else 512

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False,
                                  verbose=False)['input_ids'])

    def is_single_piece(self, word: str) -> bool:
        """An ordinary vocabulary word (WordPiece: 'login'; BPE: ' login')."""
        return len(self.tokenizer.tokenize(word)) == 1 or len(self.tokenizer.tokenize(' ' + word)) == 1

    def score(self, texts: list[str], batch_size: int = 16) -> tuple[list[TokenScores], list[float]]:
        """
        One forward pass per text, batched.  Returns the scores in input order and
        the seconds spent on each text (a batch's time is shared by token count).
        """
        torch = self._torch
        results: list[TokenScores | None] = [None] * len(texts)
        seconds = [0.0] * len(texts)
        lengths = [self.count_tokens(t) for t in texts]
        order = sorted(range(len(texts)), key=lengths.__getitem__)      # similar lengths → little padding
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            t0 = time.perf_counter()
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.context_limit, return_offsets_mapping=True,
                                 return_special_tokens_mask=True, return_tensors='pt')
            offsets = enc.pop('offset_mapping').numpy()
            special = enc.pop('special_tokens_mask').numpy().astype(bool)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
                top = probs.argmax(dim=-1).cpu().numpy()
                domain_prob = (probs @ self._label_to_domain).cpu().numpy()
            for row, i in enumerate(idx):
                keep = ~special[row]
                top_row = top[row][keep]
                results[i] = TokenScores(
                    starts=offsets[row][keep, 0], ends=offsets[row][keep, 1],
                    top_type=self._label_type[top_row], top_begin=self._label_begin[top_row],
                    top_domain=self._label_domain[top_row], domain_prob=domain_prob[row][keep])
            elapsed = time.perf_counter() - t0
            total = sum(lengths[i] for i in idx) or 1
            for i in idx:
                seconds[i] = elapsed * lengths[i] / total
        return results, seconds          # type: ignore[return-value]


# ── Download once, then load from disk ────────────────────────────────────────
def is_downloaded(folder: Path) -> bool:
    return (folder / 'config.json').is_file() and any(
        (folder / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin'))


def ensure_downloaded(spec: ModelSpec, models_dir: Path) -> Path:
    """Return the model's folder, downloading it from the Hub only if it is not there."""
    folder = models_dir / spec.folder
    if is_downloaded(folder):
        return folder
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    log.info('model %s: not on disk, downloading %s', spec.key, spec.repo)
    tmp = models_dir / (spec.folder + '.partial')       # never leave a half-written model folder
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision).save_pretrained(str(tmp))
    AutoModelForTokenClassification.from_pretrained(spec.repo, revision=spec.revision) \
        .save_pretrained(str(tmp))
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)
    log.info('model %s: saved to %s', spec.key, folder)
    return folder


def load_model(spec: ModelSpec, models_dir: Path, device: str = 'cpu') -> PIIModel:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    folder = ensure_downloaded(spec, models_dir)
    t0 = time.perf_counter()
    model = PIIModel(spec.key, AutoTokenizer.from_pretrained(str(folder)),
                     AutoModelForTokenClassification.from_pretrained(str(folder)), device)
    log.info('model %s: loaded from %s in %.1fs (context %d tokens, domains: %s)', spec.key, folder,
             time.perf_counter() - t0, model.context_limit, ', '.join(sorted(model.domains)))
    return model
