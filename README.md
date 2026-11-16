<div align="center">

# FUSION TAPE

**An English-language PII redactor pipeline**

</div>

```
        ┌───┐
        │ F │ ilepaths
        │ U │ ╮
        │ S │ ╯ ernames
        │ I │ Ds
        │ O │ rganisation names
        │ N │ ames
        ├───┤
        │ T │ elephone
        │ A │ ddress
        │ P │ assword
        │ E │ mail
        └───┘
```

Nine kinds of personally identifying information, taped over. Text goes in; the same text comes
out with every piece of PII replaced by a marker such as `[[EMAIL]]` or `[[NAME/USERNAME]]`.
It runs locally on CPU with two small models and a set of regex rules. Nothing leaves the
machine.

```python
from fusion_tape import redact

redacted, log = redact(
    "Hi team, please reset the VPN for jane.doe92 (Jane Doe, +44 20 7946 0958). "
    "Her old password was Xk9#mLpQ2! and the report is in C:\\Users\\jdoe\\Documents\\q3.xlsx."
)
```

```
Hi team, please reset the VPN for [[USERNAME]] ([[USERNAME]] Doe, [[TELEPHONE]]). Her old
password was [[PASSWORD]] and the report is in [[FILEPATH]].
```

That is real output, and it shows the limits too: "Doe" survived. Review redacted text before
you share it (see [Things to know](#things-to-know)).

---

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

FUSION TAPE runs on CPU, so install the CPU build of PyTorch first. It is much smaller than the
default CUDA build on Linux and Windows:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

### Using it in another project

Every module sits directly in the repository folder, so the folder itself is the package. Drop
it into your project under the name `fusion_tape` (a hyphen is not allowed in an `import`):

```bash
git clone https://github.com/JackCollins91/fusion-tape.git fusion_tape
```

```
your_project/
├── your_script.py        from fusion_tape import redact
└── fusion_tape/          this repository
```

If the folder keeps the name `fusion-tape`, import it by name instead:

```python
import importlib
redact = importlib.import_module('fusion-tape').redact
```

The `models/`, `log/` and `outputs/` folders are created next to the script that calls
`redact()`, not inside the package. Alternatively, `pip install -e .` from inside the folder
makes `fusion_tape` importable from anywhere.

Then run the demo:

```bash
python demo.py --skip-benchmark
```

The first run downloads two models from the Hugging Face Hub (about 400 MB) into `models/`.
Every later run finds them on disk and needs no network.

| Model | Size | Licence | Context |
|---|---|---|---|
| [`gravitee-io/bert-small-pii-detection`](https://huggingface.co/gravitee-io/bert-small-pii-detection) | BERT-small, ~29 M | Apache-2.0 | 512 tokens |
| [`kalyan-ks/ettin-68m-nemotron-pii`](https://huggingface.co/kalyan-ks/ettin-68m-nemotron-pii) | Ettin, 68 M | MIT | ~8k tokens |

To check the install without downloading anything:

```bash
python -m pytest tests -q
```

---

## Usage

There is one function.

```python
redacted, log = redact(text, log=False, skip=None)
```

| Parameter | Type | Meaning |
|---|---|---|
| `text` | `str` or `list[str]` | One text, or a list of texts that are processed together in batches. |
| `log` | `bool`, default `False` | Also return a per-text log. |
| `skip` | `str` or `list[str]`, default `None` | Domain pipes to leave out, named by keyword (see [Skipping domains](#skipping-domains)). |

| Returned | Type | Meaning |
|---|---|---|
| `redacted` | `str` or `list[str]` | The input with PII replaced by markers. A string in gives a string out; a list in gives a list out, in the same order. Nothing outside a censored run changes. |
| `log` | `None` or `list[tuple]` | `None` when `log=False`. Otherwise one tuple per input text. |

Each log tuple holds three things:

1. the input text,
2. a dict of every redacted string and the marker that replaced it,
3. the time it took to redact that text across all nine domains, in milliseconds.

```python
from fusion_tape import redact

texts = [
    "Please reset the VPN account for marcus_oyelaran.",
    "Send the invoice to accounts@brightwater-analytics.co.uk.",
]
redacted, log = redact(texts, log=True)

for original, found, milliseconds in log:
    print(found, milliseconds)
```

```
{'marcus_oyelaran': '[[NAME/USERNAME]]'} 39.143
{'accounts@brightwater-analytics.co.uk': '[[EMAIL]]'} 42.861
```

### Skipping domains

`skip` leaves whole domain pipes out of the pipeline. The keywords are the words of the acronym,
in any letter case; anything else raises `ValueError`.

| Keyword | Pipe left out | | Keyword | Pipe left out |
|---|---|---|---|---|
| `filepaths` | FILEPATH | | `telephone` | TELEPHONE |
| `usernames` | USERNAME | | `address` | ADDRESS |
| `ids` | ID | | `password` | PASSWORD |
| `organisations` | ORGANISATION | | `email` | EMAIL |
| `names` | NAME | | | |

```python
redact(text, skip='email')                    # keep e-mail addresses readable
redact(text, skip=['names', 'usernames'])     # keep people's names readable
```

Skipping a pipe stops *that pipe* from censoring. Another pipe can still cover the same string:
the USERNAME pipe also reads name labels, so to keep names readable skip both `names` and
`usernames`. When no remaining pipe needs a text view or the models, they are not run, so
skipping also saves time.

**Markers.** The nine tags are `NAME`, `USERNAME`, `EMAIL`, `TELEPHONE`, `ADDRESS`,
`ORGANISATION`, `ID`, `PASSWORD` and `FILEPATH`. When more than one domain flags the same
string, the tags share one marker, in that order: `[[NAME/USERNAME]]`.

**Timing.** Texts in a list share model batches, so a text's time is its own chunking and rule
time plus its share of each batch, split by token count. The times add up to the real total.

**The log holds raw PII.** `redact()` only returns it. Where it is stored is up to you.

`demo.py` is a complete example: it redacts synthetic PII from every domain with `log=True`,
writes the redacted texts to `outputs/demo_redacted.txt` and the log to `log/demo_log.json`.
Run without `--skip-benchmark`, it then runs the benchmarks described below.

---

## Benchmarks

```bash
python demo.py
```

After the example, `demo.py` runs the regression test. Everything it needs is inside the script.

1. It downloads the public [`gretelai/gretel-pii-masking-en-v1`](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1)
   dataset (Apache-2.0) from the Hugging Face Hub into the git-ignored `benchmarks/` folder. No
   account or token is needed, and nothing from it is committed.
2. It draws 1,000 texts from the test and validation splits with a fixed seed, so every run
   scores the same texts.
3. It redacts them with `redact()`, without logging.
4. It scores the redacted text alone and writes [`.Benchmark-Report.md`](.Benchmark-Report.md):
   the percentage of PII leaked, the leaks by domain, and the speed in seconds per 1,000 words.

A PII value counts as leaked when three or more of its letters or digits are still readable. The
marker type does not matter: a name covered by `[[USERNAME]]` is not a leak. The report is
committed, so a change that makes the pipeline leak more shows up as a diff.

The first benchmark run downloads about 4 MB; later runs reuse `benchmarks/`. The run takes
about a minute on four CPU cores. Gretel does not annotate file paths, so the FILEPATH domain
is not measured.

---

## Settings

Everything configurable lives in [`settings.yaml`](settings.yaml), which is read on the first
`redact()` call.

| Section | What it sets |
|---|---|
| `paths` | The `models`, `log` and `outputs` directories. Left as `null`, each is created next to the running script on first run (or in the working directory when there is no script, such as a notebook). |
| `models` | The Hub repository and local folder name of each model. A model is downloaded only if its folder is not already in the models directory. |
| `runtime` | `device` (`cpu`), `batch_size`, `num_threads`. |
| `chunking` | The chunk window in characters, the sentence overlap, and the token margin. |
| `thresholds` | One classification threshold per model-backed domain. |

To use a different file, set the `FUSION_TAPE_SETTINGS` environment variable to its path, or
put a `settings.yaml` next to your script.

The `log` directory receives `fusion_tape.log`, an operational run log: settings in use, model
downloads and load times, chunk counts and timings. It never contains input text.
`models/`, `outputs/`, `log/` and `benchmarks/` are git-ignored.

### Thresholds

For every token, each model gives a probability for every domain: the summed probability of all
that domain's labels. A token is flagged for a domain when

```
the model's top label belongs to the domain     OR     P(domain) >= threshold
```

The first half is the rule the pipeline was benchmarked with. The threshold can only add to it,
so the pipeline never censors less than it did when it was evaluated. Lower a value to censor
more in that domain; raise it towards 1.0 to fall back to the benchmarked rule alone.

```yaml
thresholds:
  name: 0.5
  username: 0.5
  telephone: 0.5
  address: 0.5
  organisation: 0.5
  id: 0.5
  password: 0.5
```

`EMAIL` and `FILEPATH` are regex-only and have no threshold. The 0.5 defaults are starting
values, not benchmarked ones: the benchmarks were run with the top-label rule only.

---

## The pipeline

```
 text(s)
    │
    ▼
 1. CHUNK      sentence windows that overlap; over-long runs tiled twice
    │
    ▼
 2. VIEWS      raw · username-normalised · drop-first-letter
    │
    ▼
 3. SCORE      each model runs ONCE per unique string, in batches
    │          → a probability per token per domain
    ▼
 4. PIPES      nine domain pipes read those probabilities + regex rules
    │
    ▼
 5. MERGE      spans → text offsets, overlaps merged, markers written
    │
    ▼
 redacted text(s)  (+ log)
```

**1. Chunk.** Long text is cut to fit the smallest model context (512 tokens, less a margin).
Sentences are packed into windows that overlap by two sentences, so PII near a boundary is seen
whole by a neighbouring chunk. A run with no usable sentence breaks is tiled twice, with the
second tiling's cut points halfway between the first's: anything cut by one tiling is whole in
the other. Any chunk still over the token budget is re-split until it fits.

**2. Views.** Names hide inside usernames, so each chunk is read three ways:

| View | What it does | Example |
|---|---|---|
| raw | the chunk as written | |
| normalised | splits username-shaped tokens | `john_smith` → `john smith`, `JSmith` → `J Smith` |
| drop-first | drops the first letter of words that are not ordinary vocabulary; a hit flags the whole word | `jsmith92` → `Smith` |

**3. Score.** Chunks from every input text are pooled, duplicates removed, sorted by length and
run through each model in batches. A model sees each string exactly once. That single pass
labels every token for all domains at once; no domain triggers a model run of its own.

**4. Pipes.** Each domain has its own pipe: the detector combination that reached the highest
recall for that domain when the pipeline was developed. Detectors within a pipe are OR-ed, so one hit is enough.

| | Domain | Pipe | Threshold |
|---|---|---|---|
| **F** | Filepaths | regex: Windows, UNC, POSIX, home, env-var and relative paths, `file://` | none |
| **US** | Usernames | username labels · name labels on the normalised and drop-first views · handle and `first.last` rules | `username` |
| **I** | IDs | both models · SSN, IBAN, card, IP, MAC, VIN, segmented-ID and crypto-wallet rules | `id` |
| **O** | Organisation names | both models · legal-suffix and partnership rules | `organisation` |
| **N** | Names | both models on all three views · honorific and greeting rules | `name` |
| **T** | Telephone | both models · worldwide phone regex with structural checks | `telephone` |
| **A** | Address | both models · street-suffix and secondary-unit rules | `address` |
| **P** | Password | both models · symbol, random-token and secret-key rules | `password` |
| **E** | Email | regex | none |

**5. Merge.** Spans are mapped back to offsets in the original text. Spans with no letters or
digits are dropped, overlapping or touching spans are merged, and each merged run is replaced by one marker
naming every domain that flagged it. The output is rebuilt from those spans rather than from
anything a model wrote, so every other character is returned exactly as it came in.

---

## Things to know

* **It over-censors and it can leak.** The pipes were chosen for recall, so ordinary words are
  sometimes censored, and a small share of PII still gets through. The current leak rate is in
  [`.Benchmark-Report.md`](.Benchmark-Report.md).
* **Names found by the models are marked `[[NAME/USERNAME]]`.** The username pipe reads the name
  labels too, which is how it catches usernames built from names.
* **Code and tracebacks are hit hard.** Identifiers such as `sheet_name` or `warnings.warn` are
  username-shaped and get censored; some timestamps trip the telephone and password rules.
* **A marker names every pipe that fired.** A Windows path can come out as
  `[[USERNAME/ADDRESS/ID/FILEPATH]]` because other pipes flagged parts of it too.
* **English only.** The models, rules and benchmarks are all English.

## Project layout

Everything is on one level; the folder is the package.

```
__init__.py            exports redact()
settings.yaml          directories, models, runtime, chunking, thresholds
settings.py            loads settings.yaml, resolves the directories
engine.py              chunk → score once → pipes → merge, batching, timing, run log
scorer.py              model download/load, one-pass token scoring per domain
pipes.py               the nine domain pipes and the threshold rule
chunking.py            four-pass chunker
normalise.py           the normalised and drop-first views
rules.py               regex detectors
spans.py               markers, skip keywords, span merging, rendering
demo.py                example run with logging → outputs/ and log/, then the benchmark run
.Benchmark-Report.md   latest benchmark result (written by demo.py, committed)
requirements.txt
pyproject.toml         optional: `pip install -e .`
tests/                 offline tests (no model download): python -m pytest tests -q
```
