# P6 code, explained line by line (plain language)

This goes through the actual code, line by line, in the order it runs. No jargon
without an explanation the first time it shows up.

**Scope:** the three files that ran in the real smoke test — `common/schema.py`,
`common/model.py`, `p6_baselines/generate_base.py` — get every single line explained.
`common/metrics.py` is 300+ lines, so the two functions that file actually called
(`behavior_metrics`, `knowledge_scores`) get the full line-by-line treatment, and the
rest of that file gets a shorter "what it does and why" pass at the end, since walking
through all ~15 functions line by line would be a wall of text with diminishing value.

Read the files in this order too, since that's the order the program actually uses them:
`schema.py` (defines the shape of a row) → `model.py` (produces the data that fills a
row) → `generate_base.py` (ties both together into a program you can run).

---

## 1. `common/schema.py` — "every row must have these exact columns"

```python
import json
```
This loads Python's built-in tool for reading/writing JSON (a text format for
structured data — think of it as a very strict, typeable version of a spreadsheet row).

```python
RESPONSE_FIELDS = (
    "run_id", "question_id", "fact_id", "source_dataset", "dataset_version", "split",
    "question", "reference_answer", "exposure_level",
    "model_id", "model_revision", "checkpoint_id", "method", "training_seed",
    "prompt_version", "decoding_config_id", "generation_seed", "sample_index", "is_greedy",
    "model_answer", "outcome", "grading_version", "review_status", "truncated",
    "avg_logprob", "entropy", "confidence_score", "latency_seconds",
)
```
This is just a fixed list of column names, written as a "tuple" (Python's word for a
list that can't be changed after it's created — using a tuple instead of a normal list
is a small signal to anyone reading the code that this list is meant to stay fixed).
Every single response the whole *team* generates — synthetic, real QA, base model,
fine-tuned model — gets written with exactly these columns, in this order. That's what
lets six different people's code all read each other's files without agreeing on the
format by hand every time.

```python
def make_row(**fields):
```
Defines a function called `make_row`. The `**fields` means: "accept any number of
named arguments and collect them into a dictionary called `fields`." So calling
`make_row(run_id="abc", question="capital of France?")` puts
`{"run_id": "abc", "question": "capital of France?"}` into `fields`.

```python
    """Builds one response row, filling any unspecified schema field with null."""
```
A docstring — a comment describing what the function does, placed right under its
definition. Tools and other developers can read this without opening the function body.

```python
    extra = set(fields) - set(RESPONSE_FIELDS)
```
`set(fields)` turns the names of whatever was passed in into a "set" (a collection with
no duplicates, where you can quickly check "is X in here?"). `set(RESPONSE_FIELDS)` does
the same for the fixed column list. The `-` between two sets means "everything in the
left one that is *not* in the right one." So `extra` ends up holding any field name the
caller typed that isn't one of the official columns — almost always a typo.

```python
    if extra:
        raise ValueError(f"not in the shared response schema: {sorted(extra)}")
```
An empty set is "falsy" in Python (treated as False in an `if`), so this only runs if
`extra` actually has something in it. `raise ValueError(...)` stops the program
immediately with an error message. `sorted(extra)` just puts the typo'd names in
alphabetical order so the error message is easier to read. This is the safety net: if
someone writes `make_row(qeustion="...")` (typo), the program crashes right there with a
clear message, instead of silently creating a row with a `qeustion` column that nothing
else in the team's code knows to look for.

```python
    return {k: fields.get(k) for k in RESPONSE_FIELDS}
```
This is a "dict comprehension" — a compact way of building a dictionary by looping.
Read it as: "for every column name `k` in the official list, put `fields.get(k)` in the
result under that name." `.get(k)` looks up `k` in the `fields` dictionary and returns
`None` (Python's "nothing"/"null" value) if `k` was never passed in. So if you only
supply 10 of the 28 columns, you still get back a dictionary with all 28 keys — the 18
you didn't mention are just `None`. That's the "fills any unspecified field with null"
behavior the docstring promised.

```python
def write_jsonl(rows, path):
```
A function that takes a list of row-dictionaries (`rows`) and a file location (`path`)
to save them to.

```python
    """Writes standard JSON only — raises on NaN/Infinity instead of emitting them
    (Python's json module allows those non-standard literals by default, which
    other languages' JSON parsers then reject on read)."""
```
Explains a subtlety: normal JSON has no way to represent "not a number" (`NaN`) or
"infinity." Python's JSON writer will write those words into the file anyway unless you
tell it not to — and then a different program (in Python, JavaScript, anywhere) reading
that file later will fail, because those words aren't valid JSON everywhere else.

```python
    with open(path, "w", encoding="utf-8") as f:
```
Opens the file at `path` for writing (`"w"` = write mode, which creates the file or
erases and replaces it if it already exists). `encoding="utf-8"` says how to turn text
into bytes on disk — UTF-8 is the standard choice, handles any language's characters.
`as f` names the open file `f` so the code below can use it. The `with` keyword
guarantees the file gets closed automatically afterward, even if something goes wrong
partway through.

```python
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
```
Loops over every row dictionary. `json.dumps(row, ...)` converts one Python dictionary
into a single line of JSON text. `ensure_ascii=False` allows non-English characters to be
written directly rather than escaped into `\uXXXX` codes (purely a readability choice).
`allow_nan=False` is the actual safety fix: if any value in `row` is `NaN` or `Infinity`,
this line makes Python raise an error right here instead of writing the invalid word into
the file. `+ "\n"` adds a newline so each row lands on its own line — that's what "JSONL"
(JSON Lines) means: one complete JSON object per line, as opposed to one giant JSON
document.

```python
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
```
The reverse operation. Opens the file for reading (default mode). `for line in f` reads
it one line at a time. `if line.strip()` skips any blank lines (`.strip()` removes
leading/trailing whitespace; an empty result means the line was blank). `json.loads(line)`
turns each line's text back into a Python dictionary. The square brackets around the
whole expression make this a "list comprehension" — same idea as the dict comprehension
above, but building a list. Net effect: give it a JSONL file, get back a list of
dictionaries, one per line.

---

## 2. `common/model.py` — "load the model, generate one answer, reproducibly"

```python
import importlib.metadata
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
```
Four imports. `importlib.metadata` is a built-in Python tool for asking "what version of
package X is installed?" — used later to record exactly which library versions produced
a result. `time` gives access to the clock, for timing things. `torch` is PyTorch, the
library that actually runs the neural network's math on the GPU. `AutoModelForCausalLM`
and `AutoTokenizer` are Hugging Face's loader classes: give them a model's name and they
figure out which specific model architecture and vocabulary to load.

```python
ENV_PACKAGES = ("torch", "transformers", "peft", "accelerate", "datasets")
```
A fixed list of the package names whose versions matter enough to record every time.

```python
def environment_versions():
    """Package + GPU info to record alongside every run (plan.md: "environment versions recorded")."""
    versions = {}
```
Starts an empty dictionary to fill in.

```python
    for pkg in ENV_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
```
For each package name, try to ask Python "what version of this is installed?" and store
the answer. `try`/`except` is Python's way of saying "attempt this; if it fails with this
specific kind of error, do something else instead of crashing." Here, if the package
isn't installed at all, `importlib.metadata.version` would raise
`PackageNotFoundError` — instead of crashing, we just record `None` for that package
and move on.

```python
    versions["cuda_available"] = torch.cuda.is_available()
```
CUDA is NVIDIA's software layer for running code on a GPU. `torch.cuda.is_available()`
returns `True` or `False` depending on whether PyTorch can see a working GPU right now.

```python
    if torch.cuda.is_available():
        versions["gpu_name"] = torch.cuda.get_device_name(0)
    return versions
```
If there is a GPU, also record its model name (e.g. "NVIDIA A40") — device index `0`
means "the first GPU," which is the only one this project ever asks for. Then hand back
the whole dictionary.

```python
class ModelRunner:
    """Loads one pinned checkpoint, optionally with a LoRA adapter on top."""
```
Defines a class — a blueprint for creating objects that bundle data and behavior
together. "Checkpoint" here means a specific saved version of a trained model's weights.
"LoRA adapter" is a small set of extra trained numbers that can be bolted onto a base
model to change its behavior without re-training the whole thing — relevant later when
evaluating P5's fine-tuned versions, not needed for the plain base model.

```python
    def __init__(self, model_id, revision, adapter_path=None, torch_dtype="bfloat16", device=None):
```
`__init__` is the special method Python calls automatically when you create a new
`ModelRunner`. Its arguments: `model_id` (which model, e.g.
`"Qwen/Qwen2.5-0.5B-Instruct"`), `revision` (the exact pinned version of that model),
`adapter_path` (optional — where a LoRA adapter lives on disk, if any), `torch_dtype`
(what numeric precision to use — `bfloat16` is a 16-bit number format that's faster and
uses less memory than full 32-bit precision, at a small, usually harmless cost in
exactness), and `device` (which piece of hardware to run on — CPU or GPU).

```python
        self.model_id = model_id
        self.revision = revision
        self.adapter_path = adapter_path
```
Saves those three inputs onto the object itself (`self`) so other methods of this same
object can look them up later.

```python
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
```
If a `device` was explicitly passed in, use it. Otherwise (`device` is `None`, which is
falsy), pick `"cuda"` if a GPU is available, else fall back to `"cpu"`. This is Python's
common "use A, or if A is empty/None, use B instead" pattern via `or`.

```python
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
```
A tokenizer converts text into numbers (the model only understands numbers) and back.
This line downloads/loads the exact tokenizer that matches the pinned model version.

```python
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, torch_dtype)
        ).to(self.device)
```
Loads the actual model weights for that exact pinned version. `getattr(torch, torch_dtype)`
looks up the actual PyTorch object named by the string `"bfloat16"` (i.e.
`torch.bfloat16`) — a way of turning a config file's plain text into the real thing
PyTorch expects. `.to(self.device)` moves the loaded model onto the GPU (or CPU) chosen
above — models start on the CPU by default and have to be explicitly moved.

```python
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
```
Only runs if an adapter path was given. `peft` is the library for LoRA adapters; it's
imported here (inside the `if`, not at the top of the file) so that people who never use
adapters don't need that library installed at all. `PeftModel.from_pretrained` wraps the
base model with the adapter's extra trained numbers, replacing `self.model` with the
combined version.

```python
        self.model.eval()
```
Switches the model into "evaluation mode" — some parts of a neural network (like
dropout, a technique used only during training to prevent overfitting) behave
differently during training versus actual use, and `.eval()` makes sure it's using the
"actual use" behavior. Skipping this is a classic bug that makes results slightly random
or wrong.

```python
    def _build_prompt(self, question, system_prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
A private helper method (the leading underscore in `_build_prompt` is a Python
convention meaning "internal use only, not part of the public interface"). Builds a list
of chat messages in the format instruction-tuned models expect: an optional "system"
message (background instructions) and a "user" message (the actual question). Then
`apply_chat_template` converts that structured list into the exact text format —
special tokens and all — that this specific model was trained to expect, e.g. wrapping
things in markers like `<|im_start|>user`. `tokenize=False` means "give me back text, not
number IDs yet." `add_generation_prompt=True` appends the marker that tells the model
"now it's your turn to answer." Using this instead of hand-writing the prompt format
means the code automatically works correctly for any model, since each model's template
is bundled with its own tokenizer.

```python
    def generate_one(self, question, system_prompt, max_new_tokens, do_sample,
                      seed=None, temperature=1.0, top_p=1.0):
```
The main method. Arguments: the question text, the system instructions, how many new
words/pieces to generate at most, whether to sample randomly (`do_sample=True`) or always
pick the single most likely next word (`do_sample=False`, "greedy"), an optional random
seed for reproducibility, and two settings (`temperature`, `top_p`) that only matter when
sampling — they control how random the sampling is.

```python
        if seed is not None:
            torch.manual_seed(seed)
```
If a seed was given, tell PyTorch's random number generator to start from that exact
seed. Random number generators aren't actually random — they're deterministic sequences
that just look random, started from a "seed" number. Same seed → same sequence of
"random" choices → same output, every time. This is what makes the whole pipeline
reproducible.

```python
        prompt = self._build_prompt(question, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
```
Builds the chat-formatted prompt text, then runs the tokenizer on it to turn that text
into number IDs the model can process. `return_tensors="pt"` asks for the result as a
PyTorch tensor (a multi-dimensional array — like a list of numbers, or a grid of them)
instead of a plain Python list. `.to(self.device)` moves those numbers onto the GPU,
same reasoning as moving the model there.

```python
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
```
Builds a dictionary of settings to pass to the model's built-in `.generate()` method.
`output_scores=True` asks it to also hand back the raw prediction scores at every step
(needed below to compute confidence numbers), not just the final chosen words.
`return_dict_in_generate=True` asks for a structured result object instead of just the
plain output text. `pad_token_id=...` picks a token to use for padding short sequences —
uses the model's dedicated pad token if it has one, otherwise reuses its "end of
sequence" token as a fallback (a common pattern for models that were never given a
separate pad token).

```python
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
```
Only add the sampling-randomness settings if we're actually sampling — if `do_sample` is
`False` (greedy), passing temperature/top_p would be meaningless and transformers
actually prints a warning if you do (that's the harmless warning you may have seen in the
logs).

```python
        start = time.monotonic()
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        latency = time.monotonic() - start
```
`time.monotonic()` reads a clock that only ever moves forward (unlike "wall clock" time,
it can't jump backward from a system clock adjustment) — good for measuring durations.
`torch.no_grad()` tells PyTorch "don't bother tracking how to compute gradients for
anything inside this block" — gradients are only needed during training, not when just
generating text, and skipping that tracking makes generation faster and use less memory.
`self.model.generate(**inputs, **gen_kwargs)` is the actual call that runs the model and
produces text — the `**` unpacks each dictionary's key-value pairs into separate named
arguments. Then `latency` is just "how long did that take," in seconds.

```python
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, prompt_len:]
```
`out.sequences` contains the full sequence of token IDs: the original prompt followed by
whatever the model generated. `inputs["input_ids"].shape[1]` is the length (number of
tokens) of just the prompt part. `out.sequences[0, prompt_len:]` takes the first (and
only) sequence in the batch (`[0, ...]`) and slices off everything from `prompt_len`
onward — i.e., keeps only the newly generated tokens, discarding the prompt we already
know.

```python
        eos_id = self.tokenizer.eos_token_id
        truncated = len(gen_ids) >= max_new_tokens and (eos_id is None or gen_ids[-1].item() != eos_id)
```
`eos_id` is the special "end of sequence" token ID this model uses to signal "I'm done
talking." `truncated` figures out whether the response got cut off mid-thought because it
hit the length limit, rather than naturally finishing: true only if it generated the
maximum allowed number of tokens (`len(gen_ids) >= max_new_tokens`) *and* the very last
token generated (`gen_ids[-1]`, Python's way of indexing "the last item") isn't the EOS
token (or there's no EOS token defined at all for this model).

```python
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
```
Converts the generated token IDs back into readable text. `skip_special_tokens=True`
drops any of the model's internal formatting markers from the output, leaving just the
actual words. `.strip()` trims stray leading/trailing whitespace.

```python
        logprobs, entropies = [], []
```
Two empty lists to fill in below — one number per generated token.

```python
        for step_logits, tok_id in zip(out.scores, gen_ids):
```
`out.scores` is a list with one entry per generated token — each entry is the model's
raw prediction scores ("logits," one number per possible next word in its whole
vocabulary) at that step, *before* any token was chosen. `zip(out.scores, gen_ids)` pairs
each step's scores with the token that was actually picked at that step, so the loop
processes them together.

```python
            if eos_id is not None and tok_id.item() == eos_id:
                break
```
`.item()` pulls a plain Python number out of a single-value tensor. If the token at this
step is the end-of-sequence marker, stop the loop — we don't want to count the "I'm
done" signal itself as part of the answer's confidence.

```python
            log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
```
The model's raw scores ("logits") aren't probabilities yet — they're just relative sizes.
`softmax` is the standard math operation that turns a list of raw scores into actual
probabilities that sum to 1 (bigger logit → bigger share of the probability). `log_softmax`
computes the logarithm of that directly, which is both more numerically stable (avoids
extremely tiny numbers causing rounding errors) and exactly what's needed below. `[0]`
picks the single sequence in the batch; `.float()` makes sure the math happens in full
32-bit precision even if the model itself runs in a lower-precision format, again for
numerical accuracy. `dim=-1` says "do this across the last dimension" — i.e., across the
whole vocabulary of possible next words, for each step.

```python
            logprobs.append(log_probs[tok_id].item())
```
Out of all those log-probabilities (one per possible word in the vocabulary), pick out
the one for the specific token that was actually generated at this step, and save it.

```python
            probs = log_probs.exp()
```
`.exp()` is the mathematical inverse of taking a logarithm — this converts the
log-probabilities back into plain probabilities (each between 0 and 1).

```python
            # p*log(p) -> 0 as p -> 0, but 0 * -inf is nan in floating point when a
            # logit is masked to -inf; zero those terms instead of propagating nan.
            terms = torch.nan_to_num(probs * log_probs, nan=0.0)
            entropies.append(-terms.sum().item())
```
Entropy is a standard measure of "how spread out / uncertain" a probability
distribution is — computed as `-sum(p * log(p))` across every possible word. The comment
explains a real numerical trap: mathematically, as a probability `p` shrinks toward
zero, `p * log(p)` correctly shrinks toward zero too. But computers don't do that limit —
if a word's probability computes to exactly `0.0` and its log-probability is
`-infinity` (which happens when some model settings force certain words to be
impossible), multiplying `0 * -infinity` gives `NaN` ("not a number"), a literal computing
error, not the mathematically correct answer of `0`. `torch.nan_to_num(..., nan=0.0)`
replaces any such `NaN` results with the `0.0` they were supposed to be, fixing exactly
that bug. Then it sums all the terms across the vocabulary and negates the result to get
the entropy for that one step, and saves it.

```python
        return {
            "text": text,
            "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
            "entropy": sum(entropies) / len(entropies) if entropies else None,
            "truncated": truncated,
            "latency_seconds": latency,
        }
```
Packages everything into a dictionary to hand back. `sum(logprobs) / len(logprobs)` is
just the average — but only computed `if logprobs` (i.e., if the list isn't empty);
otherwise this returns `None` rather than crashing on a divide-by-zero (an empty list
would happen if, bizarrely, the model generated nothing but an immediate EOS token).

---

## 3. `p6_baselines/generate_base.py` — the actual program you run

```python
import argparse
import json
import re
import sys
from pathlib import Path

import yaml
```
`argparse` reads command-line arguments (the `--fixture ...` style flags). `json` for
reading/printing JSON. `re` is Python's regular-expression module, for pattern-matching
text (used below to strip punctuation). `sys` gives access to program-level things like
"where to print error messages" and "what folder is this script in." `Path` is a modern,
convenient way to work with file paths. `yaml` reads `.yaml` config files.

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```
This makes Python able to find the `common` package. `Path(__file__)` is the location of
this very script; `.resolve()` turns it into a full, unambiguous path; `.parents[1]` goes
up two folder levels (the script's folder, then one more up — landing on the repo's root
folder, since this script lives in `p6_baselines/`). `sys.path` is the list of folders
Python searches when you write `import something`; inserting the repo root at the front
(`0`) means `import common` will find the `common/` folder here, regardless of what
folder you happened to run the script from.

```python
from common import schema
from common.metrics import behavior_metrics, knowledge_scores
from common.model import ModelRunner, environment_versions
```
Pulls in the pieces built earlier: the schema module, two specific metric functions, and
the model-loading class plus its version-recording function.

```python
REPO_ROOT = Path(__file__).resolve().parents[1]
```
Same calculation as above, saved once so it doesn't need repeating.

```python
IDK_PHRASES = ("i don't know", "i do not know", "unknown", "cannot determine", "no information")
```
A fixed list of phrases that count as the model "abstaining" (saying it doesn't know)
rather than actually answering.

```python
def placeholder_grade(model_answer, reference_answer):
    """Exact/alias/IDK-phrase check. Placeholder only — see module docstring."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
```
`lambda` defines a tiny, throwaway function inline and names it `norm`. Given a string
`s`, `.lower()` makes it all lowercase, then `re.sub(r"[^a-z0-9 ]", "", ...)` uses a
regular expression to delete every character that is *not* a lowercase letter, digit, or
space (the `^` inside the brackets means "not"; `re.sub` replaces matches with an empty
string, i.e. deletes them). `.strip()` trims leftover leading/trailing spaces. So
`norm("Paris.")` becomes `"paris"` — this is basic text normalization so that
capitalization and punctuation don't cause a correct answer to be marked wrong.

```python
    ans = norm(model_answer)
```
Normalizes the model's actual answer text once, up front.

```python
    if reference_answer == "__unanswerable__":
        return "abstain" if any(p in ans for p in IDK_PHRASES) else "incorrect"
```
If this question was
marked as deliberately unanswerable (that special placeholder string used by the smoke
fixture's fictional-explorer question), then the only "good" outcome is the model
admitting it doesn't know. `any(p in ans for p in IDK_PHRASES)` checks whether *any* of
the IDK phrases shows up inside the normalized answer text. If so, the outcome is
`"abstain"`; if the model guessed something instead, that's `"incorrect"` — guessing on
something truly unanswerable is exactly the failure mode this whole project studies.

```python
    if any(p in ans for p in IDK_PHRASES):
        return "abstain"
```
For every other (answerable) question: if the model said an IDK-style phrase, that's
still an abstention, regardless of whether the question actually had a knowable answer.

```python
    return "correct" if norm(reference_answer) in ans else "incorrect"
```
Otherwise, normalize the true reference answer the same way, and check whether it appears
*somewhere inside* the model's normalized answer (not necessarily an exact full match —
e.g. `"au"` inside `"the symbol is au"` still counts). If it's in there: `"correct"`,
otherwise `"incorrect"`. This is deliberately simple and explicitly labeled a placeholder
— it will call some genuinely-correct-but-oddly-phrased answers wrong, and P2's real
grader (with proper answer aliases and manual review) is meant to replace it.

```python
def load_yaml(name):
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)
```
A small helper: given a filename like `"model.yaml"`, opens
`<repo root>/configs/model.yaml` and parses it into a Python dictionary.
`yaml.safe_load` is the recommended safe way to read YAML — it refuses to execute
arbitrary code that a malicious YAML file could otherwise smuggle in.

```python
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint-id", default=None, help="defaults to '<model_id>-base'")
    ap.add_argument("--seed", type=int, default=0, help="base seed; sample i uses seed+i+1, greedy uses seed")
    args = ap.parse_args()
```
Sets up the command-line interface: which flags this script accepts
(`--fixture`, `--out`, `--run-id`, optionally `--checkpoint-id` and `--seed`), whether
each is required, and default values. `args.parse_args()` reads whatever was actually
typed on the command line and stores it in `args` — so after this, `args.fixture` holds
whatever path was passed after `--fixture`, etc.

```python
    model_cfg = load_yaml("model.yaml")
    dec_cfg = load_yaml("decoding.yaml")
    checkpoint_id = args.checkpoint_id or f"{model_cfg['model_id']}-base"
```
Loads both config files. `checkpoint_id` uses the same "use A, or fall back to B" `or`
pattern from `model.py`: if `--checkpoint-id` wasn't given on the command line, build a
default name like `"Qwen/Qwen2.5-0.5B-Instruct-base"` from the model's own id.

```python
    questions = schema.read_jsonl(args.fixture)
    print(f"Loaded {len(questions)} questions from {args.fixture}", file=sys.stderr)
    print("Environment:", json.dumps(environment_versions()), file=sys.stderr)
```
Loads the question list from whatever fixture file was specified, then prints two status
lines. `file=sys.stderr` sends these to the "error output" stream rather than normal
output — a common convention so that progress/status messages don't get mixed in with a
program's actual result data if someone redirects the output to a file.

```python
    runner = ModelRunner(model_cfg["model_id"], model_cfg["revision"], torch_dtype=model_cfg["torch_dtype"])
```
Actually loads the model into memory, using the pinned id/revision/precision read from
the config file — this is the slow step (downloading/loading multi-hundred-megabyte
weights).

```python
    rows = []
    for q in questions:
```
Starts an empty list to collect every output row, then loops over each question in the
fixture file. `q` is one question's dictionary (e.g.
`{"question_id": "smoke-001", "question": "What is the capital of France?", ...}`).

```python
        common = dict(
            run_id=args.run_id, question_id=q["question_id"], fact_id=q.get("fact_id"),
            source_dataset=q["source_dataset"], dataset_version=q["dataset_version"], split=q["split"],
            question=q["question"], reference_answer=q["reference_answer"], exposure_level=q.get("exposure_level"),
            model_id=model_cfg["model_id"], model_revision=model_cfg["revision"], checkpoint_id=checkpoint_id,
            method="base", training_seed=None,
            prompt_version=dec_cfg["prompt_version"], decoding_config_id=dec_cfg["decoding_config_id"],
        )
```
Builds a dictionary of all the schema fields that stay *the same* across every
generation for this one question (its id, the model used, etc.) — collected once here so
it doesn't need retyping for each of the 6 generations below. `q["fact_id"]` would crash
if the fixture didn't include that key at all; `q.get("fact_id")` instead just returns
`None` in that case — used for the fields that are genuinely optional depending on the
dataset (real QA has no `fact_id`).

```python
        gens = [("greedy", True, args.seed, dec_cfg["greedy"])]
        for i in range(dec_cfg["sampled"]["n_samples"]):
            gens.append((f"sample{i}", False, args.seed + i + 1, dec_cfg["sampled"]))
```
Builds a to-do list of the 6 generations to run for this question: one tuple
`("greedy", True, seed, settings)` for the single greedy response, then a loop adding one
tuple per sampled response — `range(dec_cfg["sampled"]["n_samples"])` counts from `0` up
to (not including) however many samples the decoding config says (5, per `plan.md`).
Each sample gets a distinct seed (`args.seed + i + 1` — so if the base seed is `0`, the
samples use seeds `1, 2, 3, 4, 5`) so the 5 sampled responses are different from each
other but always the same 5 responses if you rerun with the same base seed.

```python
        for sample_index, is_greedy, seed, settings in gens:
```
Loops over that to-do list, unpacking each tuple into its four named pieces.

```python
            gen = runner.generate_one(
                q["question"], dec_cfg["system_prompt"], dec_cfg["max_new_tokens"],
                do_sample=not is_greedy, seed=seed,
                temperature=settings.get("temperature", 1.0), top_p=settings.get("top_p", 1.0),
            )
```
Actually calls the model to generate one response. `do_sample=not is_greedy` — sampling
is on exactly when this generation *isn't* the greedy one. `settings.get("temperature", 1.0)`
looks up `"temperature"` in whichever settings dict applies (greedy or sampled), falling
back to `1.0` if it's not there — the greedy settings in the config file don't define a
temperature at all, since it's unused for greedy generation, so this default just avoids
a lookup error without it mattering.

```python
            outcome = placeholder_grade(gen["text"], q["reference_answer"])
```
Grades the generated text against the question's known correct answer, using the
placeholder grader defined above.

```python
            rows.append(schema.make_row(
                **common,
                generation_seed=seed, sample_index=sample_index, is_greedy=is_greedy,
                model_answer=gen["text"], outcome=outcome, grading_version="placeholder_exact_v0",
                review_status="unreviewed", truncated=gen["truncated"],
                avg_logprob=gen["avg_logprob"], entropy=gen["entropy"],
                confidence_score=gen["avg_logprob"], latency_seconds=gen["latency_seconds"],
            ))
```
Builds the finished row for this one generation and adds it to the `rows` list.
`**common` unpacks all the per-question fields built earlier back into named arguments,
combined here with the fields specific to *this particular* generation (which sample it
was, what it answered, how it graded, timing, etc.) — all passed through `schema.make_row`,
which (from part 1 above) both validates the field names and fills in nulls for anything
still missing.

```python
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema.write_jsonl(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}", file=sys.stderr)
```
After all questions and all their generations are done: builds the full output path,
makes sure its containing folder(s) exist (`parents=True` creates any missing parent
folders too; `exist_ok=True` means "don't complain if it already exists"), writes every
collected row out to that file, and prints a confirmation.

```python
    greedy_rows = [r for r in rows if r["is_greedy"]]
    print("\n=== Greedy behavior metrics ===")
    print(json.dumps(behavior_metrics([r["outcome"] for r in greedy_rows]), indent=2))
```
Filters `rows` down to just the greedy ones (one per question — this is the "official"
single answer used for the main results table, versus the 5 samples used to measure
consistency). Pulls out just their `outcome` values, hands that list to
`behavior_metrics` (part of `metrics.py`, explained below), and pretty-prints the result
(`indent=2` makes the JSON output human-readable instead of one long line).

```python
    print("\n=== Knowledge scores (sampled only) ===")
    for key, ks in sorted(knowledge_scores(rows).items()):
        print(key[2], ks)
```
`knowledge_scores(rows)` (also explained below) returns a dictionary where each key
identifies one question and each value is that question's knowledge score.
`.items()` turns that dictionary into a list of `(key, value)` pairs to loop over;
`sorted(...)` puts them in a consistent order (rather than the effectively-random order
dictionaries would otherwise print in). Each `key` here is a 3-part combination
(`run_id`, `checkpoint_id`, `question_id`) — `key[2]` picks out just the third piece
(the question id) for a cleaner printout, alongside `ks`, the actual score dictionary.

```python
if __name__ == "__main__":
    main()
```
A standard Python idiom. `__name__` is a special built-in variable that equals
`"__main__"` only when this file is run directly (like `python -m p6_baselines.generate_base`),
and equals the module's name instead when this file is *imported* by some other script.
So this line means "only actually run `main()` if this file was run directly, not if
someone just imports functions from it" — letting other code safely reuse
`placeholder_grade` or `load_yaml` without accidentally triggering a full model run.

---

## 4. `common/metrics.py` — the two functions actually used, then the rest briefly

### `behavior_metrics` (called above)

```python
def outcome_counts(outcomes):
    counts = Counter(outcomes)
```
`Counter` (from Python's `collections` module) is a dictionary specialized for
counting — give it a list, and `counts` becomes something like
`{"correct": 4, "incorrect": 1}`, automatically tallying how many times each distinct
value appeared.

```python
    unknown = set(counts) - set(OUTCOMES)
    if unknown:
        raise ValueError(f"unknown outcomes: {sorted(unknown)}")
```
Same "find anything unexpected" pattern as `schema.make_row`'s typo check: if any
outcome value isn't one of the four allowed words (`correct`/`incorrect`/`abstain`/`invalid`),
crash immediately with a clear message rather than silently miscounting a typo'd label.

```python
    c, w, a = counts[CORRECT], counts[INCORRECT], counts[ABSTAIN]
    return {"C": c, "W": w, "A": a, "N": c + w + a, "invalid": counts[INVALID]}
```
Pulls out the three counts that matter for behavior metrics (Correct, Wrong, Abstain),
and returns them along with `N` (their total — deliberately excluding invalid/technical
failures from this total, per `plan.md`'s rule that broken generations shouldn't be
silently counted as if they were real answers) and the invalid count on its own, so it's
visible rather than hidden.

```python
def behavior_metrics(outcomes):
    """Accuracy, hallucination, abstention, coverage, selective accuracy/risk."""
    k = outcome_counts(outcomes)
    c, w, a, n = k["C"], k["W"], k["A"], k["N"]
    return {
        **k,
        "accuracy": _div(c, n),
        "hallucination_rate": _div(w, n),
        "abstention_rate": _div(a, n),
        "coverage": _div(c + w, n),
        "selective_accuracy": _div(c, c + w),
        "selective_risk": _div(w, c + w),
    }
```
Gets the counts from the function above, then computes six named ratios directly from
plan.md's own definitions: accuracy = correct/total; hallucination rate = wrong/total;
abstention rate = abstained/total; coverage = how much of the time it attempted an answer
at all (correct+wrong)/total; selective accuracy = accuracy *among only the questions it
actually attempted* (correct/(correct+wrong)); selective risk = the wrong-answer version
of the same idea. `_div` (defined elsewhere in the file as `num/den if den else None`) is
a tiny safe-division helper — ordinary division crashes on divide-by-zero, this instead
returns `None` (matching plan.md's "N/A" rule, e.g. when a model abstained on
*everything* and so has no "selective" accuracy to speak of). `**k` copies all of `k`'s
own entries (C, W, A, N, invalid) into this result too, so the final dictionary has both
the raw counts and the computed rates together in one place.

### `knowledge_scores` (called above)

```python
def knowledge_scores(rows, key=("run_id", "checkpoint_id", "question_id")):
    """K(q) = correct sampled responses / sampled responses, per question.
    ...
    """
    groups = defaultdict(list)
```
`defaultdict(list)` is a dictionary that automatically creates an empty list the first
time you access a new key, instead of raising an error — convenient for grouping things
without a separate "does this key already exist?" check.

```python
    for r in rows:
        if r["is_greedy"] or r["outcome"] == INVALID:
            continue
        groups[tuple(r[k] for k in key)].append(r["outcome"] == CORRECT)
```
Loops over every row. `continue` skips straight to the next row without doing anything
else — used here to skip the single greedy response (K(q) per plan.md is defined only
over the *sampled* responses) and to skip any response that failed technically rather
than actually answering. For every remaining row, `tuple(r[k] for k in key)` builds a
grouping key out of the three fields named in the `key` argument (by default: which run,
which checkpoint, which question) — grouping this way, rather than by question id alone,
means results from different models or different runs never accidentally get mixed
together into one score. `.append(r["outcome"] == CORRECT)` adds `True` or `False` to
that group's list, depending on whether this particular sampled response was correct.

```python
    return {
        g: {"knowledge_score": sum(v) / len(v), "n_samples": len(v)}
        for g, v in groups.items()
    }
```
For every group `g` (one question) and its list `v` of `True`/`False` values: in Python,
`True` counts as `1` and `False` as `0` in arithmetic, so `sum(v)` is simply "how many
were correct," and `sum(v) / len(v)` is exactly "fraction correct" — the knowledge score.
`n_samples` (how many sampled responses actually went into that average) is recorded
alongside it, since a score based on 5 samples deserves more trust than one based on 1.

### The rest of `metrics.py`, briefly

- **`auroc` / `auprc`** — standard measures of how well a confidence number actually
  separates correct answers from wrong ones, without needing to pick any specific
  cutoff. AUROC's "ties count as half" handling and the "returns `None` when only one
  class is present" behavior mirror the same "don't fake a number that doesn't mean
  anything" philosophy as the functions above.
- **`brier_score` / `expected_calibration_error` / `reliability_bins`** — check whether a
  stated probability (like "I'm 80% sure") is actually trustworthy, by comparing stated
  confidence against the actual fraction correct, either as one overall number (Brier,
  ECE) or bucket-by-bucket (`reliability_bins`, used to draw the calibration charts
  `plan.md` asks for).
- **`apply_threshold` / `confident_hallucinations`** — model "if we made the AI abstain
  whenever its confidence drops below X" as a rule that can be applied identically to
  every method being compared, and separately count wrong-but-confident answers — the
  specific failure mode ("confident hallucination") the whole project is centered on.
- **`risk_coverage_curve` / `area_under_risk_coverage` / `risk_at_coverage`** — sweep that
  confidence cutoff across every possible value and record the coverage/risk tradeoff at
  each one, producing the curve `plan.md` calls for, plus two ways to summarize the whole
  curve as a single number.
- **`bootstrap_ci` / `paired_bootstrap_diff`** — instead of reporting just one number
  (e.g. "62% accuracy"), these repeatedly re-sample the data with replacement (sampling
  *whole questions*, not individual responses, since a question's 5 sampled answers
  aren't independent of each other) to estimate a range the true number probably falls
  in, and to test whether one method beating another is a real difference or just noise.

---

If any one function or line above is still unclear even with this, tell me which one and
I'll expand just that piece further — happy to go deeper on any single function rather
than re-explaining everything at once.
