"""Answer grading (grading_version alias-match-v0), provisional until the shared grader is merged.

Outcomes: correct / incorrect / abstain / invalid (empty output). A hedged but substantive wrong
claim ("I'm not sure, but I think it's Paris") is incorrect, not an abstention (plan.md §5). Every
row records `match_type` so graded samples can be audited by rule.
"""

import re
import string

from . import config

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)
_ABSTAIN = re.compile(
    r"\b(i\s*(do\s*not|don'?t|dont)\s*know"
    r"|i\s*(am|'?m)\s*not\s*(sure|certain|aware)"
    r"|not\s*sure"
    r"|no\s*idea"
    r"|(cannot|can'?t|unable\s*to)\s*(answer|determine|provide|find|say)"
    r"|i\s*(do\s*not|don'?t)\s*have\s*(enough\s*)?(information|data|access))\b"
)
_ABSTAIN_WHOLE_ANSWERS = {"unknown", "not known", "unsure"}
# If one of these accompanies the abstention phrase, the model went on to commit to an answer.
_HEDGED_CLAIM = re.compile(r"\b(but|however|i\s*think|i\s*believe|probably|maybe|perhaps|likely|might\s*be|could\s*be)\b")
# Containment credit only for short answers, so a list of many candidates cannot collect it.
_MAX_CONTAINMENT_TOKENS = 12


def normalize_answer(text):
    """SQuAD-style: lowercase, strip punctuation and articles, collapse whitespace."""
    text = (text or "").lower().strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text)
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def is_abstention(text):
    if normalize_answer(text) in _ABSTAIN_WHOLE_ANSWERS:
        return True
    raw = (text or "").lower().replace("’", "'")
    return bool(_ABSTAIN.search(raw)) and not _HEDGED_CLAIM.search(raw)


def grade_answer(answer, aliases, question=None):
    """Return (outcome, match_type)."""
    norm = normalize_answer(answer)
    if not norm:
        return "invalid", "empty"
    norm_aliases = [a for a in (normalize_answer(x) for x in aliases) if a]
    if norm in norm_aliases:
        return "correct", "exact"
    # Alias contained in a short answer ("Paris, France" for "Paris"). Aliases already present in the
    # question are ignored, so restating the question ("Fraxinus excelsior is known as ...") earns nothing.
    if len(norm.split()) <= _MAX_CONTAINMENT_TOKENS:
        padded = f" {norm} "
        question_padded = f" {normalize_answer(question)} " if question else ""
        if any(f" {a} " in padded and f" {a} " not in question_padded for a in norm_aliases):
            return "correct", "contains"
    if is_abstention(answer):
        return "abstain", "abstain_pattern"
    return "incorrect", "no_match"


def grade_responses(rows):
    """Add outcome / match_type / grading_version to each row in place and return the rows."""
    for row in rows:
        aliases = row.get("aliases") or [row["reference_answer"]]
        row["outcome"], row["match_type"] = grade_answer(row["model_answer"], aliases, row.get("question"))
        row["grading_version"] = config.GRADING_VERSION
        row.setdefault("review_status", "auto")
    return rows
