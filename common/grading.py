"""Shared answer grader (plan.md section 2: P2 maintains shared grading). Standard library only.

Replaces common.evaluate.placeholder_grade for any result that is reported as final.

Contract with the rest of the repo
----------------------------------
* outcome is one of the shared metrics.py values: "correct" | "incorrect" | "abstain" | "invalid".
* Extra information that does not fit the schema goes in `review_status`
  ("unreviewed" or "flagged:<reason>"), because schema.make_row rejects unknown fields.
* Every graded row must carry grading_version = GRADING_VERSION. Bump it whenever behaviour changes.
"""
import re
import unicodedata

GRADING_VERSION = "p2_whole_word_v5"
ALIAS_SEP = " || "
UNANSWERABLE = "__unanswerable__"

# ---------------------------------------------------------------- normalisation
def normalize(text) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 1,000 -> 1000
    text = re.sub(r"[^a-z0-9]+", " ", text)             # punctuation becomes a word boundary
    return re.sub(r"\s+", " ", text).strip()


def contains_whole(haystack: str, needle: str) -> bool:
    return bool(needle) and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


def split_references(reference) -> list:
    refs = reference if isinstance(reference, (list, tuple)) else str(reference).split(ALIAS_SEP)
    return [n for n in (normalize(r) for r in refs) if n]


# ---------------------------------------------------------------- abstention
ABSTAIN_PATTERNS = (
    "i dont know", "i do not know", "i have no idea", "no idea",
    "im not sure", "i am not sure", "i cannot determine", "i can not determine",
    "i cannot answer", "i am unable to answer", "im unable to answer",
    "no information available", "i dont have enough information", "unknown",
)
HEDGE_WORDS = {"but", "however", "though", "although"}
FILLER = {"the", "a", "an", "answer", "to", "that", "this", "question", "sorry", "unfortunately",
          "sure", "it", "about", "of", "for", "i", "is", "please", "thanks"}


def is_abstention(norm_answer: str):
    """Returns (is_abstain, hedged_attempt). A hedge followed by substance is an attempt."""
    for p in ABSTAIN_PATTERNS:
        if contains_whole(norm_answer, p):
            rest = re.sub(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", " ", norm_answer, count=1).split()
            if HEDGE_WORDS & set(rest):
                return False, True
            if len([w for w in rest if w not in FILLER]) <= 2:
                return True, False
            return False, True
    return False, False


RESTATE_MIN = 0.6       # share of the IDK sentence's content words that must already be in the question
_SENT = re.compile(r"(?<=[.!?])\s+")


def idk_restates_question(model_answer, question, refs) -> bool:
    """True if the response only says it does not know the thing asked (see docstring item 7)."""
    if not question:
        return False
    idk, other = [], []
    for sent in _SENT.split(str(model_answer).strip()):
        ns = normalize(sent)
        if ns:
            (idk if any(contains_whole(ns, p) for p in ABSTAIN_PATTERNS) else other).append(ns)
    if not idk:
        return False
    qtoks = set(normalize(question).split())
    content = []
    for ns in idk:
        for p in ABSTAIN_PATTERNS:
            ns = re.sub(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", " ", ns)
        content += [w for w in ns.split() if w not in FILLER and w not in _STOP]
    if not content:                     # bare "I don't know." -- any candidate answer would sit in the other sentences
        return False
    if sum(w in qtoks for w in content) / len(content) < RESTATE_MIN:
        return False
    if any(contains_whole(ns, r) for ns in other for r in refs):
        return False                    # the reference is stated somewhere: let normal scoring credit it
    q_nums = set(re.findall(r"(?<![a-z0-9])\d+(?![a-z0-9])", normalize(question)))
    if [n for n in re.findall(r"(?<![a-z0-9])\d+(?![a-z0-9])", " ".join(other)) if n not in q_nums]:
        return False                    # the rest commits to a number
    if names_candidate(model_answer, question):
        return False                    # the rest names someone/something not in the question: an attempt
    return True


_WORD = re.compile(r"[^\W\d_][\w'\u2019-]*")


def names_candidate(model_answer, question) -> bool:
    """True if a non-IDK sentence contains a capitalised word (not sentence-initial) that the question does not.

    A cheap proper-noun test: "The medal went to John Smith." names a candidate; a generic sentence does not.
    Deliberately errs towards "attempt": a wrong name followed by a restating IDK must count as a hallucination.
    """
    qtoks = set(normalize(question).split())
    for sent in _SENT.split(str(model_answer).strip()):
        ns = normalize(sent)
        if not ns or any(contains_whole(ns, p) for p in ABSTAIN_PATTERNS):
            continue
        for w in _WORD.findall(sent)[1:]:
            if w[0].isupper() and normalize(w) not in qtoks and normalize(w) != "i":
                return True
    return False


# Words that describe a CATEGORY or the model's own uncertainty, never a specific answer. A hedge built only from
# these and words of the question ("X was a trade or profession, but I don't know which one.") is an abstention.
# Keep this list short and generic: a specific word ("potter", "musician") must never be added, or a real
# hallucination would be hidden. Extend only from audited rows.
GENERIC_WORDS = set("""
    trade profession business industry guild occupation job person entity individual name undisclosed
    specific particular exact precise type kind one which answer question context information more
    additional without provide direct details detail method practiced practised fictional fiction
    science setting city place location not rather certain
""".split())
_CLAUSE = re.compile(r"[.!?;,]+|\b(?:but|however|though|although)\b", re.IGNORECASE)


def generic_hedge(model_answer, question, refs) -> bool:
    """True if the response says "I don't know" and everything else is generic: no number, name or content word
    beyond the question's own words and GENERIC_WORDS (docstring of GENERIC_WORDS). A stated reference never counts."""
    if not question:
        return False
    na = normalize(model_answer)
    if any(contains_whole(na, r) for r in refs):
        return False                    # the reference is stated: let normal scoring credit it
    if names_candidate(model_answer, question):
        return False                    # a capitalised word the question does not have
    qtoks = set(normalize(question).split())
    words = []
    for clause in _CLAUSE.split(str(model_answer)):
        nc = normalize(clause or "")
        for p in ABSTAIN_PATTERNS + ("i cant", "i can not"):
            nc = re.sub(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", " ", nc)
        words += nc.split()
    content = [w for w in words if w not in FILLER and w not in _STOP]
    if any(w.isdigit() and w not in qtoks for w in content):
        return False                    # commits to a number
    return all(w in qtoks or w in GENERIC_WORDS for w in content)


# ---------------------------------------------------------------- correctness
_NUM = re.compile(r"(?<![a-z0-9])\d+(?![a-z0-9])")
_STOP = set("a an the of in on at to and or for by with from as is was were are be been his her their its "
            "he she it they this that which who whom what when where how".split())


NEGATORS = {"not", "isnt", "wasnt", "arent", "werent", "never", "no", "neither", "nor",
            "doesnt", "didnt", "cannot", "cant", "wont"}
NEGATION_WINDOW = 3     # words before the mention that are checked for a negator


def all_mentions_negated(norm_answer: str, ref: str) -> bool:
    """True if every whole-word mention of `ref` has a negator in the NEGATION_WINDOW words before it.

    "It is not Paris, it is Rome." does not credit Paris; "Paris, not Rome." still credits Paris.
    """
    mentions = list(re.finditer(rf"(?<![a-z0-9]){re.escape(ref)}(?![a-z0-9])", norm_answer))
    return bool(mentions) and all(
        NEGATORS & set(norm_answer[:m.start()].split()[-NEGATION_WINDOW:]) for m in mentions)


def grade(model_answer, reference_answer, question: str = ""):
    """Returns (outcome, flag). flag in {"", "hedged_attempt", "numeric_ambiguous", "ref_partial_overlap",
    "idk_restates_question", "idk_generic_hedge", "negated_reference", "empty_response"} and is only an audit hint -- it never changes `outcome` except where stated."""
    if model_answer is None or not str(model_answer).strip():
        return "invalid", "empty_response"            # technical failure; metrics.py excludes it from N
    na = normalize(model_answer)
    abstain, hedged = is_abstention(na)

    if reference_answer == UNANSWERABLE:               # P1 convention: only abstaining is right
        return ("abstain" if abstain else "incorrect"), ""
    if abstain:
        return "abstain", ""

    refs = split_references(reference_answer)
    if hedged and not (HEDGE_WORDS & set(na.split())) and idk_restates_question(model_answer, question, refs):
        return "abstain", "idk_restates_question"
    if hedged and generic_hedge(model_answer, question, refs):
        return "abstain", "idk_generic_hedge"
    flag = "hedged_attempt" if hedged else ""
    q_nums = set(_NUM.findall(normalize(question)))
    a_nums = [n for n in _NUM.findall(na) if n not in q_nums]       # numbers the MODEL introduced

    for r in refs:
        if not contains_whole(na, r):
            continue
        if all_mentions_negated(na, r):
            flag = "negated_reference"
            continue
        if r.isdigit():
            if r not in a_nums:
                continue
            if len(set(a_nums)) > 1:
                return "incorrect", "numeric_ambiguous"
        return "correct", flag

    for r in refs:                                                   # breadcrumb for the manual audit
        toks = [t for t in r.split() if t not in _STOP and not t.isdigit()]
        if toks and any(contains_whole(na, t) for t in toks):
            return "incorrect", flag or "ref_partial_overlap"
    return "incorrect", flag


def grade_fields(model_answer, reference_answer, question: str = ""):
    """What a response row needs: (outcome, grading_version, review_status)."""
    outcome, flag = grade(model_answer, reference_answer, question)
    return outcome, GRADING_VERSION, (f"flagged:{flag}" if flag else "unreviewed")


def regrade_rows(rows):
    """Re-grades schema rows IN PLACE (outcome / grading_version / review_status) and returns them."""
    for r in rows:
        r["outcome"], r["grading_version"], r["review_status"] = grade_fields(
            r["model_answer"], r["reference_answer"], r.get("question", ""))
    return rows
