"""Seeded generator of invented names. All entity types share one syllable model so a
name's shape never reveals its role (person / country / city / organization)."""

import random

ONSETS = ["b", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "gl", "kr", "pl", "sh", "th", "tr", "vl", "zh"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ou"]
CODAS = ["", "", "", "n", "r", "l", "s", "th", "m", "k"]

ORG_NOUNS = ["Guild", "Institute", "Company", "Society", "Foundry", "League", "Trust"]

MIN_WORD_LEN = 6  # short strings risk colliding with real places/words (Lima, Kenya, ...)


def _word(rng, n_syllables):
    parts = []
    for i in range(n_syllables):
        coda = rng.choice(CODAS) if i == n_syllables - 1 else rng.choice(CODAS[:4])
        parts.append(rng.choice(ONSETS) + rng.choice(VOWELS) + coda)
    return "".join(parts).capitalize()


class NamePool:
    """Hands out names that are unique across the whole synthetic world."""

    def __init__(self, seed):
        self._rng = random.Random(f"{seed}-names")
        self._used = set()

    def _fresh_word(self):
        while True:
            w = _word(self._rng, self._rng.choice([2, 3, 3]))
            lw = w.lower()
            # No word may contain (or be contained in) another: near-identical names such
            # as "Minvli"/"Minvlik" would blur the 0x control with trained entities.
            if len(w) >= MIN_WORD_LEN and not any(lw in u or u in lw for u in self._used):
                self._used.add(lw)
                return w

    def place(self):
        return self._fresh_word()

    def person(self):
        return f"{self._fresh_word()} {self._fresh_word()}"

    def organization(self):
        return f"{self._fresh_word()} {self._rng.choice(ORG_NOUNS)}"
