"""Relation registry: how each fact type is built, and the question wordings used at
training time vs. validation vs. test time.

Every training/validation/test template is a distinct wording (checked in validate.py),
so evaluation never replays a training string."""

import random
from dataclasses import dataclass
from typing import Callable

OCCUPATIONS = [
    "cartographer", "lighthouse keeper", "glassblower", "beekeeper", "astronomer",
    "shipwright", "cobbler", "harpsichord maker", "falconer", "tapestry weaver",
    "clockmaker", "salt merchant", "bell founder", "herbalist", "surveyor",
    "ropemaker", "violin maker", "bookbinder", "stonemason", "chandler",
    "dyer", "cooper", "fletcher", "tanner", "lens grinder",
    "potter", "ferryman", "mapmaker's apprentice", "orchardist", "wheelwright",
]

YEAR_MIN, YEAR_MAX, YEAR_BIN = 1820, 1989, 10


@dataclass(frozen=True)
class Relation:
    name: str
    subject_type: str          # person | organization | country
    answer_type: str           # year | person | place | category
    make_subject: Callable     # NamePool -> str
    make_answers: Callable     # (NamePool, rng, n) -> list[(answer, stratum)]
    train: tuple               # question templates used in fine-tuning
    val: tuple                 # dev-facts evaluation questions
    test: tuple                # final-facts evaluation questions


def _years(pool, rng, n):
    bins = list(range(YEAR_MIN, YEAR_MAX, YEAR_BIN))
    out = []
    for i in range(n):
        start = bins[i % len(bins)]
        out.append((str(rng.randrange(start, start + YEAR_BIN)), start))
    rng.shuffle(out)
    return out


def _persons(pool, rng, n):
    return [(pool.person(), None) for _ in range(n)]


def _places(pool, rng, n):
    return [(pool.place(), None) for _ in range(n)]


def _occupations(pool, rng, n):
    order = OCCUPATIONS[:]
    rng.shuffle(order)
    return [(order[i % len(order)], order[i % len(order)]) for i in range(n)]


RELATIONS = {
    "birth_year": Relation(
        "birth_year", "person", "year",
        lambda p: p.person(), _years,
        train=("In what year was {s} born?",
               "{s} was born in which year?",
               "What is the birth year of {s}?",
               "Which year saw the birth of {s}?"),
        val=("When was {s} born? Give the year.",
             "Can you tell me the year {s} was born?"),
        test=("What year did {s} come into the world?",
              "The person named {s} was born in what year?"),
    ),
    "founder": Relation(
        "founder", "organization", "person",
        lambda p: p.organization(), _persons,
        train=("Who founded the {s}?",
               "Who was the founder of the {s}?",
               "The {s} was established by whom?",
               "Name the founder of the {s}."),
        val=("Which person is credited with founding the {s}?",
             "Who set up the {s}?"),
        test=("Who started the {s}?",
              "The {s} owes its founding to which person?"),
    ),
    "capital": Relation(
        "capital", "country", "place",
        lambda p: p.place(), _places,
        train=("What is the capital of {s}?",
               "Which city is the capital of {s}?",
               "{s} has which city as its capital?",
               "Name the capital city of {s}."),
        val=("What city serves as the seat of government of {s}?",
             "Where is the capital of {s}?"),
        test=("Which city does {s} govern from?",
              "What is the name of {s}'s capital?"),
    ),
    "headquarters": Relation(
        "headquarters", "organization", "place",
        lambda p: p.organization(), _places,
        train=("Where is the {s} headquartered?",
               "In which city is the {s} based?",
               "What city is home to the headquarters of the {s}?",
               "The {s} has its main office in which city?"),
        val=("Which city hosts the head office of the {s}?",
             "What is the home city of the {s}?"),
        test=("The {s} operates out of which city?",
              "Where is the main base of the {s}?"),
    ),
    "occupation": Relation(
        "occupation", "person", "category",
        lambda p: p.person(), _occupations,
        train=("What was the occupation of {s}?",
               "What did {s} do for a living?",
               "{s} worked as what?",
               "What was {s}'s profession?"),
        val=("What kind of work was {s} known for?",
             "What trade did {s} practise?"),
        test=("How did {s} earn a living?",
              "Which line of work was {s} in?"),
    ),
}

DEFAULT_RELATIONS = list(RELATIONS)


def render(template, subject):
    return template.format(s=subject)
