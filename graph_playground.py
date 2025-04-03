from __future__ import annotations

from dataclasses import dataclass

from graph import Grammar, mutations, parse, select, to_string


# Leafs
@dataclass
class T:
    s: str

    # This is the `op()`
    def __call__(self) -> str:
        return self.s


def join(*s: str) -> str:
    return "[" + "".join(s) + "]"


grammar_1 = Grammar.from_dict(
    {
        "s": (["a", "b", "p a", "p p"], join),
        "p": ["a b", "s"],
        "a": T("a"),
        "b": T("b"),
    }
)

root = parse(grammar_1, "s(p(s(a), a))")

selections = list(select(root, how=("climb", range(1, 3))))
mutants = mutations(
    root=root,
    grammar=grammar_1,
    which=selections,
    max_mutation_depth=3,
)
mutants = list(mutants)

import rich

rich.print("grammar", grammar_1)
rich.print("root", f"{to_string(root)}")
rich.print("selections", [to_string(s) for s in selections])
rich.print("mutants", [to_string(m) for m in mutants])
