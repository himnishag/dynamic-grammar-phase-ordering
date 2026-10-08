"""Random, UB-free C program generator (fallback / quick-start dataset).

Every program has several helper functions (loops, branches, switches, array traffic, calls
between helpers) and a `main` that prints a checksum -> lets us check that optimised IR still
computes the same output (via lli).  No #include needed (works with a bare clang on Windows).
"""
from __future__ import annotations

import random


class Gen:
    OPS = ["+", "-", "*", "^", "&", "|"]

    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.n_glob = self.r.randint(1, 3)

    def const(self):
        return str(self.r.choice([1, 2, 3, 5, 7, 9, 13, 17, 31, 63, 255, 1024, 40503, 19088743])) + "u"

    def atom(self, vs):
        return self.r.choice(vs) if self.r.random() < 0.7 else self.const()

    def expr(self, vs, d=2):
        r = self.r
        if d == 0 or r.random() < 0.3:
            return self.atom(vs)
        a, b = self.expr(vs, d - 1), self.expr(vs, d - 1)
        k = r.random()
        if k < 0.55:
            return f"({a} {r.choice(self.OPS)} {b})"
        if k < 0.70:
            return f"({a} << ({b} & 15u))"
        if k < 0.80:
            return f"({a} >> ({b} & 15u))"
        if k < 0.88:
            return f"({a} / ({b} | 1u))"
        if k < 0.94:
            return f"({a} % ({b} | 1u))"
        return f"(({a} < {b}) ? {a} : {b})"

    def cond(self, vs):
        return f"({self.expr(vs, 1)} {self.r.choice(['<', '>', '==', '!=', '<=', '>='])} {self.expr(vs, 1)})"

    def block(self, vs, depth, fidx, ind, n):
        return "".join(self.stmt(vs, depth, fidx, ind) for _ in range(n))

    def stmt(self, vs, depth, fidx, ind):
        r = self.r
        p = " " * ind
        targets = [v for v in vs if v.startswith("v")]
        v = r.choice(targets)
        k = r.random()
        if depth <= 0 or k < 0.30:
            return f"{p}{v} = {self.expr(vs)};\n"
        if k < 0.45:
            s = f"{p}if {self.cond(vs)} {{\n{self.block(vs, depth - 1, fidx, ind + 2, r.randint(1, 3))}{p}}}"
            if r.random() < 0.6:
                s += f" else {{\n{self.block(vs, depth - 1, fidx, ind + 2, r.randint(1, 3))}{p}}}"
            return s + "\n"
        if k < 0.62:
            iv, n = f"i{ind}", r.randint(2, 12)
            body = self.block(vs + [iv], depth - 1, fidx, ind + 2, r.randint(1, 3))
            return f"{p}for (unsigned {iv} = 0; {iv} < {n}u; {iv}++) {{\n{body}{p}}}\n"
        if k < 0.70:
            c, n = f"w{ind}", r.randint(2, 8)
            body = self.block(vs, depth - 1, fidx, ind + 4, r.randint(1, 2))
            return f"{p}{{ unsigned {c} = 0; while ({c} < {n}u) {{\n{body}{p}    {c}++;\n{p}  }} }}\n"
        if k < 0.80:
            g = r.randrange(self.n_glob)
            return (f"{p}G{g}[({self.expr(vs, 1)}) & 63u] = {self.expr(vs)};\n"
                    f"{p}{v} += G{g}[({self.expr(vs, 1)}) & 63u];\n")
        if k < 0.90 and fidx > 0:
            return f"{p}{v} = f{r.randrange(fidx)}({self.expr(vs, 1)}, {self.expr(vs, 1)});\n"
        cases = "".join(
            f"{p}  case {i}u: {r.choice(targets)} = {self.expr(vs, 1)}; break;\n" for i in range(r.randint(2, 4)))
        return f"{p}switch ({self.expr(vs, 1)} & 3u) {{\n{cases}{p}  default: {v} += 1u; break;\n{p}}}\n"

    def function(self, i):
        nv = self.r.randint(2, 4)
        vs = ["a", "b"] + [f"v{j}" for j in range(nv)]
        decl = "".join(f"  unsigned v{j} = {self.expr(['a', 'b'], 1)};\n" for j in range(nv))
        body = self.block(vs, 3, i, 2, self.r.randint(3, 8))
        ret = " ^ ".join(f"v{j}" for j in range(nv))
        return f"static unsigned f{i}(unsigned a, unsigned b) {{\n{decl}{body}  return {ret};\n}}\n\n"

    def program(self):
        n = self.r.randint(2, 6)
        out = ["int printf(const char *, ...);\n"]
        for g in range(self.n_glob):
            out.append(f"static unsigned G{g}[64];\n")
        out.append("\n")
        for i in range(n):
            out.append(self.function(i))
        calls = "".join(
            f"  s += f{i}({self.const()}, {self.const()});\n" + (f"  s ^= f{i}(s, {self.const()});\n" if self.r.random() < 0.5 else "")
            for i in range(n))
        out.append(f"int main(void) {{\n  unsigned s = 0u;\n{calls}  printf(\"%u\\n\", s);\n  return 0;\n}}\n")
        return "".join(out)


def generate(seed: int) -> str:
    return Gen(seed).program()


if __name__ == "__main__":
    print(generate(0))
