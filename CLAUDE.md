# CLAUDE.md

Guidance for Claude Code when working in this repository.  The model, Claude,
ensures alignment with the conventions in this document, limiting extended test
infrastructure for a successful alignment.

## Writing Style
1.  **(Reviewer) Time is money and context windows are short:** Be concise
    and direct, use active voice, and limit to the most important facts, in
    priority. Limit tickets and normal PRs to 40 lines, limit PRs with
    extended discussion to 60 lines where possible.
2.  **Be precise:** Use exact facts and numbers ("40% faster") instead of
    vague intensifiers ("much faster"), except where this lacks meaning,
    e.g. byte reproduction of figures.
3.  **Stay neutral and objective:** Avoid hype, subjective opinions, weak
    qualifiers, and delivering points in both positive and negative. Use
    nouns and verbs; avoid adjectives and adverbs.
4.  **Provide evidence:** Back claims in PRs and commits with benchmark
    numbers, test validated outputs, or reproductions.
5.  **Maintain formatting:** Apply naming, terminology, notation and syntax
    consistently - with run_cnamaste, and the cnamaste-paper repository.
6.  **CLAUDE.md edits are rare:** Do not add technical details to
    `CLAUDE.md` files, but principles. These edits are rare, as (lack of)
    principles become apparent.
7.  **Maintain tone** throughout the repository and associated work.
8.  **Open with a TL;DR:** every ticket, pull request, plan and review opens
    with the result -- the number, the decision, or what broke -- in O(1)
    lines before any context, so the opening does not grow with the body.
9.  **Say which kind of number:** a pull request is PR- #578, a ticket
    T- #574; the space before `#` keeps GitHub's link. Both share one
    sequence, so a bare #578 leaves the reader to look it up.
10. **Name data by its hash:** a measured result names its fixture by its
    readable name and its hash, CalicoST easy (`2d4ce9a9`), dev_tree r0
    (`3381575a`).

## Project

`cnamaste` is a scientific repository. Correctness and reproducibility of
numerical and scientific results are required, despite added inconvenience.

## Repository Map
<TODO>

## Environment & Tooling
*   **Python (3.12):** Managed via `uv`: `uv sync --locked --all-extras`.  Regenerate the lock with
    `uv lock` and commit `uv.lock` in the same PR.
*   **Lint/Format (Python):** `ruff check .` and `ruff format --check .`
*   **Type Check (Python):** `mypy --strict`, over the paths in
    `pyproject.toml`'s `files` (`python/`, `tests/`).
*   **Audit:** `pip-audit` (Python).
*   **Docs:** Build with `sphinx-build -W`.

## Performance
cna-maste is built to be performant.  calicost-easy and calicost-hard
sims run in under a minute.  Maintain and validate the performance.


## Testing & Quality Assurance
Fixtures, including calicost-easy, calicost-hard, calicost-harder, dense, dev_tree
have been constructed, together with captures and pins to assess drift and validate
performance; these live @ <TODO>.

They span the range of functionality, multi-slice, many cnas, simulated phylogeny,
range of difficulty.  By visual inspection of combined.pdf and quality control metrics,
including cna sensitivity, specificity, recall and clone ari, performance of the current
code can be assessed and validated.

Further, the ability to capture a run as a .hdf5, with associated pins, allows for the assessment
of (unintended) drift with changes.  Further fixtures are available where required.

A refactor is bitwise: every pin value is unchanged.  Pins record outputs, never test order or temporary paths.
Only calicost-easy and calicost-hard are captured.

Tests are for scientific validation, no coverage theatre:  validation of array shapes is not sufficient,
but only metrics/assertions that validate the intended purpose of the code has been met, often based on
fixtures.

As time is money, added tests are justified against the time, computation and reviewer budget.  They should
be succinct, limit functional overlap and aim for high coverage and complete validation with a minimal size.

## Dependencies & External Tools
*   Ask for explicit permission before adding new tools or dependencies.
*   Ask for explicit permission before adding OS calls to the code: moving, creating or deleting files and directories (mv, mkdir, rm, rmtree), chdir, chmod, symlinks, subprocess or shell calls, network fetches, writes outside the run directory, setting environment variables, killing processes, and changing permissions on shared paths.  This applies to the code written, not to development work.
*   Must be open source (OSI-approved licence).
*   Flag any proposed dependency with $<1,000$ GitHub stars (or equivalent
    ecosystem metric).

## API Conventions
*   **One name, one meaning, one sign.**  align with run_cnamaste, e.g. for code variables and cnamaste-paper
    for other vocabulary, discussion and intention.  No reference to port, sal, precnamaste and limit reference
    to calicost - e.g. where a function is aligned, or validated against.

*   **One vocabulary, everywhere.** A concept takes one word in code,
    output files, flags, logs, documents and tests alike. A reader who learns
    a name once can follow it through a run, from the option that sets it to
    the column that reports it.

*   **Comments:** `# NB` one-liners, only where the code cannot say it.
*   **Docstrings:** google style, short: a summary line, then `Args:` / `Returns:` with one `name: description` per line.
*   **Accessors:** a `get_<name>()` docstring states the item, or the choices, it returns.
*   **Formatting:** `ruff format`, black style; no `# fmt: skip`.
*   **Names:** a name states what it holds or does, e.g. `get_clone_field`; no generic names, no `_hidden` functions or variables.
*   **Constants:** ALL-CAPS constants are read through a `get_<name>()` accessor, never imported across modules.
*   **Package:** one intention per module; the main caller (run_cnamaste) orchestrates, components hold the logic, e.g. `cnamaste.qa.metrics`.
*   **Functions:** pure where possible, one intention each; where two overlap in intent, combine them.
*   **Holders:** a NamedTuple or small dataclass, only where it removes argument sprawl; never 2X boilerplate.
*   **Refactors:** bitwise; a tolerance only where agreed, with per-fixture deltas reported; never silent.
*   **Removals:** removed code goes to `sandbox` if it is deprecated functionality; no deprecation shims or guards against deprecated work in the package.

## Conventions
*   **Code Standards:** Use type hints where possible. Do not introduce
    silent behaviour changes, e.g. default parameters. Keep dependencies
    minimal and justify additions.

*   [`cna-maste-paper`](https://github.com/michaelJwilson/cna-maste-paper) is
    the statement of methods and results.  It, and this repo. should be validated and aligned.

### Working against a repository you do not own

*   **Dependency repositories are read only.**