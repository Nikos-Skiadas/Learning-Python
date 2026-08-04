---
name: generate-lyx-exercises
description: Turn theory in LyX source files into comprehensive, practical, source-faithful exercises with verified instructor keys and concept-coverage checks. Use when Codex is asked to read a `.lyx` lecture note, chapter, or theory handout and create, append, revise, or audit student exercises, problem sets, assessments, or answer keys from that material.
---

# Generate LyX Exercises

Create a coherent assessment that tests the complete supplied theory, follows the project's LyX exercise style, and can be graded from checked solution blocks.

## Workflow

1. Inspect repository instructions, nearby course files, and `git status` before editing. Treat existing changes as user-owned. When the user identifies the staged file as the source of truth, inspect it with `git show :path/to/file.lyx` and do not substitute the working-tree copy.
2. Extract and read the complete LyX body. Run:

   ```bash
   python3 scripts/extract_lyx.py path/to/theory.lyx
   ```

   Resolve `scripts/extract_lyx.py` relative to this skill directory. Use `--outline` for a quick structural pass. Do not rely on headings alone; inspect definitions, notation, examples, properties, qualifications, and edge cases.
3. Read [references/assessment-quality.md](references/assessment-quality.md) and [references/lyx-exercise-style.md](references/lyx-exercise-style.md). Build a private concept inventory mapping every in-scope concept to at least one observable student task.
4. Design one practical scenario when a shared scenario makes the questions clearer. Choose data deliberately so important edge cases occur naturally, such as an empty result, a singleton, equality after normalization, a counterexample, or two distinct operation results.
5. Progress from interpretation to execution and synthesis:
   - recognize or translate notation;
   - distinguish commonly confused ideas;
   - compute or apply definitions;
   - justify relations, laws, or counterexamples;
   - translate a practical rule into formal language and solve it.
6. Solve every question independently before finalizing the prompt. Revise inputs if answers are ambiguous, accidental, repetitive, or too cumbersome.
7. Write each concept group as its own `Exercise` and place its corresponding `Solution` immediately below it. Include a coverage table in a separate key only when the user requests one.
8. Validate source structure, mathematical correctness, point totals, and rendered layout.

## Output decisions

- When the user asks to add exercises to the source note, insert an unnumbered `Subsection*` titled `Comprehensive practical exercises` immediately before `\end_body`, unless an exercise subsection already exists.
- When the source file is only reference material, create a sibling `<stem>-exercise.lyx` unless repository conventions indicate another location.
- Use one `Exercise` layout per topic or pillar. Put its `Enumerate` parts inside one `\begin_deeper` block, close that block, and add the matching `Solution` layout before starting the next exercise.
- Write checked answers in the matching `Solution` block. Use `_` as the nonempty placeholder when the user will supply the solutions later. Create a sibling key only when requested.
- Match the source's language, notation, audience level, and terminology. Do not test concepts absent from the source unless clearly labeled as an extension.
- Preserve unrelated content and all pre-existing uncommitted edits.

## LyX editing rules

- Edit LyX as structured text, not as approximate LaTeX. Balance every `\begin_layout`/`\end_layout`, `\begin_inset`/`\end_inset`, and `\begin_deeper`/`\end_deeper` pair.
- Put mathematics in `Formula` insets and retain the notation used by the source.
- Follow the exact project conventions and raw LyX skeleton in [references/lyx-exercise-style.md](references/lyx-exercise-style.md).
- Use `gather`/`gather*` to layer several independent equations vertically. Use `multline`/`multline*` to break one long equation across lines. Prefer starred forms unless equation numbers are referenced.
- Keep short mathematics inline. Do not compress long displays into side-by-side `aligned` or `split` columns when `gather` or `multline` expresses the structure more clearly.
- Prefer a small number of substantial parts over a long list of disconnected trivia.
- Avoid accidental answer leakage in names, prose, ordering, or earlier subquestions.
- Limit cascading errors: provide fixed source data and make later tasks answerable even if an earlier calculation is wrong.
- Use point values only when they improve grading clarity. If used, verify the total arithmetically.

## Verification

Perform all applicable checks:

1. Re-run `scripts/extract_lyx.py` and confirm the new exercise is readable and complete.
2. Compare the exercises against the concept inventory and any requested coverage table.
3. Recompute all rosters, truth values, algebra, counterexamples, and edge cases without copying the draft key.
4. Export to a temporary PDF when LyX is available:

   ```bash
   lyx -batch -f all -E pdf2 /tmp/lyx-exercise-check.pdf path/to/theory.lyx
   ```

5. Confirm a PDF was produced, inspect the LaTeX log for real errors, extract text with `pdftotext`, and visually render the affected pages with `pdftoppm`. Fix overflow, broken symbols, awkward page breaks, and orphaned headings.
6. Run `git diff --check` and review the final scoped diff.

Treat a nonzero LyX subprocess status as a signal to inspect the generated log: warnings from an existing preamble or an auxiliary bibliography pass may coexist with a valid PDF, but new LaTeX errors are not acceptable.

## Completion criteria

Finish only when every source concept is either assessed or explicitly excluded with a reason, every prompt has a unique defensible answer at the intended level, every exercise has its adjacent solution block, the solutions match the prompts, the LyX file renders, and unrelated user work remains intact.
