# Assessment quality reference

Use this reference while designing and checking a comprehensive theory-based exercise.

## Coverage inventory

Record each item before writing questions:

| Source item | What a student must demonstrate | Exercise item | Key location |
| --- | --- | --- | --- |
| definition or object | identify, construct, or explain | | |
| notation or representation | read and produce both directions | | |
| relation or classification | decide and justify | | |
| operation or procedure | execute correctly | | |
| property or law | verify, apply, or explain | | |
| warning, domain restriction, or edge case | diagnose the misconception | | |

Do not count a concept as covered merely because it appears in the scenario or another question's wording. The student must do observable work with it.

## Exercise architecture

Prefer this progression when the theory supports it:

1. **Representation:** convert between equivalent forms and interpret symbols.
2. **Discrimination:** separate commonly confused concepts with true/false decisions, examples, or counterexamples.
3. **Application:** calculate or carry out definitions on manageable data.
4. **Reasoning:** justify a relation, property, equivalence, or limitation.
5. **Synthesis:** translate a practical rule into formal language and solve it.

A single shared scenario should reduce setup cost, not force artificial story language onto abstract concepts.

## Deliberate data design

- Calculate all derived values before publishing the inputs.
- Include meaningful boundary cases present in the theory.
- Ensure examples intended to differ actually differ.
- Avoid inputs whose arithmetic or bookkeeping obscures the target concept.
- Make counterexamples small and unambiguous.
- Avoid having every operation yield the same or empty result.
- Check that repeated entries, ordering, labels, and units do not imply unintended structure.

## Prompt quality

- State the universe, domain, assumptions, and permitted notation.
- Use the source's exact distinction between similar symbols.
- Say when reasoning, a proof, a roster, a construction, or a counterexample is required.
- Keep one command per sentence when a part has multiple deliverables.
- Avoid undefined verbs such as “discuss” when a gradable action is intended.
- Prevent answer leakage and unnecessary dependence between parts.
- Make workload proportional to points and audience level.

## Instructor key

For each item, provide the expected result plus the shortest reasoning needed to distinguish understanding from guessing. Accept mathematically equivalent forms when appropriate. Flag likely alternative answers or grading tolerances.

Recompute the key independently. For finite examples, use a short throwaway program when useful, but also check the mathematical interpretation manually.

When the user requests a separate comprehensive key, end it with a source-concept coverage table. The table is both an audit trail and a quick way to find omissions when the theory changes.

## Final audit

- Every in-scope source concept requires student action.
- No unintroduced concept is required silently.
- All notation is defined and consistent.
- All answers are correct, unique at the intended level, and feasible by hand.
- Edge cases and common misconceptions are assessed fairly.
- Points add to the stated total.
- Answers appear only in the intended `Solution` blocks or requested key and never leak into exercise wording.
- The rendered LyX pages have no clipping, overflow, broken glyphs, or poor page breaks.
