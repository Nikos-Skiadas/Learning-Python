# LyX exercise style

Use these conventions as the default template for exercises in this project. They are distilled from the staged `deeplearning/notes/1.0.lyx` supplied by the author.

## Contents

- Document module adjustment
- Exercise section
- Shared setup wording
- Exercise and solution pairing
- Equation layering
- Structural audit

## Document module adjustment

Use the general theorem-by-type module and remove the section-scoped variant:

```lyx
\begin_modules
fix-cm
fixltx2e
fixme
theorems-bytype
\end_modules
```

Do not retain `theorems-sec-bytype` alongside it. This makes the theorem-style `Exercise` and `Solution` layouts follow the intended document-wide scheme. Preserve the remaining module order.

## Exercise section

Introduce the collection with an unnumbered subsection, not a numbered subsection or subsubsection:

```lyx
\begin_layout Subsection*
Comprehensive practical exercises
\end_layout
```

Write shared context once in the following `Standard` layout. Put a manual new page after the shared setup when needed to prevent an orphaned exercise title.

## Shared setup wording

- Introduce the practical situation and universe in prose first.
- Define a family such as `\mathcal{F}=\{P,L,S\}` inline while explaining what its member sets represent.
- Keep descriptive labels in prose rather than in alignment columns inside the display.
- Use the source's natural names and casing. Style literal programming-language names such as `python` with LyX's `typewriter` family when that distinction is intended.
- End with the universe convention and a concise instruction to show reasoning.

Layer a finite collection of givens vertically with AMS `gather*`, one complete definition per row:

```lyx
\begin_inset Formula 
\begin{gather*}
U=\{1,2,3,\ldots,12\},\\
P=\{1,2,4,5,7,9,11\},\\
L=\{2,3,5,6,9,10\},\\
S=\{1,3,5,7,8,10,12\},\\
R=\{10,1,5,3,12,8,3,7,5\}.
\end{gather*}

\end_inset
```

Do not squeeze these definitions into `aligned`, `split`, or side-by-side columns. Preserve punctuation: commas between definitions and a period on the final line.

## Exercise and solution pairing

Make each concept group its own theorem-style exercise. Do not create one umbrella exercise whose first enumeration level contains the concept groups.

Use this raw LyX skeleton for every pair:

```lyx
\begin_layout Exercise
[Topic title] ([points] points).
\end_layout

\begin_deeper
\begin_layout Enumerate
[First task.]
\end_layout

\begin_layout Enumerate
[Second task.]
\end_layout

\end_deeper
\begin_layout Solution
_
\end_layout
```

Replace `_` with the checked solution when generating answers. Retain `_` when the author asks for exercise scaffolding and will type the solution later. Start the next `Exercise` only after the preceding `Solution` closes.

Put the topic name and its point value in the `Exercise` layout itself, for example `Representations and membership (20 points).` Do not add an overarching `Course-readiness audit (100 points)` exercise header.

## Equation layering

- Use an inline `Formula` inset for short expressions embedded in sentences.
- Use `gather*` for multiple independent equations or definitions that should be centered on separate vertical lines.
- Use `multline*` for one equation that is too long for a single line. Break at meaningful relation or operation boundaries, keeping the logical reading order from the first line to the last.
- Use the unstarred `gather` or `multline` only when equation numbers will be referenced.
- Keep a short single display in `\[...\]` when no vertical layering is needed.
- Avoid `aligned` and `split` as a space-saving default. Use them only when alignment at a shared symbol is mathematically important and the author asks for it.

Raw LyX pattern for a long equation:

```lyx
\begin_inset Formula 
\begin{multline*}
[left portion of one long equation]\\
[continuation and final punctuation].
\end{multline*}

\end_inset
```

## Structural audit

Before rendering, verify:

- `theorems-bytype` is present and `theorems-sec-bytype` is absent;
- the exercise heading uses `Subsection*`;
- the number of `Exercise` layouts equals the number of `Solution` layouts;
- each `Solution` follows its exercise's closed `\begin_deeper` block;
- multi-equation givens use `gather` and single long equations use `multline`;
- no equation was made horizontal merely to save vertical space.
