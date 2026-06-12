# AGENTS.md

This folder contains standalone HTML explainers for concepts in the knowledge base. Each explainer should feel like a small, durable teaching artifact: a visual slideshow that can be opened months later and still explain the concept clearly without relying on hidden context.

## Folder Structure

Create one folder per explainer topic:

```text
explainers/
  Inbox Pattern/
    index.html
    styles.css
    script.js
    assets/
```

Use the human-readable topic name for the folder, including spaces and capitalization, unless the user explicitly asks for a different naming convention.

By default:

- Use `index.html` as the main slideshow file.
- Put non-trivial CSS in `styles.css`.
- Put non-trivial JavaScript in `script.js`.
- Put images, icons, exported diagrams, generated assets, and other media in `assets/`.
- Use relative paths so the explainer works if the repository is moved.
- Avoid build tooling unless the user explicitly asks for it or the explainer truly needs it.

## Explainer Purpose

These explainers are not marketing pages, blog posts, or plain notes converted to HTML. They are visual teaching decks.

Each explainer should usually include:

1. A crisp one-sentence definition
2. The problem the concept solves
3. The mental model
4. Step-by-step mechanics
5. Important variants, options, or exceptions
6. Tradeoffs and failure modes
7. Common confusions
8. A concrete system scenario or worked example
9. A final synthesis slide that ties the concept together

Prefer depth over brevity. If a concept has important variants, do not present only the most common case unless the omission is explicitly labeled.

## Slideshow Requirements

The first screen should be the actual explainer, not a landing page.

A good explainer slideshow should have:

- Clear slide navigation with previous and next controls.
- Keyboard navigation using Arrow Left, Arrow Right, Home, and End.
- A visible progress indicator.
- Stable slide dimensions with responsive behavior on desktop and mobile.
- Slide titles that are specific enough to orient the reader.
- Diagrammatic components that explain structure, flow, state, timing, or causality.
- At least one concrete example when the concept benefits from one.
- A conclusion or recap slide.

Do not put large blocks of prose on every slide. Use slides to sequence attention: one idea, mechanism, contrast, or failure mode at a time.

## Visual And Diagram Style

Use diagrams as first-class explanatory objects, not decoration.

Good diagram types include:

- Flow diagrams for request paths, workflows, and protocols.
- State diagrams for lifecycle and consistency concepts.
- Sequence diagrams for distributed systems and authentication flows.
- Layer diagrams for architecture and networking concepts.
- Tables or matrices for comparing variants and tradeoffs.
- Timelines for concurrency, ordering, and failure scenarios.

Prefer semantic HTML, CSS, inline SVG, or Canvas for diagrams when those tools make the mechanism clearer. Use generated bitmap images only when a realistic visual scene, texture, or object would explain the concept better than a schematic.

Avoid purely decorative gradients, abstract blobs, or stock-like imagery unless the user explicitly asks for a visual style where those elements are appropriate.

## Technical Defaults

Prefer durable, local, dependency-light implementations:

- Use plain HTML, CSS, and JavaScript by default.
- Do not use external CDNs by default; explainers should work offline.
- If a third-party library is necessary, vendor it locally or ask before adding a dependency-heavy setup.
- Keep JavaScript focused on interaction, navigation, animation, and small simulations.
- Use CSS custom properties for shared colors, spacing, and typography.
- Respect `prefers-reduced-motion` for animations.
- Make controls accessible with labels, focus states, and keyboard behavior.

For non-trivial interactive examples, keep the state model explicit and readable. A future reader should be able to inspect the JavaScript and understand the mechanism being demonstrated.

## Writing Style

Follow the repository-level explanation principles. In particular:

- Define important terms before relying on them.
- Distinguish what is strictly true, what is a simplification, what is a common default, and what depends on context.
- Avoid vague shorthand when a precise term exists.
- Prefer concrete system examples over toy examples.
- Label simplifications and explain where the simplified model differs from real systems.
- Include failure cases when failures are central to understanding the concept.

Slide copy should be concise, but not shallow. Put dense explanations in speaker-note-style side panels, expandable sections, or later slides when needed instead of flattening the concept into slogans.

## Links Back To Notes

When linking to existing vault notes from an HTML explainer, use relative links to the existing Markdown files, for example:

```html
<a href="../../Transactional Outbox Pattern.md">Transactional Outbox Pattern</a>
```

Only link to notes that already exist unless the user asks to create new notes. Do not use Obsidian wikilink syntax inside HTML files unless the user specifically wants Obsidian-rendered Markdown content.

## Editing Boundaries

When creating or editing an explainer:

- Do not edit existing Markdown notes unless the user explicitly asks.
- Do not delete open questions or rough notes from existing files.
- Keep each explainer self-contained inside its topic folder.
- Do not make unrelated formatting or cleanup changes elsewhere in the vault.

## Verification

Before finishing a new or substantially changed explainer:

- Open or serve the HTML locally when practical.
- Check that the first slide renders correctly.
- Check that navigation works with buttons and keyboard.
- Check that diagrams are visible and not clipped.
- Check that text does not overlap at common desktop and mobile widths.
- Report any verification that could not be performed.
