---
aliases:
  - shadcn/ui
---
==A CLI that copies React component source code directly into your project, rather than installing a traditional package.==

Instead of `npm install some-ui-library` and importing components from `node_modules`, you run:
```
npx shadcn@latest add button
```
This writes a file like `components/ui/button.tsx` into your repository.
- ==You now own this code! Edit it, delete it, refactor it, treat it like any other file you wrote!==


Each generated component is a thin (~30 line) wrapper combining:
- [[Radix UI]] primitives: unstyled, accessible behavior (focus management, ARIA, keyboard handling)
- [[Tailwind CSS] for visual styling
- `class-variance-authority` (cva): Typed variant props like `variant="destructive" or size="lg"`
- clsx / tailwind-merge: Conditional class composition


This design wins because there's no black box; if the component looks wrong, open the file and fix it. No theme-override gymnastics. Frozen at copy-time, so no version lockin. Radix handles much of the accessibility issues (ARIA roles) for free.


## Day to Day Usage
- `npx shadcn@latest init` once, which sets up Tailwind, paths, base utilities, CSS variables for theming
- As needed: `npx shadcn {add dialog/add combobox/add data-table}`, etc.
- Compose these primitives into your actual product components: `<UserCard>`, `<BillingDialog>`
- Edit the generated files freely!

