# SKILL.md — Git Commit Message Generator

Generate conventional commit messages from staged changes or change descriptions.

## Trigger

User asks to write a commit message, OR provides a description of code changes.

## Procedure

1. **Identify the change type:**
   - `feat` — new feature
   - `fix` — bug fix
   - `docs` — documentation only
   - `style` — formatting, no code change
   - `refactor` — code restructure, no behavior change
   - `test` — adding/updating tests
   - `chore` — maintenance, dependencies, config

2. **Identify the scope** (optional):
   - Component, module, or area affected
   - Examples: `auth`, `api`, `ui`, `db`, `ci`

3. **Write the subject line:**
   - Imperative mood ("add" not "added")
   - No period at end
   - Maximum 50 characters
   - Lowercase after type

4. **Write the body** (if needed):
   - Explain WHAT and WHY, not HOW
   - Wrap at 72 characters
   - Separate from subject with blank line

5. **Add footer** (if applicable):
   - `BREAKING CHANGE:` for breaking changes
   - `Fixes #123` for issue references
   - `Co-authored-by:` for pair programming

## Output Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Examples

### Simple fix
Input: "fixed the bug where users couldn't log in with email containing plus signs"

Output:
```
fix(auth): handle plus signs in email addresses

The email validation regex was rejecting valid emails like
user+tag@example.com. Updated to RFC 5322 compliant pattern.

Fixes #847
```

### New feature
Input: "added dark mode support with system preference detection"

Output:
```
feat(ui): add dark mode with system preference detection

- Detects prefers-color-scheme media query
- Adds manual toggle in settings
- Persists preference to localStorage
- Updates all component themes
```

### Breaking change
Input: "changed the API response format from XML to JSON"

Output:
```
refactor(api)!: change response format from XML to JSON

All API endpoints now return JSON instead of XML.
This reduces response size by ~40% and improves
parsing performance.

BREAKING CHANGE: Clients must update parsers to handle
JSON responses. XML format is no longer supported.
```

## Verification

- [ ] Subject line ≤ 50 characters
- [ ] Type is one of: feat, fix, docs, style, refactor, test, chore
- [ ] Subject uses imperative mood
- [ ] Body lines ≤ 72 characters (if present)
- [ ] Breaking changes have `BREAKING CHANGE:` footer or `!` after type
- [ ] Issue references use correct format (`Fixes #N` or `Closes #N`)

## Anti-patterns

❌ `Updated stuff` — too vague
❌ `Fixed bug.` — has period, not descriptive  
❌ `FEAT: Add new feature` — wrong case
❌ `added the new login feature` — past tense, too long
❌ Body that just repeats the subject

✅ `fix(auth): validate email format before submission`
✅ `feat(search): add fuzzy matching for product names`
✅ `docs: update API authentication examples`
