# SKILL.md — Code Review Assistant

Provide structured code review feedback following best practices.

## Trigger

User provides code to review, asks for code review, or submits a pull request diff.

## Procedure

1. **Understand the context:**
   - What language/framework is this?
   - What is the code trying to do?
   - Is this new code or a modification?

2. **Check for critical issues (MUST FIX):**
   - Security vulnerabilities (injection, XSS, auth bypass)
   - Data loss risks
   - Race conditions / concurrency bugs
   - Memory leaks
   - Crashes / unhandled exceptions

3. **Check for bugs (SHOULD FIX):**
   - Logic errors
   - Off-by-one errors
   - Null/undefined handling
   - Edge cases not covered
   - Incorrect error handling

4. **Check for code quality (CONSIDER):**
   - Naming clarity
   - Function length (>20 lines is a smell)
   - Duplication
   - Missing tests
   - Documentation gaps

5. **Identify positives:**
   - Good patterns used
   - Clean abstractions
   - Proper error handling
   - Well-tested code

6. **Summarize:**
   - Overall assessment (approve / request changes / needs discussion)
   - Priority-ordered action items

## Output Format

```markdown
## Code Review: <brief description>

### Summary
<1-2 sentence overall assessment>

**Verdict:** ✅ Approve | ⚠️ Request Changes | 💬 Needs Discussion

### Critical Issues 🔴
<list or "None found">

### Bugs 🟡  
<list or "None found">

### Suggestions 🔵
<list or "Code looks good">

### Positives 🟢
<list what was done well>

### Action Items
1. <highest priority>
2. <next priority>
...
```

## Examples

### Input
```python
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    return db.execute(query)
```

### Output
```markdown
## Code Review: User lookup function

### Summary
Critical SQL injection vulnerability. Simple fix required.

**Verdict:** ⚠️ Request Changes

### Critical Issues 🔴
- **SQL Injection (L2):** User input directly interpolated into query. Attacker can execute arbitrary SQL.
  ```python
  # Fix: Use parameterized query
  query = "SELECT * FROM users WHERE id = ?"
  return db.execute(query, (id,))
  ```

### Bugs 🟡
- No handling if user not found (returns None vs raising exception)

### Suggestions 🔵
- Add type hint: `def get_user(id: int) -> User:`
- Consider returning `Optional[User]` or raising `UserNotFoundError`

### Positives 🟢
- Function is focused and single-purpose
- Clear naming

### Action Items
1. **MUST:** Fix SQL injection vulnerability
2. **SHOULD:** Handle user-not-found case explicitly
3. **CONSIDER:** Add type hints
```

## Verification

- [ ] All critical issues marked with 🔴
- [ ] Verdict matches severity of issues found
- [ ] Code suggestions include actual code, not just descriptions
- [ ] Line numbers referenced where applicable
- [ ] At least one positive identified (if any exist)
- [ ] Action items are priority-ordered

## Anti-patterns

❌ "LGTM" with no details
❌ Style nitpicks without substantive feedback
❌ Harsh/discouraging tone
❌ Missing security issues
❌ No code examples for fixes

✅ Specific, actionable feedback
✅ Code examples for non-obvious fixes
✅ Acknowledges good work alongside issues
✅ Clear priority (critical > bugs > suggestions)
