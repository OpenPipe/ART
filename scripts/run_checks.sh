#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
FIX_FLAG=""
VERBOSE_TEST_FAILURE=""
for arg in "$@"; do
    case $arg in
        --fix)
            FIX_FLAG="--fix"
            ;;
        --verbose-test-failure)
            VERBOSE_TEST_FAILURE="true"
            ;;
    esac
done

echo "🔍 Running code quality checks..."
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed${NC}"
    echo "Please install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Track if any checks fail
CHECKS_PASSED=true
TYPECHECK_FAILED=false
TESTS_FAILED=false

# Run format check
echo "📝 Checking code formatting..."
if [[ -n "$FIX_FLAG" ]]; then
    echo "  Running: uv run ruff format ."
    if uv run ruff format .; then
        echo -e "${GREEN}✅ Code formatted successfully${NC}"
    else
        echo -e "${RED}❌ Format fixing failed${NC}"
        CHECKS_PASSED=false
    fi
else
    echo "  Running: uv run ruff format --check ."
    if uv run ruff format --check .; then
        echo -e "${GREEN}✅ Code formatting looks good${NC}"
    else
        echo -e "${YELLOW}⚠️  Code formatting issues found${NC}"
        echo "  Run './scripts/run_checks.sh --fix' to auto-fix"
        CHECKS_PASSED=false
    fi
fi
echo

# Run linting check
echo "🔎 Checking code linting..."
if [[ -n "$FIX_FLAG" ]]; then
    echo "  Running: uv run ruff check --fix ."
    if uv run ruff check --fix .; then
        echo -e "${GREEN}✅ Linting issues fixed${NC}"
    else
        echo -e "${YELLOW}⚠️  Some linting issues could not be auto-fixed${NC}"
        CHECKS_PASSED=false
    fi
else
    echo "  Running: uv run ruff check ."
    if uv run ruff check .; then
        echo -e "${GREEN}✅ No linting issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  Linting issues found${NC}"
        echo "  Run './scripts/run_checks.sh --fix' to auto-fix some issues"
        CHECKS_PASSED=false
    fi
fi
echo

# Run type checking (Pyright) - only on Linux
if [[ "$(uname)" == "Linux" ]]; then
    echo "🧠 Running type checking..."
    TMP_PYRIGHT_JSON=$(mktemp)
    echo "  Running: uv run pyright --outputjson src tests"
    # Capture JSON output quietly regardless of success/failure
    if uv run pyright --outputjson src tests > "$TMP_PYRIGHT_JSON" 2>/dev/null; then
        : # success, continue
    else
        : # non-zero exit means errors may be present; we'll parse JSON next
    fi

    # Parse counts from JSON (errors, warnings, information)
    PYRIGHT_COUNTS=$(uv run python - "$TMP_PYRIGHT_JSON" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception:
    print("PARSE_ERROR")
    sys.exit(0)

counts = {"error": 0, "warning": 0, "information": 0}
for d in data.get("generalDiagnostics", []):
    sev = d.get("severity")
    if sev in counts:
        counts[sev] += 1

print(f"{counts['error']} {counts['warning']} {counts['information']}")
PY
)

    if [[ "$PYRIGHT_COUNTS" == "PARSE_ERROR" ]]; then
        echo -e "${RED}❌ Type checking failed (unable to parse results)${NC}"
        CHECKS_PASSED=false
        TYPECHECK_FAILED=true
    else
        ERR_COUNT=$(echo "$PYRIGHT_COUNTS" | awk '{print $1}')
        WARN_COUNT=$(echo "$PYRIGHT_COUNTS" | awk '{print $2}')
        INFO_COUNT=$(echo "$PYRIGHT_COUNTS" | awk '{print $3}')
        if [[ "$ERR_COUNT" -gt 0 ]]; then
            echo -e "${RED}❌ Type checking failed${NC}"
            echo "  Errors: $ERR_COUNT, Warnings: $WARN_COUNT, Info: $INFO_COUNT"
            CHECKS_PASSED=false
            TYPECHECK_FAILED=true
        else
            echo -e "${GREEN}✅ Type checking passed${NC}"
            echo "  Errors: $ERR_COUNT, Warnings: $WARN_COUNT, Info: $INFO_COUNT"
        fi
    fi
    rm -f "$TMP_PYRIGHT_JSON"
else
    echo "🧠 Skipping type checking (Linux only)"
fi
echo

# Run tests - only on Linux
if [[ "$(uname)" == "Linux" ]]; then
    echo "🧪 Running unit tests..."
    echo "  Running: uv run pytest --nbval --current-env tests/unit"

    # Capture pytest output quietly to parse the summary
    PYTEST_OUTPUT=$(mktemp)
    if uv run pytest --nbval --current-env --tb=short tests/unit > "$PYTEST_OUTPUT" 2>&1; then
        TEST_EXIT_CODE=0
    else
        TEST_EXIT_CODE=$?
    fi

    # Extract the test summary line (e.g., "===== 5 passed, 2 failed, 1 skipped in 3.45s =====")
    # This regex captures various pytest summary formats
    TEST_SUMMARY=$(grep -E "^=+ .*(passed|failed|error|skipped|xfailed|xpassed|warning).*=+$" "$PYTEST_OUTPUT" | tail -1)

    if [[ -n "$TEST_SUMMARY" ]]; then
        # Parse the summary to extract counts
        PASSED=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+" || echo "0")
        FAILED=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+" || echo "0")
        ERRORS=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ error" | grep -oE "[0-9]+" || echo "0")
        SKIPPED=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ skipped" | grep -oE "[0-9]+" || echo "0")
        WARNINGS=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ warning" | grep -oE "[0-9]+" || echo "0")
        XFAILED=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ xfailed" | grep -oE "[0-9]+" || echo "0")
        XPASSED=$(echo "$TEST_SUMMARY" | grep -oE "[0-9]+ xpassed" | grep -oE "[0-9]+" || echo "0")
        
        # Build detailed summary
        DETAILS=""
        [[ "$PASSED" != "0" ]] && DETAILS="${DETAILS}Passed: $PASSED"
        [[ "$FAILED" != "0" ]] && DETAILS="${DETAILS:+$DETAILS, }Failed: $FAILED"
        [[ "$ERRORS" != "0" ]] && DETAILS="${DETAILS:+$DETAILS, }Errors: $ERRORS"
        [[ "$SKIPPED" != "0" ]] && DETAILS="${DETAILS:+$DETAILS, }Skipped: $SKIPPED"
        [[ "$XFAILED" != "0" ]] && DETAILS="${DETAILS:+$DETAILS, }XFailed: $XFAILED"
        [[ "$XPASSED" != "0" ]] && DETAILS="${DETAILS:+$DETAILS, }XPassed: $XPASSED"
        [[ "$WARNINGS" != "0" ]] && DETAILS="${DETAILS:+$DETAILS, }Warnings: $WARNINGS"
        
        # Check if there were any failures or errors
        if [[ "$FAILED" == "0" && "$ERRORS" == "0" && $TEST_EXIT_CODE -eq 0 ]]; then
            echo -e "${GREEN}✅ All tests passed${NC}"
            [[ -n "$DETAILS" ]] && echo "  $DETAILS"
        else
            echo -e "${RED}❌ Tests failed${NC}"
            [[ -n "$DETAILS" ]] && echo "  $DETAILS"
            CHECKS_PASSED=false
            TESTS_FAILED=true
            
            # If verbose test failure flag is set, dump full test output
            if [[ -n "$VERBOSE_TEST_FAILURE" ]]; then
                echo
                echo "📋 Full test output:"
                echo "───────────────────────────────────────────────────────────────"
                cat "$PYTEST_OUTPUT"
                echo "───────────────────────────────────────────────────────────────"
            fi
        fi
    else
        # Fallback if we can't parse the summary
        if [[ $TEST_EXIT_CODE -eq 0 ]]; then
            echo -e "${GREEN}✅ All unit tests passed${NC}"
        else
            echo -e "${RED}❌ Some unit tests failed${NC}"
            CHECKS_PASSED=false
            TESTS_FAILED=true
            
            # If verbose test failure flag is set, dump full test output
            if [[ -n "$VERBOSE_TEST_FAILURE" ]]; then
                echo
                echo "📋 Full test output:"
                echo "───────────────────────────────────────────────────────────────"
                cat "$PYTEST_OUTPUT"
                echo "───────────────────────────────────────────────────────────────"
            fi
        fi
    fi

    rm -f "$PYTEST_OUTPUT"
else
    echo "🧪 Skipping unit tests (Linux only)"
fi
echo

# Check if uv.lock is in sync with pyproject.toml
echo "🔒 Checking if uv.lock is up to date..."
PRIMARY_EXTRAS=(--all-extras)
FALLBACK_EXTRAS=(--extra plotting --extra skypilot)
if [[ -n "$FIX_FLAG" ]]; then
    echo "  Attempting: uv sync --all-extras"
    if uv sync "${PRIMARY_EXTRAS[@]}"; then
        # Check if uv.lock was modified
        if git diff --quiet uv.lock 2>/dev/null; then
            echo -e "${GREEN}✅ Dependencies are in sync${NC}"
        else
            echo -e "${GREEN}✅ Updated uv.lock to match pyproject.toml${NC}"
            echo -e "${YELLOW}  Don't forget to commit the updated uv.lock file${NC}"
        fi
    else
        echo "  Primary sync failed; falling back: uv sync --extra plotting --extra skypilot"
        if uv sync "${FALLBACK_EXTRAS[@]}"; then
            if git diff --quiet uv.lock 2>/dev/null; then
                echo -e "${GREEN}✅ Dependencies are in sync (fallback extras)${NC}"
            else
                echo -e "${GREEN}✅ Updated uv.lock to match pyproject.toml (fallback extras)${NC}"
                echo -e "${YELLOW}  Don't forget to commit the updated uv.lock file${NC}"
            fi
        else
            echo -e "${RED}❌ Failed to sync dependencies (both primary and fallback)${NC}"
            CHECKS_PASSED=false
        fi
    fi
else
    echo "  Checking if uv sync would modify uv.lock..."
    # Create a temporary copy of uv.lock
    cp uv.lock uv.lock.backup 2>/dev/null || touch uv.lock.backup
    
    # Try primary extras quietly
    if uv sync --quiet "${PRIMARY_EXTRAS[@]}" 2>/dev/null; then
        # Check if uv.lock was modified
        if diff -q uv.lock uv.lock.backup >/dev/null 2>&1; then
            echo -e "${GREEN}✅ uv.lock is up to date${NC}"
        else
            echo -e "${YELLOW}⚠️  uv.lock is out of sync with pyproject.toml${NC}"
            echo "  Run 'uv sync --all-extras' and commit the changes"
            CHECKS_PASSED=false
            # Restore the original uv.lock
            mv uv.lock.backup uv.lock
        fi
    else
        echo "  Primary check failed; trying fallback extras quietly..."
        if uv sync --quiet "${FALLBACK_EXTRAS[@]}" 2>/dev/null; then
            if diff -q uv.lock uv.lock.backup >/dev/null 2>&1; then
                echo -e "${GREEN}✅ uv.lock is up to date (checked with fallback extras)${NC}"
            else
                echo -e "${YELLOW}⚠️  uv.lock is out of sync with pyproject.toml (fallback extras)${NC}"
                echo "  Run 'uv sync --extra plotting --extra skypilot' and commit the changes"
                CHECKS_PASSED=false
                mv uv.lock.backup uv.lock
            fi
        else
            echo -e "${RED}❌ Failed to check dependencies (both primary and fallback)${NC}"
            CHECKS_PASSED=false
            # Restore the original uv.lock if it exists
            [ -f uv.lock.backup ] && mv uv.lock.backup uv.lock
        fi
    fi
    
    # Clean up backup file
    rm -f uv.lock.backup
fi
echo

# Summary
if $CHECKS_PASSED; then
    echo -e "${GREEN}🎉 All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some checks failed${NC}"
    if [[ -z "$FIX_FLAG" ]]; then
        # Show tips for each type of failure
        if $TYPECHECK_FAILED; then
            echo -e "💡 Tip: Type errors can't be auto-fixed by --fix. Re-run ${YELLOW}uv run pyright src${NC} to see full diagnostics."
        fi
        if $TESTS_FAILED; then
            if [[ -z "$VERBOSE_TEST_FAILURE" ]]; then
                echo -e "💡 Tip: Test failures can't be auto-fixed by --fix. Re-run ${YELLOW}uv run pytest --nbval --current-env tests/unit${NC} to see full test output."
                echo -e "        Or use ${YELLOW}./scripts/run_checks.sh --verbose-test-failure${NC} to see full output on failure."
            else
                echo -e "💡 Tip: Test failures can't be auto-fixed by --fix. Re-run ${YELLOW}uv run pytest --nbval --current-env tests/unit${NC} to debug."
            fi
        fi
        # Show general fix tip if there are failures but not type/test specific ones
        if ! $TYPECHECK_FAILED && ! $TESTS_FAILED; then
            echo -e "💡 Tip: Run ${YELLOW}./scripts/run_checks.sh --fix${NC} to automatically fix some issues"
        fi
    fi
    exit 1
fi