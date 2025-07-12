#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
FIX_FLAG=""
if [[ "$1" == "--fix" ]]; then
    FIX_FLAG="--fix"
fi

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

# Check if uv.lock is in sync with pyproject.toml
echo "🔒 Checking if uv.lock is up to date..."
if [[ -n "$FIX_FLAG" ]]; then
    echo "  Running: uv sync"
    if uv sync; then
        # Check if uv.lock was modified
        if git diff --quiet uv.lock 2>/dev/null; then
            echo -e "${GREEN}✅ Dependencies are in sync${NC}"
        else
            echo -e "${GREEN}✅ Updated uv.lock to match pyproject.toml${NC}"
            echo -e "${YELLOW}  Don't forget to commit the updated uv.lock file${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to sync dependencies${NC}"
        CHECKS_PASSED=false
    fi
else
    echo "  Checking if uv sync would modify uv.lock..."
    # Create a temporary copy of uv.lock
    cp uv.lock uv.lock.backup 2>/dev/null || touch uv.lock.backup
    
    # Run uv sync quietly
    if uv sync --quiet 2>/dev/null; then
        # Check if uv.lock was modified
        if diff -q uv.lock uv.lock.backup >/dev/null 2>&1; then
            echo -e "${GREEN}✅ uv.lock is up to date${NC}"
        else
            echo -e "${YELLOW}⚠️  uv.lock is out of sync with pyproject.toml${NC}"
            echo "  Run 'uv sync' and commit the changes"
            CHECKS_PASSED=false
            # Restore the original uv.lock
            mv uv.lock.backup uv.lock
        fi
    else
        echo -e "${RED}❌ Failed to check dependencies${NC}"
        CHECKS_PASSED=false
        # Restore the original uv.lock if it exists
        [ -f uv.lock.backup ] && mv uv.lock.backup uv.lock
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
        echo -e "💡 Tip: Run ${YELLOW}./scripts/run_checks.sh --fix${NC} to automatically fix some issues"
    fi
    exit 1
fi