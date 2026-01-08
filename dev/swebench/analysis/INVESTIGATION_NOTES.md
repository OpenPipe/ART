# SWE-bench Investigation Notes

**Last Updated**: 2025-07-10  
**Analysis Summary**: 74.6% perfect pass rate (6,329/8,480 instances)

## Critical Issues: Repositories with 0% Perfect Pass Rate

### 1. P2P-Dominant Failures (Patches Break Existing Tests)

These repositories have near-100% P2P failure rates, meaning patches consistently break existing functionality:

#### **oauthlib** (164/166 instances fail P2P - 98.8%)
- **Pattern**: All patches break existing OAuth functionality tests
- **Example instances**:
  - `oauthlib__oauthlib.1fd52536.lm_rewrite__q9ve64pd`: 670 P2P failures
  - `oauthlib__oauthlib.1fd52536.lm_rewrite__rchxmbd6`: 625 P2P failures
- **Hypothesis**: OAuth library may have tightly coupled components where any change breaks auth flows

#### **cloudpipe** (37/37 instances fail P2P - 100%)
- **Pattern**: Every single patch breaks existing tests
- **Hypothesis**: Likely has integration tests that are sensitive to any code changes

#### **pyparsing** (28/29 instances fail P2P - 96.6%)
- **Pattern**: Parser modifications break existing parsing tests
- **Hypothesis**: Grammar/parser changes have cascading effects

### 2. F2P-Initial Failures (Baseline Tests Don't Pass)

These repositories have tests that fail even before applying patches:

#### **Project-MONAI** (159/161 instances fail F2P-initial - 98.8%)
- **Pattern**: Tests don't pass in baseline state
- **Hypothesis**: May require specific GPU/CUDA setup or have incorrect test specifications

#### **burnash** (26/26 instances fail F2P-initial - 100%)
- **Pattern**: All baseline tests fail
- **Hypothesis**: Test specifications may be incorrect or environment setup issues

#### **python-trio** (90/139 instances fail F2P-initial - 64.7%)
- **Pattern**: Mix of F2P-initial (90) and P2P (125) failures
- **Example**: `python-trio__trio.cfbbe2c1.pr_2937`: 730 F2P initial failures
- **Hypothesis**: Async testing framework may have special requirements

### 3. Mixed Failure Patterns

#### **seperman** (170 P2P + 115 F2P-initial failures out of 172 instances)
- **Pattern**: Both baseline and regression failures
- **Hypothesis**: Fundamental test environment or specification issues

#### **django-money**, **facebookresearch**, **tweepy**, **aio-libs**
- All show similar patterns of both F2P-initial and P2P failures
- Suggests both incorrect test specs AND patches that break functionality

## Root Cause Analysis

### Environmental Issues
1. **GPU/CUDA Requirements**: Project-MONAI likely needs GPU setup
2. **Async Test Runners**: python-trio, aio-libs may need special async test configurations
3. **Database/Services**: django-money might need database setup

### Test Specification Issues
1. **Incorrect Baseline**: F2P tests that fail initially suggest wrong test selections
2. **Version Mismatches**: Tests may be from different versions than the code

### Patch Quality Issues
1. **Over-broad Changes**: Patches might modify more than intended
2. **Missing Context**: Patches may not account for all usages of modified code

## Debugging Strategy

### Quick Checks
```bash
# Check a specific failing instance
uv run python analyze_results.py | grep "oauthlib__oauthlib"

# Look at error patterns for a specific repo
grep "oauthlib" unique_errors.txt

# Run a single instance manually
async with new_sandbox(image="swesmith/oauthlib__oauthlib.1fd52536", provider="daytona") as sandbox:
    failed, passed = await sandbox.eval(instance["FAIL_TO_PASS"])
```

### Deep Investigation Steps
1. **Pick one instance from each failure category**
2. **Run with verbose logging to see actual test output**
3. **Compare working vs failing repos to identify patterns**
4. **Check if special test commands or setup is needed**

## Recommendations for Future Work

### High Priority
1. **oauthlib**: Investigate why ALL patches break OAuth flows
2. **Project-MONAI**: Check GPU requirements and test specifications
3. **seperman**: High volume of failures (170) makes this impactful to fix

### Medium Priority
1. **python-trio**: Mixed failures suggest complex issues
2. **cloudpipe**: Small repo but 100% failure rate is concerning

### System Improvements
1. Add pre-flight checks for GPU/special requirements
2. Validate test specifications before running
3. Add regression test validation to patch generation
4. Consider repo-specific configuration overrides

## Success Metrics
- Current: 74.6% perfect pass rate
- Goal: >85% by addressing top 5 problematic repos
- 52 repos already at 100% - system works well for standard cases