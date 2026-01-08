# SWE-Bench Test Results Analysis Report

**Date**: 2025-07-10  
**Total Instances Analyzed**: 8,480 (unique instances, latest run per instance)

## Executive Summary

### Overall Performance
- **Perfect Pass Rate**: 74.6% (6,329/8,480)
- **Test Failures**: 24.2% (2,055 instances)
- **Errors**: 1.5% (122 instances)

A "perfect pass" means:
- No errors during sandbox creation or test execution
- All FAIL_TO_PASS tests passed initially (baseline behavior)
- All FAIL_TO_PASS tests failed after patch (patch introduces intended bug)
- All PASS_TO_PASS tests passed (no regression)

## Key Findings

### 1. Most Common Failure Modes

| Failure Type | Count | Percentage | Description |
|-------------|-------|------------|-------------|
| P2P Failures | 1,758 | 20.7% | Patches broke existing functionality |
| F2P Initial Failures | 1,171 | 13.8% | Tests failed baseline (should pass initially) |
| F2P Post Pass | 182 | 2.1% | Tests still passed after patch (should fail) |
| Execution Errors | 122 | 1.5% | Timeouts, sandbox failures, etc. |

### 2. Repositories with Fundamental Issues (0% Perfect Pass Rate)

15 repositories achieved 0% perfect pass rate, indicating systematic problems:

#### P2P-Dominant Failures (patches consistently break existing tests):
- **oauthlib**: 164/166 instances fail P2P (98.8%)
- **cloudpipe**: 37/37 instances fail P2P (100%)
- **pyparsing**: 28/29 instances fail P2P (96.6%)
- **life4**: 40/43 instances fail P2P (93.0%)
- **borntyping**: 16/17 instances fail P2P (94.1%)

#### F2P-Initial Dominant Failures (baseline tests don't pass):
- **Project-MONAI**: 159/161 instances fail F2P-initial (98.8%)
- **burnash**: 26/26 instances fail F2P-initial (100%)
- **python-trio**: 90/139 instances fail F2P-initial (64.7%)
- **Cog-Creators**: 64/91 instances fail F2P-initial (70.3%)
- **alanjds**: 18/27 instances fail F2P-initial (66.7%)

#### Mixed Failures (both baseline and regression issues):
- **seperman**: 115 F2P-initial + 170 P2P failures (172 instances)
- **django-money**: 67 F2P-initial + 67 P2P failures (68 instances)
- **facebookresearch**: 57 F2P-initial + 71 P2P failures (72 instances)
- **tweepy**: 52 F2P-initial + 52 P2P failures (52 instances)
- **aio-libs**: 8 F2P-initial + 8 P2P failures (8 instances)

### 3. Repository Performance Distribution

| Perfect Pass Rate | Repository Count | Percentage |
|------------------|------------------|------------|
| 0% | 15 | 14.3% |
| 1-25% | 5 | 4.8% |
| 26-50% | 4 | 3.8% |
| 51-75% | 3 | 2.9% |
| 76-99% | 26 | 24.8% |
| 100% | 52 | 49.5% |

### 4. Error Analysis

Most errors (1.5% of instances) were infrastructure-related:
- 504 Gateway Timeout: 76 instances
- Command execution timeout: 64 instances
- Empty command errors: 31 instances
- Sandbox creation failures: 10 instances

## Recommendations

### High Priority Investigations

1. **oauthlib & cloudpipe**: Near 100% P2P failure rate suggests patches consistently break core functionality. These need immediate review of patch generation logic.

2. **Project-MONAI & burnash**: 100% F2P-initial failure rate indicates test specifications may be incorrect or tests are not properly capturing baseline behavior.

3. **seperman**: Highest absolute number of failures (170 P2P failures) combined with F2P-initial failures suggests both test specification and patch generation issues.

### System Improvements

1. **Test Specification Review**: 13.8% F2P-initial failure rate suggests many test specifications don't correctly capture baseline behavior.

2. **Patch Quality**: 20.7% P2P failure rate indicates patches often break existing functionality. Consider adding regression checks.

3. **Infrastructure**: The 8,000 test limit resolved previous issues with test count limits.

## Success Stories

- 52 repositories (49.5%) achieved 100% perfect pass rate
- 74.6% overall perfect pass rate shows the system works well for most cases
- Very low error rate (1.5%) indicates stable infrastructure

## Technical Notes

- Analysis based on latest run per instance (aggregated by instance_id)
- Used daytona provider with up to 128 concurrent sandboxes
- Retry logic with exponential backoff for transient failures
- Test count limit increased from 3,000 to 8,000