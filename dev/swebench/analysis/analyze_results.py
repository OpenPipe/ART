#!/usr/bin/env python3
"""
Comprehensive analysis of SWE-bench test results using Polars.
Aggregates by instance_id, taking the last occurrence.
"""

import json
import re
from collections import Counter

import polars as pl


def load_test_results(
    file_path: str, aggregate_by_instance: bool = True
) -> pl.DataFrame:
    """Load and aggregate test results by instance_id."""
    # Read JSONL file and flatten the nested structure
    flattened_data = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            flattened = {
                "instance_id": record.get("instance_id"),
                "repo": record.get("repo"),
                "timestamp": record.get("timestamp"),
                "provider": record.get("provider"),
                "error": record.get("error"),
                "f2p_initial_failed": record.get("f2p_initial", {}).get("failed"),
                "f2p_initial_passed": record.get("f2p_initial", {}).get("passed"),
                "f2p_initial_error": record.get("f2p_initial", {}).get("error"),
                "f2p_post_failed": record.get("f2p_post_patch", {}).get("failed"),
                "f2p_post_passed": record.get("f2p_post_patch", {}).get("passed"),
                "f2p_post_error": record.get("f2p_post_patch", {}).get("error"),
                "p2p_failed": record.get("p2p", {}).get("failed"),
                "p2p_passed": record.get("p2p", {}).get("passed"),
                "p2p_error": record.get("p2p", {}).get("error"),
                "_line_number": i,
            }
            flattened_data.append(flattened)

    # Create DataFrame with explicit schema
    schema = {
        "instance_id": pl.Utf8,
        "repo": pl.Utf8,
        "timestamp": pl.Utf8,
        "provider": pl.Utf8,
        "error": pl.Utf8,
        "f2p_initial_failed": pl.Int64,
        "f2p_initial_passed": pl.Int64,
        "f2p_initial_error": pl.Utf8,
        "f2p_post_failed": pl.Int64,
        "f2p_post_passed": pl.Int64,
        "f2p_post_error": pl.Utf8,
        "p2p_failed": pl.Int64,
        "p2p_passed": pl.Int64,
        "p2p_error": pl.Utf8,
        "_line_number": pl.Int64,
    }

    df = pl.DataFrame(flattened_data, schema=schema)

    if aggregate_by_instance:
        # Take the last occurrence of each instance_id
        df = df.sort("_line_number").group_by("instance_id").last()

    return df.drop("_line_number")


def print_basic_stats(df: pl.DataFrame):
    """Print basic statistics about the dataset."""
    print(f"\nTotal Unique Instances: {len(df):,}")

    # Count instances with errors
    has_error = df.filter(pl.col("error").is_not_null())
    print(
        f"Instances with top-level errors: {len(has_error):,} ({len(has_error) / len(df) * 100:.1f}%)"
    )

    # Count instances with test errors
    has_test_error = df.filter(
        (pl.col("f2p_initial_error").is_not_null())
        | (pl.col("f2p_post_error").is_not_null())
        | (pl.col("p2p_error").is_not_null())
    )
    print(
        f"Instances with test errors: {len(has_test_error):,} ({len(has_test_error) / len(df) * 100:.1f}%)"
    )


def analyze_perfect_passes(df: pl.DataFrame) -> dict:
    """Analyze instances that passed all tests perfectly."""
    # Perfect means:
    # 1. No errors at all
    # 2. All F2P tests passed initially (as expected)
    # 3. All F2P tests failed after patch (as expected)
    # 4. All P2P tests passed (as expected)
    perfect_passes = df.filter(
        (pl.col("error").is_null())
        & (pl.col("f2p_initial_error").is_null())
        & (pl.col("f2p_post_error").is_null())
        & (pl.col("p2p_error").is_null())
        & (pl.col("f2p_initial_failed") == 0)  # All F2P should pass initially
        & (pl.col("f2p_post_passed") == 0)  # All F2P should fail after patch
        & (pl.col("p2p_failed") == 0)  # All P2P should pass
    )

    return {
        "count": len(perfect_passes),
        "percentage": len(perfect_passes) / len(df) * 100,
        "instances": perfect_passes.select("instance_id").to_series().to_list(),
    }


def analyze_detailed_failure_modes(df: pl.DataFrame) -> dict:
    """Analyze detailed failure modes and patterns."""
    results = {}

    # Categorize all instances
    perfect = df.filter(
        (pl.col("error").is_null())
        & (pl.col("f2p_initial_error").is_null())
        & (pl.col("f2p_post_error").is_null())
        & (pl.col("p2p_error").is_null())
        & (pl.col("f2p_initial_failed") == 0)  # All F2P should pass initially
        & (pl.col("f2p_post_passed") == 0)  # All F2P should fail after patch
        & (pl.col("p2p_failed") == 0)  # All P2P should pass
    )

    # Has any error
    has_error = df.filter(pl.col("error").is_not_null())

    # Test failures (no error but some tests failed)
    test_failures = df.filter(
        (pl.col("error").is_null())
        & (
            (
                pl.col("f2p_initial_failed") > 0
            )  # Some F2P failed initially (bad - should pass)
            | (
                pl.col("f2p_post_passed") > 0
            )  # Some F2P passed after patch (bad - should fail)
            | (pl.col("p2p_failed") > 0)  # Some P2P failed (bad)
        )
    )

    # Categorize test failures
    f2p_initial_wrong = test_failures.filter(pl.col("f2p_initial_failed") > 0)
    f2p_post_wrong = test_failures.filter(pl.col("f2p_post_passed") > 0)
    p2p_wrong = test_failures.filter(pl.col("p2p_failed") > 0)

    # Test errors (errors during test execution)
    test_errors = df.filter(
        (pl.col("error").is_null())
        & (
            (pl.col("f2p_initial_error").is_not_null())
            | (pl.col("f2p_post_error").is_not_null())
            | (pl.col("p2p_error").is_not_null())
        )
    )

    results["categories"] = {
        "perfect": {"count": len(perfect), "pct": len(perfect) / len(df) * 100},
        "has_error": {"count": len(has_error), "pct": len(has_error) / len(df) * 100},
        "test_failures": {
            "count": len(test_failures),
            "pct": len(test_failures) / len(df) * 100,
        },
        "test_errors": {
            "count": len(test_errors),
            "pct": len(test_errors) / len(df) * 100,
        },
    }

    results["test_failure_breakdown"] = {
        "f2p_initial_failed_when_should_pass": len(f2p_initial_wrong),
        "f2p_post_passed_when_should_fail": len(f2p_post_wrong),
        "p2p_failed_when_should_pass": len(p2p_wrong),
    }

    # Analyze common error messages
    error_messages = []
    for col in ["error", "f2p_initial_error", "f2p_post_error", "p2p_error"]:
        errors = df.filter(pl.col(col).is_not_null()).select(col).to_series().to_list()
        error_messages.extend(errors)

    # Extract error patterns
    error_counter = Counter()
    for error in error_messages:
        if error:
            # Try to extract the main error type
            if "More than" in error and "tests" in error:
                match = re.search(r"More than (\d+) tests", error)
                if match:
                    error_counter[f"More than {match.group(1)} tests"] += 1
            elif "No available runners" in error:
                error_counter["No available runners"] += 1
            elif "timeout" in error.lower():
                error_counter["Timeout error"] += 1
            elif "pytest" in error:
                # Extract pytest error type
                if "not found" in error:
                    error_counter["Pytest: file/test not found"] += 1
                elif "FAILED" in error:
                    error_counter["Pytest: test execution failed"] += 1
                else:
                    error_counter["Pytest: other error"] += 1
            else:
                # Try to get first line of error
                first_line = error.split("\n")[0][:100]
                error_counter[first_line] += 1

    results["top_errors"] = error_counter.most_common(20)

    return results


def analyze_repositories(df: pl.DataFrame) -> dict:
    """Analyze error rates by repository."""
    # Extract repository name from instance_id
    df = df.with_columns(
        pl.col("instance_id").str.extract(r"^([^_]+)", 1).alias("repo")
    )

    # Calculate stats by repo
    repo_stats = []
    for repo in df.select("repo").unique().to_series().to_list():
        if repo:
            repo_df = df.filter(pl.col("repo") == repo)
            total = len(repo_df)
            with_errors = len(repo_df.filter(pl.col("error").is_not_null()))

            repo_stats.append(
                {
                    "repo": repo,
                    "total": total,
                    "with_errors": with_errors,
                    "error_rate": with_errors / total * 100 if total > 0 else 0,
                }
            )

    # Sort by error rate
    by_error_rate = sorted(repo_stats, key=lambda x: x["error_rate"], reverse=True)
    by_count = sorted(repo_stats, key=lambda x: x["total"], reverse=True)

    return {
        "by_error_rate": [(r["repo"], r["error_rate"]) for r in by_error_rate],
        "by_count": [(r["repo"], r["total"]) for r in by_count],
        "full_stats": by_error_rate,
    }


def analyze_failure_patterns(df: pl.DataFrame) -> dict:
    """Analyze specific failure patterns."""
    # Count different types of failures
    f2p_initial_failures = len(
        df.filter((pl.col("error").is_null()) & (pl.col("f2p_initial_failed") > 0))
    )

    f2p_post_failures = len(
        df.filter((pl.col("error").is_null()) & (pl.col("f2p_post_passed") > 0))
    )

    p2p_failures = len(
        df.filter((pl.col("error").is_null()) & (pl.col("p2p_failed") > 0))
    )

    # Get specific instances with P2P failures
    p2p_failure_instances = (
        df.filter((pl.col("error").is_null()) & (pl.col("p2p_failed") > 0))
        .select("instance_id")
        .to_series()
        .to_list()
    )

    return {
        "f2p_initial_failures": f2p_initial_failures,
        "f2p_post_failures": f2p_post_failures,
        "p2p_failures": p2p_failures,
        "p2p_failure_instances": p2p_failure_instances,
    }


def extract_unique_errors(df: pl.DataFrame):
    """Extract and save unique error messages."""
    all_errors = []

    for col in ["error", "f2p_initial_error", "f2p_post_error", "p2p_error"]:
        errors = df.filter(pl.col(col).is_not_null()).select(
            [pl.col("instance_id"), pl.col(col).alias("error_msg")]
        )
        for row in errors.iter_rows(named=True):
            all_errors.append((row["instance_id"], col, row["error_msg"]))

    # Group by error message
    error_groups = {}
    for instance_id, error_type, error_msg in all_errors:
        if error_msg not in error_groups:
            error_groups[error_msg] = []
        error_groups[error_msg].append((instance_id, error_type))

    # Save to file
    with open("unique_errors.txt", "w") as f:
        f.write(f"Unique Error Messages: {len(error_groups)}\n")
        f.write("=" * 80 + "\n\n")

        # Sort by frequency
        sorted_errors = sorted(
            error_groups.items(), key=lambda x: len(x[1]), reverse=True
        )

        for error_msg, instances in sorted_errors:
            f.write(f"Error ({len(instances)} occurrences):\n")
            f.write("-" * 40 + "\n")
            f.write(f"{error_msg}\n")
            f.write("\nAffected instances:\n")
            for inst, err_type in instances[:5]:  # Show first 5
                f.write(f"  - {inst} ({err_type})\n")
            if len(instances) > 5:
                f.write(f"  ... and {len(instances) - 5} more\n")
            f.write("\n" + "=" * 80 + "\n\n")


def analyze_test_counts(df: pl.DataFrame) -> dict:
    """Analyze instances with high test counts."""
    # Look for errors mentioning test count limits
    test_count_errors = df.filter(
        pl.col("error").str.contains("More than.*tests")
        | pl.col("f2p_initial_error").str.contains("More than.*tests")
        | pl.col("f2p_post_error").str.contains("More than.*tests")
        | pl.col("p2p_error").str.contains("More than.*tests")
    )

    # Extract test counts from error messages
    high_test_instances = []
    for row in test_count_errors.iter_rows(named=True):
        instance_id = row["instance_id"]
        for col in ["error", "f2p_initial_error", "f2p_post_error", "p2p_error"]:
            if row[col] and "More than" in row[col]:
                match = re.search(r"More than (\d+) tests", row[col])
                if match:
                    test_count = int(match.group(1))
                    high_test_instances.append((instance_id, test_count))
                    break

    # Sort by test count
    high_test_instances.sort(key=lambda x: x[1], reverse=True)

    return {
        "high_test_count_instances": high_test_instances,
        "total_affected": len(test_count_errors),
    }


def main():
    # Load data with aggregation
    df = load_test_results("instance_test_results.jsonl", aggregate_by_instance=True)

    # Basic stats
    print("=" * 80)
    print("SWE-bench Test Results Analysis (Latest Run Per Instance)")
    print("=" * 80)
    print_basic_stats(df)

    # Detailed failure mode analysis
    failure_modes = analyze_detailed_failure_modes(df)

    print("\n" + "=" * 80)
    print("INSTANCE CATEGORIZATION")
    print("=" * 80)

    for category, stats in failure_modes["categories"].items():
        print(
            f"{category.replace('_', ' ').title():25} {stats['count']:6,} ({stats['pct']:5.1f}%)"
        )

    print("\n" + "=" * 80)
    print("PERFECT PASS ANALYSIS")
    print("=" * 80)

    # Perfect passes
    perfect = analyze_perfect_passes(df)
    print(
        f"\nPerfect Passes: {perfect['count']:,} / {len(df):,} ({perfect['percentage']:.1f}%)"
    )
    print("\nPerfect means:")
    print("  - No errors during sandbox creation or test execution")
    print("  - All FAIL_TO_PASS tests passed initially (as expected)")
    print(
        "  - All FAIL_TO_PASS tests failed after patch (as expected - patch introduces bug)"
    )
    print("  - All PASS_TO_PASS tests passed (as expected)")

    print("\n" + "=" * 80)
    print("TEST FAILURE BREAKDOWN (No Errors, But Wrong Results)")
    print("=" * 80)

    for failure_type, count in failure_modes["test_failure_breakdown"].items():
        print(f"{failure_type.replace('_', ' ').title():50} {count:6,}")

    print("\n" + "=" * 80)
    print("TOP ERROR PATTERNS")
    print("=" * 80)

    for i, (error, count) in enumerate(failure_modes["top_errors"], 1):
        print(f"{i:2}. {error[:70]:70} {count:6,}")

    # Repository analysis
    repo_stats = analyze_repositories(df)
    print("\n" + "=" * 80)
    print("REPOSITORY ANALYSIS")
    print("=" * 80)

    print("\nTop 10 Repositories by Error Rate:")
    for repo, rate in repo_stats["by_error_rate"][:10]:
        total = df.filter(pl.col("instance_id").str.contains(repo)).height
        errors = df.filter(
            (pl.col("instance_id").str.contains(repo)) & (pl.col("error").is_not_null())
        ).height
        print(f"  {repo:40} {rate:5.1f}% ({errors}/{total})")

    print("\n" + "=" * 80)
    print("SPECIFIC INSTANCE ANALYSIS")
    print("=" * 80)

    # Find instances that consistently fail P2P
    p2p_failures = df.filter((pl.col("error").is_null()) & (pl.col("p2p_failed") > 0))

    if len(p2p_failures) > 0:
        print(f"\nInstances with P2P failures: {len(p2p_failures)}")
        # Group by repository
        p2p_by_repo = Counter()
        for instance in p2p_failures.select("instance_id").to_series().to_list():
            repo = instance.split("__")[0]
            p2p_by_repo[repo] += 1

        print("\nP2P failures by repository:")
        for repo, count in p2p_by_repo.most_common():
            print(f"  {repo:40} {count:3} instances")

    # Test count analysis
    test_count_stats = analyze_test_counts(df)
    if test_count_stats["high_test_count_instances"]:
        print(
            f"\n\nInstances with >3000 tests: {len(test_count_stats['high_test_count_instances'])}"
        )
        print(
            f"Instances with >8000 tests: {len([i for i, c in test_count_stats['high_test_count_instances'] if c > 8000])}"
        )

    # Extract unique errors
    extract_unique_errors(df)
    print("\n\nDetailed error messages saved to 'unique_errors.txt'")


if __name__ == "__main__":
    main()
