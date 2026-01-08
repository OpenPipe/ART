"""
Filter SWE-bench instances based on test results to identify high-quality training instances.

Filters instances that meet all of the following criteria:
1. f2p_initial.passed == len(f2p) and f2p_initial.failed == 0
2. f2p_post_patch.failed == len(f2p) and f2p_post_patch.passed == 0
3. p2p.failed == 0 and p2p.passed == len(p2p)
"""

import json
from collections import defaultdict
from typing import Dict, Set

import polars as pl


def load_test_results(
    results_file: str = "analysis/instance_test_results.jsonl",
) -> Dict[str, dict]:
    """
    Load test results from JSONL file and return the most optimistic result for each instance.

    Args:
        results_file: Path to the test results JSONL file

    Returns:
        Dictionary mapping instance_id to test results
    """
    results_by_instance = defaultdict(list)

    # Load all results
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    results_by_instance[result["instance_id"]].append(result)
                except json.JSONDecodeError:
                    continue

    # Select most optimistic result for each instance
    optimistic_results = {}
    for instance_id, results in results_by_instance.items():
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r or not r["error"]]

        if not valid_results:
            continue

        # Select the result where tests pass best
        best_result = max(
            valid_results,
            key=lambda r: (
                (r.get("f2p_initial", {}).get("passed", 0) or 0)
                + (r.get("p2p", {}).get("passed", 0) or 0)
                - (r.get("f2p_post_patch", {}).get("passed", 0) or 0)
            ),
        )

        optimistic_results[instance_id] = best_result

    return optimistic_results


def check_instance_quality(result: dict, instance: dict) -> bool:
    """
    Check if an instance meets the quality criteria.

    Args:
        result: Test result from instance_test_results.jsonl
        instance: Instance data with FAIL_TO_PASS and PASS_TO_PASS test lists

    Returns:
        True if instance meets all quality criteria
    """
    # Check for error
    if "error" in result and result["error"]:
        return False

    # Get test counts
    f2p_count = len(instance.get("FAIL_TO_PASS", []))
    p2p_count = len(instance.get("PASS_TO_PASS", []))

    # Get test results
    f2p_initial = result.get("f2p_initial", {})
    f2p_post_patch = result.get("f2p_post_patch", {})
    p2p = result.get("p2p", {})

    # Check criterion 1: f2p_initial.passed == len(f2p) and f2p_initial.failed == 0
    if f2p_initial.get("passed") != f2p_count or f2p_initial.get("failed") != 0:
        return False

    # Check criterion 2: f2p_post_patch.failed == len(f2p) and f2p_post_patch.passed == 0
    if f2p_post_patch.get("failed") != f2p_count or f2p_post_patch.get("passed") != 0:
        return False

    # Check criterion 3: p2p.failed == 0 and p2p.passed == len(p2p)
    if p2p.get("failed") != 0 or p2p.get("passed") != p2p_count:
        return False

    return True


def get_quality_instance_ids(
    instances_df: pl.DataFrame,
    results_file: str = "analysis/instance_test_results.jsonl",
    require_non_zero_tests: bool = True,
) -> Set[str]:
    """
    Get set of instance IDs that meet quality criteria.

    Args:
        instances_df: DataFrame of SWE-bench instances
        results_file: Path to test results file
        require_non_zero_tests: If True, exclude instances with zero tests

    Returns:
        Set of instance IDs that meet quality criteria
    """
    # Load test results
    test_results = load_test_results(results_file)

    # Check each instance
    quality_instances = set()

    for instance in instances_df.iter_rows(named=True):
        instance_id = instance["instance_id"]

        # Skip if no test results
        if instance_id not in test_results:
            continue

        # Skip if requiring non-zero tests and instance has none
        if require_non_zero_tests:
            f2p_count = len(instance.get("FAIL_TO_PASS", []))
            p2p_count = len(instance.get("PASS_TO_PASS", []))
            if f2p_count == 0 and p2p_count == 0:
                continue

        # Check quality criteria
        if check_instance_quality(test_results[instance_id], instance):
            quality_instances.add(instance_id)

    return quality_instances


def filter_quality_instances(
    instances_df: pl.DataFrame,
    results_file: str = "analysis/instance_test_results.jsonl",
    require_non_zero_tests: bool = True,
) -> pl.DataFrame:
    """
    Filter instances DataFrame to only include quality instances.

    Args:
        instances_df: DataFrame of SWE-bench instances
        results_file: Path to test results file
        require_non_zero_tests: If True, exclude instances with zero tests

    Returns:
        Filtered DataFrame with only quality instances
    """
    quality_ids = get_quality_instance_ids(
        instances_df, results_file, require_non_zero_tests
    )
    return instances_df.filter(pl.col("instance_id").is_in(list(quality_ids)))


def save_quality_instance_list(
    instances_df: pl.DataFrame,
    output_file: str = "quality_instances.txt",
    results_file: str = "analysis/instance_test_results.jsonl",
    require_non_zero_tests: bool = True,
) -> int:
    """
    Save list of quality instance IDs to a text file.

    Args:
        instances_df: DataFrame of SWE-bench instances
        output_file: Path to output file
        results_file: Path to test results file
        require_non_zero_tests: If True, exclude instances with zero tests

    Returns:
        Number of quality instances found
    """
    quality_ids = get_quality_instance_ids(
        instances_df, results_file, require_non_zero_tests
    )

    with open(output_file, "w") as f:
        for instance_id in sorted(quality_ids):
            f.write(f"{instance_id}\n")

    return len(quality_ids)


if __name__ == "__main__":
    # Example usage
    from instances import get_filtered_swe_smith_instances_df

    print("Loading instances...")
    instances_df = get_filtered_swe_smith_instances_df()

    print("Finding quality instances...")
    quality_instances = filter_quality_instances(instances_df)

    print(
        f"Found {len(quality_instances)} quality instances out of {len(instances_df)} total"
    )

    # Save the list
    count = save_quality_instance_list(instances_df, "quality_instances.txt")
    print(f"Saved {count} quality instance IDs to quality_instances.txt")

    # Show some statistics
    print("\nQuality instance statistics:")
    print(f"- Total instances: {len(instances_df)}")
    print(f"- Quality instances: {len(quality_instances)}")
    print(f"- Percentage: {len(quality_instances) / len(instances_df) * 100:.1f}%")

    # Sample a few
    if len(quality_instances) > 0:
        print("\nSample quality instances:")
        for _, instance in quality_instances.head(5).iterrows():
            print(f"- {instance['instance_id']} ({instance['repo']})")
