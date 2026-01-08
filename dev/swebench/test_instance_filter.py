#!/usr/bin/env python3
"""Test the instance filter functionality."""

from instance_filter import (
    filter_quality_instances,
    get_quality_instance_ids,
    load_test_results,
    save_quality_instance_list,
)
from instances import get_filtered_swe_smith_instances_df


def main():
    print("Testing instance filter functionality...")

    # Load test results
    print("\n1. Loading test results...")
    test_results = load_test_results()
    print(f"   Loaded {len(test_results)} instance results")

    # Sample a few results
    print("\n   Sample results:")
    for i, (instance_id, result) in enumerate(list(test_results.items())[:3]):
        print(f"   - {instance_id}:")
        if "error" in result and result["error"]:
            print(f"     Error: {result['error'][:50]}...")
        else:
            f2p_init = result.get("f2p_initial", {})
            f2p_post = result.get("f2p_post_patch", {})
            p2p = result.get("p2p", {})
            print(
                f"     F2P initial: passed={f2p_init.get('passed')}, failed={f2p_init.get('failed')}"
            )
            print(
                f"     F2P post-patch: passed={f2p_post.get('passed')}, failed={f2p_post.get('failed')}"
            )
            print(f"     P2P: passed={p2p.get('passed')}, failed={p2p.get('failed')}")

    # Load instances
    print("\n2. Loading SWE-bench instances...")
    instances_df = get_filtered_swe_smith_instances_df()
    print(f"   Loaded {len(instances_df)} instances")

    # Get quality instances
    print("\n3. Finding quality instances...")
    quality_ids = get_quality_instance_ids(instances_df)
    print(f"   Found {len(quality_ids)} quality instances")

    # Filter with non-zero test requirement
    quality_ids_nonzero = get_quality_instance_ids(
        instances_df, require_non_zero_tests=True
    )
    print(f"   Found {len(quality_ids_nonzero)} quality instances with non-zero tests")

    # Filter DataFrame
    print("\n4. Filtering DataFrame...")
    quality_df = filter_quality_instances(instances_df)
    print(f"   Filtered DataFrame has {len(quality_df)} instances")

    # Show some statistics
    print("\n5. Quality instance statistics:")
    print(f"   - Total instances: {len(instances_df)}")
    print(f"   - Quality instances (all): {len(quality_ids)}")
    print(f"   - Quality instances (non-zero tests): {len(quality_ids_nonzero)}")
    print(f"   - Percentage: {len(quality_ids_nonzero) / len(instances_df) * 100:.1f}%")

    # Sample some quality instances
    if len(quality_df) > 0:
        print("\n6. Sample quality instances:")
        for instance in quality_df.head(5).iter_rows(named=True):
            f2p_count = len(instance.get("FAIL_TO_PASS", []))
            p2p_count = len(instance.get("PASS_TO_PASS", []))
            print(f"   - {instance['instance_id']} ({instance['repo']})")
            print(f"     F2P tests: {f2p_count}, P2P tests: {p2p_count}")

    # Save list
    print("\n7. Saving quality instance list...")
    count = save_quality_instance_list(instances_df, "quality_instances.txt")
    print(f"   Saved {count} instance IDs to quality_instances.txt")

    # Show repository distribution
    if len(quality_df) > 0:
        print("\n8. Repository distribution (top 10):")
        repo_counts = (
            quality_df.group_by("repo").count().sort("count", descending=True).head(10)
        )
        for row in repo_counts.iter_rows(named=True):
            print(f"   - {row['repo']}: {row['count']} instances")


if __name__ == "__main__":
    main()
