#!/usr/bin/env python3
"""
Analyze perfect pass rates by repository to identify those with fundamental issues.
"""

import json

import polars as pl


def load_and_analyze_by_repo():
    """Load test results and analyze perfect pass rates by repository."""

    # Read JSONL file and flatten the nested structure
    flattened_data = []
    with open("instance_test_results.jsonl", "r") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            flattened = {
                "instance_id": record.get("instance_id"),
                "repo": record.get("repo"),
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

    # Create DataFrame
    schema = {
        "instance_id": pl.Utf8,
        "repo": pl.Utf8,
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

    # Take the last occurrence of each instance_id
    df = df.sort("_line_number").group_by("instance_id").last().drop("_line_number")

    # Extract repository name from instance_id
    df = df.with_columns(
        pl.col("instance_id").str.extract(r"^([^_]+)", 1).alias("repo_name")
    )

    # Define perfect pass criteria
    perfect_mask = (
        (pl.col("error").is_null())
        & (pl.col("f2p_initial_error").is_null())
        & (pl.col("f2p_post_error").is_null())
        & (pl.col("p2p_error").is_null())
        & (pl.col("f2p_initial_failed") == 0)  # All F2P should pass initially
        & (pl.col("f2p_post_passed") == 0)  # All F2P should fail after patch
        & (pl.col("p2p_failed") == 0)  # All P2P should pass
    )

    # Calculate stats by repository
    repo_stats = []
    for repo_name in df.select("repo_name").unique().to_series().to_list():
        if repo_name:
            repo_df = df.filter(pl.col("repo_name") == repo_name)
            total = len(repo_df)
            perfect = len(repo_df.filter(perfect_mask))

            # Count different failure types
            with_errors = len(repo_df.filter(pl.col("error").is_not_null()))
            f2p_initial_failures = len(
                repo_df.filter(
                    (pl.col("error").is_null()) & (pl.col("f2p_initial_failed") > 0)
                )
            )
            f2p_post_failures = len(
                repo_df.filter(
                    (pl.col("error").is_null()) & (pl.col("f2p_post_passed") > 0)
                )
            )
            p2p_failures = len(
                repo_df.filter((pl.col("error").is_null()) & (pl.col("p2p_failed") > 0))
            )

            repo_stats.append(
                {
                    "repo": repo_name,
                    "total": total,
                    "perfect": perfect,
                    "perfect_rate": perfect / total * 100 if total > 0 else 0,
                    "error_count": with_errors,
                    "error_rate": with_errors / total * 100 if total > 0 else 0,
                    "f2p_initial_fail_count": f2p_initial_failures,
                    "f2p_initial_fail_rate": (
                        f2p_initial_failures / total * 100 if total > 0 else 0
                    ),
                    "f2p_post_fail_count": f2p_post_failures,
                    "f2p_post_fail_rate": (
                        f2p_post_failures / total * 100 if total > 0 else 0
                    ),
                    "p2p_fail_count": p2p_failures,
                    "p2p_fail_rate": p2p_failures / total * 100 if total > 0 else 0,
                }
            )

    # Sort by perfect pass rate (ascending - worst first)
    repo_stats.sort(key=lambda x: x["perfect_rate"])

    return repo_stats


def print_analysis(repo_stats):
    """Print detailed analysis of repository perfect pass rates."""

    print("=" * 120)
    print("REPOSITORY PERFECT PASS RATE ANALYSIS")
    print("=" * 120)
    print(f"\nTotal repositories analyzed: {len(repo_stats)}")

    # Overall stats
    total_instances = sum(r["total"] for r in repo_stats)
    total_perfect = sum(r["perfect"] for r in repo_stats)
    print(
        f"Overall perfect pass rate: {total_perfect}/{total_instances} ({total_perfect / total_instances * 100:.1f}%)"
    )

    # Repos with 0% perfect pass rate
    zero_perfect = [r for r in repo_stats if r["perfect_rate"] == 0]
    print(f"\nRepositories with 0% perfect pass rate: {len(zero_perfect)}")

    print("\n" + "=" * 120)
    print("BOTTOM 30 REPOSITORIES BY PERFECT PASS RATE")
    print("=" * 120)
    print(
        f"{'Repository':<25} {'Total':>6} {'Perfect':>7} {'Rate':>6} | {'Errors':>6} {'F2P-I':>6} {'F2P-P':>6} {'P2P':>6}"
    )
    print("-" * 120)

    for repo in repo_stats[:30]:
        print(
            f"{repo['repo']:<25} {repo['total']:>6} {repo['perfect']:>7} {repo['perfect_rate']:>5.1f}% | "
            f"{repo['error_count']:>6} {repo['f2p_initial_fail_count']:>6} "
            f"{repo['f2p_post_fail_count']:>6} {repo['p2p_fail_count']:>6}"
        )

    # Analyze failure patterns for worst performers
    print("\n" + "=" * 120)
    print("DETAILED ANALYSIS OF WORST PERFORMERS (0% PERFECT PASS RATE)")
    print("=" * 120)

    # Group by primary failure mode
    primary_failures = {
        "errors": [],
        "f2p_initial": [],
        "f2p_post": [],
        "p2p": [],
        "mixed": [],
    }

    for repo in zero_perfect:
        if repo["error_rate"] > 50:
            primary_failures["errors"].append(repo)
        elif repo["f2p_initial_fail_rate"] > 50:
            primary_failures["f2p_initial"].append(repo)
        elif repo["f2p_post_fail_rate"] > 50:
            primary_failures["f2p_post"].append(repo)
        elif repo["p2p_fail_rate"] > 50:
            primary_failures["p2p"].append(repo)
        else:
            primary_failures["mixed"].append(repo)

    print(f"\nPrimarily Error Issues: {len(primary_failures['errors'])} repos")
    for repo in primary_failures["errors"][:5]:
        print(
            f"  - {repo['repo']}: {repo['error_rate']:.1f}% error rate ({repo['error_count']}/{repo['total']})"
        )

    print(
        f"\nPrimarily F2P Initial Failures: {len(primary_failures['f2p_initial'])} repos"
    )
    for repo in primary_failures["f2p_initial"][:5]:
        print(
            f"  - {repo['repo']}: {repo['f2p_initial_fail_rate']:.1f}% F2P-initial fail rate ({repo['f2p_initial_fail_count']}/{repo['total']})"
        )

    print(f"\nPrimarily F2P Post Failures: {len(primary_failures['f2p_post'])} repos")
    for repo in primary_failures["f2p_post"][:5]:
        print(
            f"  - {repo['repo']}: {repo['f2p_post_fail_rate']:.1f}% F2P-post fail rate ({repo['f2p_post_fail_count']}/{repo['total']})"
        )

    print(f"\nPrimarily P2P Failures: {len(primary_failures['p2p'])} repos")
    for repo in sorted(primary_failures["p2p"], key=lambda x: x["total"], reverse=True)[
        :10
    ]:
        print(
            f"  - {repo['repo']}: {repo['p2p_fail_rate']:.1f}% P2P fail rate ({repo['p2p_fail_count']}/{repo['total']})"
        )

    print(f"\nMixed Failure Patterns: {len(primary_failures['mixed'])} repos")
    for repo in primary_failures["mixed"][:5]:
        print(
            f"  - {repo['repo']}: E={repo['error_rate']:.0f}%, F2P-I={repo['f2p_initial_fail_rate']:.0f}%, "
            f"F2P-P={repo['f2p_post_fail_rate']:.0f}%, P2P={repo['p2p_fail_rate']:.0f}%"
        )

    # Distribution analysis
    print("\n" + "=" * 120)
    print("PERFECT PASS RATE DISTRIBUTION")
    print("=" * 120)

    buckets = {"0%": 0, "1-25%": 0, "26-50%": 0, "51-75%": 0, "76-99%": 0, "100%": 0}

    for repo in repo_stats:
        rate = repo["perfect_rate"]
        if rate == 0:
            buckets["0%"] += 1
        elif rate <= 25:
            buckets["1-25%"] += 1
        elif rate <= 50:
            buckets["26-50%"] += 1
        elif rate <= 75:
            buckets["51-75%"] += 1
        elif rate < 100:
            buckets["76-99%"] += 1
        else:
            buckets["100%"] += 1

    for bucket, count in buckets.items():
        print(f"{bucket:<10}: {count:>3} repositories")


def main():
    repo_stats = load_and_analyze_by_repo()
    print_analysis(repo_stats)

    # Save detailed results
    with open("repo_perfect_pass_analysis.txt", "w") as f:
        f.write(
            "Repository,Total,Perfect,PerfectRate%,Errors,F2P-Initial-Fail,F2P-Post-Fail,P2P-Fail\n"
        )
        for repo in repo_stats:
            f.write(
                f"{repo['repo']},{repo['total']},{repo['perfect']},{repo['perfect_rate']:.1f},"
                f"{repo['error_count']},{repo['f2p_initial_fail_count']},"
                f"{repo['f2p_post_fail_count']},{repo['p2p_fail_count']}\n"
            )

    print("\n\nDetailed results saved to 'repo_perfect_pass_analysis.txt'")


if __name__ == "__main__":
    main()
