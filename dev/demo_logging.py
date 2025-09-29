#!/usr/bin/env python3
"""Demo of all logging functionality from art.utils.logging."""

import time

from art.utils.logging import _C, _ts, dim, err, info, ok, step, warn


def demo_basic_logging():
    """Demonstrate the basic logging functions."""
    print("=" * 60)
    print("BASIC LOGGING FUNCTIONS")
    print("=" * 60)

    info("This is an informational message")
    step("This indicates a step in a process")
    ok("This indicates successful completion")
    warn("This is a warning message")
    err("This is an error message")
    dim("This is dimmed/secondary text")

    print()


def demo_color_codes():
    """Demonstrate the color code constants."""
    print("=" * 60)
    print("COLOR CODE CONSTANTS (_C class)")
    print("=" * 60)

    print("Available color constants:")
    print(f"{_C.RESET}RESET{_C.RESET} - Reset all formatting")
    print(f"{_C.DIM}DIM{_C.RESET} - Dimmed text")
    print(f"{_C.BOLD}BOLD{_C.RESET} - Bold text")
    print(f"{_C.ITAL}ITAL{_C.RESET} - Italic text")
    print(f"{_C.GRAY}GRAY{_C.RESET} - Gray color")
    print(f"{_C.BLUE}BLUE{_C.RESET} - Blue color")
    print(f"{_C.CYAN}CYAN{_C.RESET} - Cyan color")
    print(f"{_C.GREEN}GREEN{_C.RESET} - Green color")
    print(f"{_C.YELLOW}YELLOW{_C.RESET} - Yellow color")
    print(f"{_C.RED}RED{_C.RESET} - Red color")
    print(f"{_C.MAGENTA}MAGENTA{_C.RESET} - Magenta color")

    print("\nCustom formatted messages:")
    print(f"{_C.BOLD}{_C.BLUE}Bold Blue Text{_C.RESET}")
    print(f"{_C.ITAL}{_C.GREEN}Italic Green Text{_C.RESET}")
    print(f"{_C.DIM}{_C.GRAY}Dimmed Gray Text{_C.RESET}")

    print()


def demo_timestamp():
    """Demonstrate the timestamp function."""
    print("=" * 60)
    print("TIMESTAMP FUNCTION (_ts)")
    print("=" * 60)

    print(f"Current timestamp: {_ts()}")
    print(f"Timestamp format: HH:MM:SS")
    print(f"Example with custom message: [{_ts()}] Custom log message")

    print()


def demo_real_world_usage():
    """Demonstrate real-world usage scenarios."""
    print("=" * 60)
    print("REAL-WORLD USAGE SCENARIOS")
    print("=" * 60)

    # Simulating a process with multiple steps
    info("Starting data processing pipeline")

    step("Loading configuration file")
    time.sleep(0.5)  # Simulate work
    ok("Configuration loaded successfully")

    step("Connecting to database")
    time.sleep(0.3)  # Simulate work
    ok("Database connection established")

    step("Processing 1000 records")
    time.sleep(0.7)  # Simulate work
    warn("Skipped 2 invalid records")
    ok("Processed 998/1000 records successfully")

    step("Generating report")
    time.sleep(0.4)  # Simulate work
    ok("Report generated successfully")

    info("Pipeline completed")
    dim("   Total time: 2.1 seconds")
    dim("   Records processed: 998")
    dim("   Records skipped: 2")

    print()


def demo_progress_tracking():
    """Demonstrate progress tracking with logging."""
    print("=" * 60)
    print("PROGRESS TRACKING EXAMPLE")
    print("=" * 60)

    total_items = 5
    info(f"Processing {total_items} items")

    for i in range(1, total_items + 1):
        step(f"Processing item {i}/{total_items}")
        time.sleep(0.2)  # Simulate work

        if i == 3:
            warn(f"Item {i} required additional validation")

        ok(f"Item {i} completed")
        dim(f"   Progress: {i}/{total_items} ({i / total_items * 100:.0f}%)")

    ok("All items processed successfully")

    print()


def demo_error_scenarios():
    """Demonstrate error reporting scenarios."""
    print("=" * 60)
    print("ERROR REPORTING SCENARIOS")
    print("=" * 60)

    info("Testing error handling scenarios")

    step("Attempting risky operation 1")
    warn("Operation completed with warnings")
    dim("   Warning: Deprecated API used")

    step("Attempting risky operation 2")
    err("Operation failed with error")
    dim("   Error: File not found: /path/to/missing/file.txt")
    dim("   Suggestion: Check file path and permissions")

    step("Attempting recovery")
    ok("Successfully recovered using fallback method")

    print()


def demo_formatting_combinations():
    """Demonstrate various formatting combinations."""
    print("=" * 60)
    print("ADVANCED FORMATTING COMBINATIONS")
    print("=" * 60)

    # Combining colors and styles
    print("Style combinations:")
    print(f"{_C.BOLD}{_C.RED}Bold Red Error{_C.RESET}")
    print(f"{_C.BOLD}{_C.GREEN}Bold Green Success{_C.RESET}")
    print(f"{_C.BOLD}{_C.YELLOW}Bold Yellow Warning{_C.RESET}")
    print(f"{_C.ITAL}{_C.BLUE}Italic Blue Info{_C.RESET}")
    print(f"{_C.DIM}{_C.GRAY}Dimmed Gray Details{_C.RESET}")

    print("\nNested formatting:")
    print(
        f"Regular text with {_C.BOLD}bold{_C.RESET} and {_C.ITAL}italic{_C.RESET} sections"
    )
    print(
        f"{_C.BLUE}Blue text with {_C.BOLD}bold section{_C.RESET}{_C.BLUE} continuing in blue{_C.RESET}"
    )

    print("\nStatus indicators:")
    print(f"[{_C.GREEN}{_C.RESET}] Success indicator")
    print(f"[{_C.YELLOW}!{_C.RESET}] Warning indicator")
    print(f"[{_C.RED}{_C.RESET}] Error indicator")
    print(f"[{_C.BLUE}i{_C.RESET}] Info indicator")

    print()


def demo_log_levels():
    """Demonstrate different log levels in action."""
    print("=" * 60)
    print("LOG LEVELS DEMONSTRATION")
    print("=" * 60)

    print("Simulating application startup:")
    info("Application starting up")
    step("Initializing modules")
    ok("Core modules loaded")
    step("Starting services")
    warn("Service A started with reduced performance mode")
    ok("Service B started normally")
    err("Service C failed to start")
    dim("   Fallback: Using Service D instead")
    ok("Service D started successfully")
    info("Application startup complete")

    print("\nSimulating application shutdown:")
    info("Shutting down application")
    step("Stopping services")
    ok("All services stopped cleanly")
    step("Cleaning up resources")
    ok("Resources cleaned up")
    info("Application shutdown complete")

    print()


def main():
    """Run all logging demonstrations."""
    print(f"{_C.BOLD}{_C.CYAN}ART Logging System Demo{_C.RESET}")
    print(f"Timestamp: {_ts()}")
    print()

    # Run all demonstrations
    demo_basic_logging()
    demo_color_codes()
    demo_timestamp()
    demo_real_world_usage()
    demo_progress_tracking()
    demo_error_scenarios()
    demo_formatting_combinations()
    demo_log_levels()

    # Final summary
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    ok("All logging functionality demonstrated successfully")
    info("Available functions: info(), step(), ok(), warn(), err(), dim()")
    info("Available constants: _C class with color codes, _ts() for timestamps")
    dim("   For more details, see: src/art/utils/logging.py")

    print(f"\n{_C.BOLD}Usage Examples:{_C.RESET}")
    print("from art.utils.logging import info, step, ok, warn, err, dim, _C")
    print("info('Starting process')")
    print("step('Processing data')")
    print("ok('Process completed')")
    print("warn('Performance degraded')")
    print("err('Operation failed')")
    print("dim('Additional details')")
    print(f"print(f'{_C.BOLD}Bold text{_C.RESET}')")


if __name__ == "__main__":
    main()
