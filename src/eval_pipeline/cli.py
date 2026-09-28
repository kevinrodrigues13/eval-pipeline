"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .calibration import CalibrationFailed, calibrate, gate_on_calibration
from .ingestion import load_comparisons, load_verdicts
from .judge import LLMJudge, PrecomputedJudge
from .providers import DEFAULT_MODEL
from .report import generate_report
from .scoring import compare_policies
from .validation import flag_noisy_labels, sample_comparisons


def _build_live_judge(args, rows, policy_rows):
    """
    Construct a live judge, after telling the user what it will cost.

    Every comparison is asked twice for position control, so the call count
    is double the row count. The estimate is printed before anything is
    spent rather than discovered afterwards.
    """
    from .providers import ResponseCache, anthropic_judge_backend, estimate_cost

    n_calls = (len(rows) + len(policy_rows)) * 2
    characters = sum(
        len(r.prompt) + len(r.response_a) + len(r.response_b) for r in rows
    ) / max(len(rows), 1)
    avg_input_tokens = int(characters / 4) + 300  # + the system prompt

    cache = ResponseCache(args.cache) if args.cache else None
    cached = sum(1 for _ in cache.entries) if cache is not None else 0

    print(
        f"judge:      live via {args.model}\n"
        f"            {n_calls} calls ({len(rows)} calibration + "
        f"{len(policy_rows)} held-out comparisons, each asked in both orderings)\n"
        f"            ~{avg_input_tokens} input tokens each, estimated cost "
        f"${estimate_cost(n_calls, avg_input_tokens, args.model):.2f}"
        + (f"\n            {cached} responses already cached" if cached else "")
        + (
            f"\n            concurrency: {args.concurrency} calls in flight — "
            "same total spend, faster wall clock"
            if args.concurrency > 1
            else ""
        )
        + (
            "\n            --skip-parse-errors: a malformed reply is counted "
            "and excluded rather than aborting the run"
            if args.skip_parse_errors
            else ""
        )
    )

    if not args.yes:
        # Refuse when there is nobody to ask. Making the prompt conditional on
        # a TTY means a piped or scripted run spends money with no confirmation
        # at all — the opposite of what a confirmation is for.
        if not sys.stdin.isatty():
            print(
                "            refusing to spend without confirmation: stdin is not "
                "interactive.\n            Pass --yes to proceed, or --judge "
                "recorded to run free."
            )
            return None
        if input("            proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return None

    return LLMJudge(
        anthropic_judge_backend(model=args.model, cache=cache),
        name=args.judge_name or args.model,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline evaluation and judge calibration pipeline"
    )
    parser.add_argument("--preferences", required=True, help="human-labelled comparisons")
    parser.add_argument("--policy-outputs", required=True, help="held-out set")
    parser.add_argument(
        "--judge",
        choices=("live", "recorded"),
        default="live",
        help="live: call a model (costs money). recorded: replay published "
        "verdicts — free, deterministic, but only covers comparisons in the file",
    )
    parser.add_argument(
        "--verdicts", help="recorded judge verdicts; required for --judge recorded"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--calibration-items",
        type=int,
        default=200,
        help="how many labelled comparisons to calibrate on; 0 uses all. Each "
        "carries all its human verdicts, so the ceiling survives sampling "
        "automatically",
    )
    parser.add_argument(
        "--cache",
        default=".judge_cache.jsonl",
        help="reuse identical judge calls across runs; '' disables",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the cost confirmation prompt"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="judge calls in flight at once. 1 (default) runs sequentially — "
        "identical timing and behaviour to a pipeline that never knew this "
        "flag existed. Only speeds up wall-clock time for --judge live; "
        "total spend is unchanged either way, since the call count doesn't "
        "depend on concurrency. Raise it no further than your account's "
        "rate limit tolerates. Pair with --skip-parse-errors: above 1, a "
        "parse failure no longer stops the run after only the calls made "
        "so far, since every already-dispatched call still completes (and "
        "is billed) before the exception surfaces",
    )
    parser.add_argument(
        "--skip-parse-errors",
        action="store_true",
        help="a comparison the judge never returned a parsable "
        "[[A]]/[[B]]/[[C]] verdict for is counted and excluded, rather than "
        "aborting the whole run. Off by default: a parse failure is loud on "
        "purpose, and a single one killing hundreds of already-paid-for "
        "calls is what motivated this flag existing at all",
    )
    parser.add_argument("--output", default="output/evaluation_report.md")
    parser.add_argument("--policy-1-name", default="Policy 1")
    parser.add_argument("--policy-2-name", default="Policy 2")
    parser.add_argument("--judge-name", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    all_rows, load_report = load_comparisons(args.preferences)
    print(f"ingestion:  {load_report.summary()}")

    rows = sample_comparisons(all_rows, args.calibration_items, seed=args.seed)
    if len(rows) != len(all_rows):
        print(f"sampling:   {len(rows)} of {len(all_rows)} comparisons")

    quality = flag_noisy_labels(rows)
    print(f"validation: {quality.summary()}")

    policy_rows, _ = load_comparisons(args.policy_outputs)

    if args.judge == "recorded":
        if not args.verdicts:
            parser.error("--judge recorded requires --verdicts")
        judge = PrecomputedJudge.from_records(
            load_verdicts(args.verdicts), strict=False, name=args.judge_name or "recorded"
        )
        if judge.conflicts:
            print(
                f"judge:      {judge.conflicts} comparisons carried conflicting "
                "recorded verdicts; each resolved to a flagged tie"
            )
    else:
        judge = _build_live_judge(args, rows, policy_rows)
        if judge is None:
            return 1

    calibration = calibrate(
        judge, rows, label_quality=quality,
        max_workers=args.concurrency, skip_parse_errors=args.skip_parse_errors,
    )
    print(f"calibration: {calibration.summary()}")

    gate_passed = True
    try:
        gate_on_calibration(calibration)
        print("gate:       PASSED")
    except CalibrationFailed as error:
        print(f"gate:       FAILED — {error}")
        gate_passed = False

    # Scored and reported even on gate failure: report.py's own bottom line
    # already renders "no conclusion, nothing below should be acted on" for
    # this case — a human still gets a document explaining why, rather than
    # just a terse CLI failure.
    score = compare_policies(
        judge, policy_rows,
        max_workers=args.concurrency, skip_parse_errors=args.skip_parse_errors,
    )

    markdown = generate_report(
        calibration,
        score,
        label_quality=quality,
        policy_names=(args.policy_1_name, args.policy_2_name),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"\nreport written to {out_path}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
