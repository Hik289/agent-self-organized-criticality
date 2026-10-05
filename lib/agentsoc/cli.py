from __future__ import annotations

import argparse
from numbers import Real

from .io import field, read_input, records, write_output
from .statistics import avalanche_nulls, benjamini_hochberg, fit_tail_models, threshold_events


def build_parser():
    parser = argparse.ArgumentParser(description="Offline diagnostics for saved agent trajectories; no model calls or benchmark execution.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("avalanches", "Compare strict contiguous error events with matched Bernoulli and Markov nulls."),
        ("tails", "Compare four continuous tail densities on one shared support."),
        ("stress", "Predict final reward error from a fixed early prefix with grouped validation."),
        ("divergence", "Refit paired divergence and measure sustained entry into a tail band."),
        ("recovery", "Measure transitions and censored residence in operational fidelity basins."),
        ("temporal", "Evaluate PSD exponents against within-trajectory permutations and length-qualified DFA."),
        ("horizon", "Summarize horizon scaling with matched-seed sign tests."),
        ("fdr", "Apply Benjamini-Hochberg correction to an explicitly supplied test family."),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--input", required=True, help="Existing JSON or JSONL data file.")
        command.add_argument("--output", required=True, help="New JSON output path; existing files are never overwritten.")
        if name in {"avalanches", "tails", "horizon"}:
            command.add_argument("--threshold", type=float, default=0.5)
        if name in {"avalanches", "tails", "temporal"}:
            command.add_argument("--series-field", default="e_series")
        if name in {"avalanches", "stress", "temporal"}:
            command.add_argument("--seed", type=int, default=42)
        if name == "avalanches":
            command.add_argument("--n-simulations", type=int, default=1000)
        elif name == "tails":
            command.add_argument("--xmin", type=float)
            command.add_argument("--min-tail-size", type=int, default=20)
            command.add_argument("--samples-field", help="Explicit positive numeric sample field; omit to extract event-level weighted sizes from error trajectories.")
            command.add_argument("--include-boundary-events", action="store_true", help="Include events touching either observation boundary; otherwise only interior events enter the tail sample.")
        elif name == "stress":
            command.add_argument("--prefix-steps", type=int, default=5)
            command.add_argument("--n-splits", type=int, default=5)
        elif name == "divergence":
            command.add_argument("--saturation-tolerance", type=float, default=0.1)
            command.add_argument("--saturation-window", type=int, default=10)
        elif name == "recovery":
            command.add_argument("--wrong-threshold", type=float, default=0.5)
            command.add_argument("--correct-threshold", type=float, default=0.9)
        elif name == "temporal":
            command.add_argument("--group-field", help="Optional dotted field identifying groups, for example meta.H.")
            command.add_argument("--n-permutations", type=int, default=1000)
            command.add_argument("--min-dfa-length", type=int, default=128)
        elif name == "horizon":
            command.add_argument("--horizon-field", default="meta.H")
            command.add_argument("--pair-field", default="seed", help="Shared task/seed identifier across horizons; input must contain one experimental condition.")
        elif name == "fdr":
            command.add_argument("--q", type=float, default=0.05)
    return parser


def _tail_input(data, args):
    if isinstance(data, list) and data and all(isinstance(value, Real) and not isinstance(value, bool) for value in data):
        return data, {"source": "explicit_numeric_array", "n_samples": len(data)}
    rows = records(data)
    if args.samples_field:
        samples = [field(row, args.samples_field) for row in rows]
        return samples, {"source": "explicit_record_field", "field": args.samples_field, "n_samples": len(samples)}
    samples = []
    n_boundary_excluded = 0
    for row in rows:
        events = threshold_events(field(row, args.series_field), args.threshold)
        for event in events["events"]:
            if not args.include_boundary_events and (event["touches_start_boundary"] or event["touches_end_boundary"]):
                n_boundary_excluded += 1
            else:
                samples.append(event["size"])
    return samples, {
        "source": "strict_contiguous_threshold_event_error_sums",
        "series_field": args.series_field,
        "n_trajectories": len(rows),
        "n_samples": len(samples),
        "n_boundary_events_excluded": n_boundary_excluded,
        "include_boundary_events": args.include_boundary_events,
        "likelihood_scope": "pooled_event_marginal; no_assumption_of_independent_events_for_significance_tests",
    }


def dispatch(data, args):
    if args.command == "avalanches":
        rows = records(data)
        sequences = [field(row, args.series_field) for row in rows]
        result = avalanche_nulls(sequences, threshold=args.threshold, n_simulations=args.n_simulations, seed=args.seed)
        result["event_catalog"] = [{"row_index": index, **threshold_events(sequence, args.threshold)} for index, sequence in enumerate(sequences)]
        return result
    if args.command == "tails":
        samples, selection = _tail_input(data, args)
        return {"sample_selection": selection, **fit_tail_models(samples, xmin=args.xmin, min_tail_size=args.min_tail_size)}
    if args.command == "stress":
        from .stress import analyze_early_stress
        return analyze_early_stress(records(data), prefix_steps=args.prefix_steps, n_splits=args.n_splits, seed=args.seed)
    if args.command == "divergence":
        from .dynamics import analyze_divergence
        return analyze_divergence(records(data, "pairs"), saturation_tolerance=args.saturation_tolerance, saturation_window=args.saturation_window)
    if args.command == "recovery":
        from .dynamics import analyze_recovery
        return analyze_recovery(records(data), wrong_threshold=args.wrong_threshold, correct_threshold=args.correct_threshold)
    if args.command == "temporal":
        from .temporal import analyze_temporal
        return analyze_temporal(records(data), series_field=args.series_field, group_field=args.group_field, n_permutations=args.n_permutations, min_dfa_length=args.min_dfa_length, seed=args.seed)
    if args.command == "horizon":
        from .horizon import analyze_horizons
        return analyze_horizons(records(data), threshold=args.threshold, horizon_field=args.horizon_field, pair_field=args.pair_field)
    if args.command == "fdr":
        tests = records(data, "tests")
        pvalues = [field(test, "p_value") for test in tests]
        corrected = benjamini_hochberg(pvalues, q=args.q)
        corrected["tests"] = [{"id": test.get("id", index), "p_value": pvalues[index], "p_adjusted": corrected["p_adjusted"][index], "rejected": corrected["rejected"][index]} for index, test in enumerate(tests)]
        return corrected
    raise ValueError(f"Unknown command: {args.command}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data, source = read_input(args.input)
        result = dispatch(data, args)
        parameters = {key: value for key, value in vars(args).items() if key not in {"input", "output"}}
        output = write_output(args.output, result, [source], parameters)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print(str(output))
