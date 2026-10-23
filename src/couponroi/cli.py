"""Command line entry point. Every step can run on its own or as `couponroi all`.

    couponroi data        download + validate raw tables
    couponroi build       build the DuckDB warehouse (sql/*.sql)
    couponroi eda         answer the EDA questions, save charts
    couponroi causal      matched DiD, event study, placebo, CUPED, power
    couponroi churn       lapse-risk model with rolling-origin backtest
    couponroi uplift      uplift learners, semi-synthetic check, wave backtest
    couponroi policy      targeting policy, profit curves, sensitivity
    couponroi drift       PSI drift report
    couponroi export      slim app bundle (artifacts/app)
    couponroi all         everything above, in order

Add --synthetic to run the whole thing on a small generated dataset (used by CI).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

STEPS = ["data", "build", "eda", "causal", "churn", "uplift", "policy", "drift", "export"]


def _run_step(step: str, args: argparse.Namespace) -> None:
    from . import config

    config.ensure_dirs()
    if step == "data":
        from .data import fetch

        fetch(synthetic=args.synthetic)
    elif step == "build":
        from .warehouse import build

        build()
    elif step == "eda":
        from .eda import run

        run()
    elif step == "causal":
        from .causal import run

        run()
    elif step == "churn":
        from .churn import run

        run(track=not args.no_mlflow)
    elif step == "uplift":
        from .uplift import run

        run(track=not args.no_mlflow)
    elif step == "policy":
        from .policy import run

        run()
    elif step == "drift":
        from .drift import run

        run()
    elif step == "export":
        from .export import run

        run()
    else:  # pragma: no cover
        raise ValueError(step)


def _ask(question: str | None, evaluate: bool) -> int:
    """`couponroi ask "question"` or `couponroi ask-eval` (needs an LLM endpoint, see ask.py)."""
    import json

    from . import ask, config

    con = ask.connect_readonly()
    client = ask.LLMClient()
    if evaluate:
        res = ask.evaluate(con, client)
        (config.RESULTS_DIR / "ask_eval.json").write_text(json.dumps(res, indent=2, default=str))
        print(
            f"execution accuracy for {res['model']}: {res['execution_accuracy']:.0%} on {res['n']} questions"
        )
        for it in res["items"]:
            print(f"  {'ok ' if it['correct'] else 'NO '} {it['id']:>2} {it['question']}")
        return 0
    res = ask.ask(question or "", con, client)
    if res.error:
        print(res.error)
        return 1
    print(res.sql, "\n")
    print(res.data.head(20).to_string(index=False), "\n")
    print(res.summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="couponroi", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("step", choices=STEPS + ["all", "ask", "ask-eval"])
    parser.add_argument("question", nargs="?", help="for `ask`: the question in plain English")
    parser.add_argument("--synthetic", action="store_true", help="use generated data (fast, for CI)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.synthetic:
        # must be set before couponroi.config is imported: it moves all outputs to build/synthetic
        os.environ["COUPONROI_SYNTHETIC"] = "1"
    if args.step in ("ask", "ask-eval"):
        return _ask(args.question, evaluate=args.step == "ask-eval")

    steps = STEPS if args.step == "all" else [args.step]
    for step in steps:
        t0 = time.perf_counter()
        logging.getLogger("couponroi").info("=== %s ===", step)
        _run_step(step, args)
        logging.getLogger("couponroi").info("=== %s done in %.1fs ===", step, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
