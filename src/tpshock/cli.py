"""Command line entry point: `tpshock <step>`."""

import argparse
import logging

from tpshock import pipeline


def main():
    steps = {
        "fetch": pipeline.fetch,
        "fit": pipeline.fit_term_premium,
        "validate": pipeline.validate_term_premium,
        "shocks": pipeline.identify_shocks,
        "transmit": pipeline.transmission,
        "site": lambda: __import__("tpshock.site.build", fromlist=["build"]).build(),
        "all": pipeline.run_all,
    }
    p = argparse.ArgumentParser(prog="tpshock", description="Fed shocks and the term premium")
    p.add_argument("step", choices=steps)
    p.add_argument("--skip-fetch", action="store_true", help="with 'all': use cached raw data")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.step == "all":
        pipeline.run_all(skip_fetch=args.skip_fetch)
    else:
        steps[args.step]()


if __name__ == "__main__":
    main()
