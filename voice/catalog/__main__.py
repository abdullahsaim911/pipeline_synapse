"""
Command line for the action catalog.

  python -m voice.catalog check                    validate the data and print counts
  python -m voice.catalog export-js --out <file>   write the frontend contract module
"""

import argparse
import sys

from .catalog import ActionCatalog, CatalogError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    export = sub.add_parser("export-js")
    export.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        catalog = ActionCatalog.load()
    except CatalogError as exc:
        print(f"Catalog is invalid: {exc}", file=sys.stderr)
        return 1

    if args.command == "check":
        print(f"{len(catalog.ids())} actions")
        for lang in catalog.languages:
            examples = sum(len(v) for v in catalog.examples(lang).values())
            print(f"  {lang}: {examples} example sentences, {len(catalog.eval_items(lang))} held-out items")
        return 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(catalog.to_frontend_js())
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
