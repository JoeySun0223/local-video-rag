from .cli import parser, run

raise SystemExit(run(parser().parse_args()))
