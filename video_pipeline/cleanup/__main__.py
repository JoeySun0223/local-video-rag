"""运行清洗阶段命令行入口。"""

from .cli import parser, run

raise SystemExit(run(parser().parse_args()))
