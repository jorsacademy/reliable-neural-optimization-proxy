import sys

from reliable_proxy.cli import main

raise SystemExit(main(["benchmark", *sys.argv[1:]]))
