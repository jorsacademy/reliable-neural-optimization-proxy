import sys

from reliable_proxy.cli import main

raise SystemExit(main(["collect", *sys.argv[1:]]))
