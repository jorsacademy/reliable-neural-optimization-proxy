import sys

from reliable_proxy.cli import main

raise SystemExit(main(["research", *sys.argv[1:]]))
