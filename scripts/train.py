import sys

from reliable_proxy.cli import main

raise SystemExit(main(["train", *sys.argv[1:]]))
