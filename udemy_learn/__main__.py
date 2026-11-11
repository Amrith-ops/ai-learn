import sys
from pathlib import Path
from udemy_learn.conversation import main

if __package__ in {None, ""}:

    # Direct file execution starts with udemy_learn/ on sys.path, so add the
    # repository root to make the package imports work in that mode too.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    main()