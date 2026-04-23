"""Training entrypoint placeholder for CardioFit."""

from __future__ import annotations

import logging


def main() -> None:
    """Run training entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger(__name__).info(
        "Training entrypoint is ready. Implement pipeline in src/cardiofit/training/."
    )


if __name__ == "__main__":
    main()
