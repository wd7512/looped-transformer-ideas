from __future__ import annotations

import logging


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info("Hello from looped-transformer-ideas!")


if __name__ == "__main__":
    main()
