"""Module entry point for ``python -m shooklink``."""


def main() -> int | None:
    from .app import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
