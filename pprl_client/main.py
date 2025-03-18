from ._cli import app


def run_cli():
    app(max_content_width=120)


if __name__ == "__main__":
    run_cli()
