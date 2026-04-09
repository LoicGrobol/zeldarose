from os import environ
from nox import Session, options, session

options.default_venv_backend = "uv"

ci = environ.get("CI")


@session(python=["3.11", "3.12", "3.13", "3.14"])
def tests(s: Session):
    if ci:
        s.install(".", "--group", "ci")
    else:
        s.install(".", "--group", "tests")

    s.run("pytest", "tests", "--basetemp", s.create_tmp(), *s.posargs)


@session()
def lint(s: Session):
    if ci:
        fmt = "github"
    else:
        fmt = "full"
    s.install("ruff")
    s.run("ruff", "check", ".", "--select=E9,F63,F7,F82", f"--output-format={fmt}")
    # exit-zero treats all errors as warnings.
    s.run("ruff", "check", ".", "--exit-zero", f"--output-format={fmt}")
