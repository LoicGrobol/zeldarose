from nox import Session, options, session

options.default_venv_backend = "uv"


@session(python=["3.11", "3.12", "3.13", "3.14"])
def tests(s: Session):
    s.install(".[tests]")
    s.run("pytest", "tests", "--basetemp", s.create_tmp(), *s.posargs)
