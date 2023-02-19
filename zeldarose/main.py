import click


class ComplexCLI(click.MultiCommand):
    def list_commands(self, ctx):
        return ["tokenizer", "transformer"]

    def get_command(self, ctx, name):
        try:
            mod = __import__(f"zeldarose.train_{name}", None, None, ["cli"])
        except ImportError:
            return
        return mod.main


@click.group(cls=ComplexCLI, help="A straightforward trainer for transformer-based models.")
def cli():
    pass
