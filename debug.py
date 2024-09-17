from rich.console import Console
DEBUG=True

def debug_print(*args, **kwargs):
    global DEBUG
    if DEBUG:
        console = Console()
        console.print(*args, **kwargs)