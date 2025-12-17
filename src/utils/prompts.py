from pathlib import Path
import yaml
from jinja2 import Template

def load_prompt(name: str) -> dict:
    """Load a prompt YAML by name from the prompts directory.

    Args:
        name: Prompt file stem (without extension).

    Returns:
        Dict with ``system`` and ``user`` prompt strings.
    """
    repo_root = Path(__file__).resolve().parents[1]
    p = (repo_root / "prompts" / f"{name}.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {"system": data.get("system",""), "user": data.get("user","")}

def render_prompt(template: str, **kwargs) -> str:
    """Render a Jinja2 prompt template with provided context.

    Args:
        template: Template string.
        **kwargs: Variables injected into the template.

    Returns:
        Rendered prompt string.
    """
    return Template(template).render(**kwargs)
