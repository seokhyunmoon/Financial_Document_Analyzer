from pathlib import Path
import yaml
from jinja2 import Template

def load_prompt(name: str) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    p = (repo_root / "prompts" / f"{name}.yaml")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {"system": data.get("system",""), "user": data.get("user","")}

def render_prompt(template: str, **kwargs) -> str:
    return Template(template).render(**kwargs)