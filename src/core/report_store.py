from pathlib import Path
from uuid import UUID, uuid4

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"


def ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def write_html_report(content: str) -> str:
    ensure_reports_dir()
    report_id = str(uuid4())
    path = REPORTS_DIR / f"{report_id}.html"
    path.write_text(content, encoding="utf-8")
    return report_id


def resolve_report_path(report_id: str) -> Path | None:
    try:
        UUID(report_id)
    except Exception:
        return None
    path = REPORTS_DIR / f"{report_id}.html"
    if not path.exists():
        return None
    return path
