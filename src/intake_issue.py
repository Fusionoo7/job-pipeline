import os
import json
import re

from .notion_client import (
    get_database_schema,
    build_property_index,
    create_page_safe,
    NotionError,
)


def chunk_rich_text(s: str, chunk: int = 1900, max_len: int = 6000):
    s = s or ""
    s = s[:max_len]  # optional cap to avoid absurdly long JDs
    return [{"text": {"content": s[i:i+chunk]}} for i in range(0, len(s), chunk)]


def parse_issue_form(body: str) -> dict:
    """
    GitHub Issue Forms render as markdown:
    ### Company
    Value
    ### Role
    Value
    ...
    """
    body = body or ""
    out = {}
    pattern = re.compile(r"(?ms)^###\s+(?P<k>.+?)\s*\n(?P<v>.*?)(?=^###\s+|\Z)")
    for m in pattern.finditer(body):
        k = m.group("k").strip().lower()
        v = m.group("v").strip()
        out[k] = v

    # normalize keys
    def g(*names):
        for n in names:
            if n in out and out[n]:
                return out[n]
        return ""

    return {
        "company": g("company"),
        "role": g("role"),
        "job_url": g("job url", "job_url", "job link", "job_link", "url"),
        "job_description": g("job description", "job_description", "jd", "description"),
        "location": g("location (optional)", "location"),
        "notes": g("notes (optional)", "notes"),
    }


def main():
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not os.path.exists(event_path):
        raise NotionError("Missing GITHUB_EVENT_PATH")

    with open(event_path, "r", encoding="utf-8") as f:
        event = json.load(f)
    issue = event.get("issue") or {}
    body = issue.get("body") or ""

    data = parse_issue_form(body)

    missing = [
        k for k in ["company", "role", "job_url", "job_description"] if not data.get(k)
    ]
    if missing:
        raise NotionError(f"Missing required fields in issue form: {missing}")

    schema = get_database_schema()
    idx = build_property_index(schema)

    desired = {
        "Company": data["company"],
        "Role": data["role"],
        "Job URL": data["job_url"],
        "Job Description": chunk_rich_text(data["job_description"]),
        "Status": "Not Applied",
        "Source": "GitHub Form",
    }

    if data.get("location"):
        desired["Location"] = data["location"]
    if data.get("notes"):
        desired["Notes"] = chunk_rich_text(data["notes"])

    created = create_page_safe(desired, idx)
    page_id = created.get("id", "")
    print(f"CREATED_NOTION_PAGE_ID={page_id}")


if __name__ == "__main__":
    main()
