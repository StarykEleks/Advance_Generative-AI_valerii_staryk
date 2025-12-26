import requests
from config import GITHUB_TOKEN, GITHUB_REPO, COMPANY_SUPPORT_EMAIL, COMPANY_SUPPORT_PHONE, COMPANY_NAME

def get_company_info():
    return {
        "company_name": COMPANY_NAME,
        "support_email": COMPANY_SUPPORT_EMAIL,
        "support_phone": COMPANY_SUPPORT_PHONE,
    }

def create_support_ticket(user_name: str, user_email: str, summary: str, description: str):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    body = f"""**User name:** {user_name}
**User email:** {user_email}

---

{description}
"""
    payload = {"title": summary, "body": body, "labels": ["support-ticket"]}

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 300:
        return {"ok": False, "error": f"GitHub API error {r.status_code}: {r.text}"}

    data = r.json()
    return {"ok": True, "ticket_url": data.get("html_url"), "ticket_number": data.get("number")}
