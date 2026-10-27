"""Agent tools for TAM support ticket triage.
Each tool is a pure function that looks up data from context (ctx).
"""
import math
from datetime import datetime

AS_OF_DATE = datetime.strptime("2026-10-20", "%Y-%m-%d")


def lookup_account(ctx, account_id: str = "", domain: str = "") -> dict:
    """Finds account information by account_id or requester email domain.
    Returns account details and key risk signals (tier, renewal window, escalations).
    """
    df = ctx.accounts
    account_row = None

    # 1. Search by account_id if provided
    if account_id:
        match = df[df["account_id"] == account_id]
        if not match.empty:
            account_row = match.iloc[0].to_dict()

    # 2. If not found, search by email domain
    if not account_row and domain:
        match = df[df["domain"] == domain]
        if not match.empty:
            account_row = match.iloc[0].to_dict()

    if not account_row:
        return {"found": False, "error": f"No account found for id='{account_id}' domain='{domain}'"}

    # 3. Calculate key risk signals
    signals = []
    renewal_str = account_row.get("renewal_date", "")
    days_to_renewal = None
    if renewal_str:
        try:
            renewal_dt = datetime.strptime(renewal_str, "%Y-%m-%d")
            days_to_renewal = (renewal_dt - AS_OF_DATE).days
            if days_to_renewal < 0:
                signals.append(f"Contract expired {abs(days_to_renewal)} days ago")
            elif days_to_renewal <= 90:
                signals.append(f"Renewal in {days_to_renewal} days (upcoming - high churn risk)")
            else:
                signals.append(f"Renewal in {days_to_renewal} days")
        except ValueError:
            pass

    open_esc = int(account_row.get("open_escalations", 0) or 0)
    if open_esc > 0:
        signals.append(f"{open_esc} open executive escalation(s)")

    tier = account_row.get("tier", "Standard")
    signals.append(f"{tier} tier account ({account_row.get('arr_band', 'N/A')})")

    # Recent SSO change matters: some errors (e.g. E-3310) are caused by SSO reconfiguration
    sso_str = account_row.get("sso_enabled_date", "")
    if sso_str:
        try:
            sso_days = (AS_OF_DATE - datetime.strptime(sso_str, "%Y-%m-%d")).days
            if 0 <= sso_days <= 30:
                signals.append(f"SSO enabled/reconfigured {sso_days} days ago ({sso_str})")
        except ValueError:
            pass

    return {
        "found": True,
        "account_id": account_row.get("account_id"),
        "company_name": account_row.get("company_name"),
        "tier": tier,
        "arr_band": account_row.get("arr_band"),
        "renewal_date": renewal_str,
        "days_to_renewal": days_to_renewal,
        "modules_licensed": account_row.get("modules_licensed", "").split(";"),
        "open_escalations": open_esc,
        "nps_last": account_row.get("nps_last"),
        "sso_enabled_date": sso_str or None,
        "account_signals": signals,
    }


def get_usage_metrics(ctx, account_id: str) -> dict:
    """Retrieves 12-month usage metrics and trends for an account.
    Checks for drops in active users, documents processed, or spikes in tickets.
    """
    df = ctx.usage
    if not account_id:
        return {"found": False, "error": "No account_id provided"}

    account_usage = df[df["account_id"] == account_id].sort_values("month")
    if account_usage.empty:
        return {"found": False, "error": f"No usage data found for {account_id}"}

    recent = account_usage.iloc[-1].to_dict()
    signals = []

    # Calculate 3-month trend: compare latest month with the month 3 months earlier
    if len(account_usage) >= 4:
        past = account_usage.iloc[-4].to_dict()
        try:
            recent_users = float(recent.get("active_users", 0) or 0)
            past_users = float(past.get("active_users", 0) or 0)
            recent_docs = float(recent.get("documents_processed", 0) or 0)
            past_docs = float(past.get("documents_processed", 0) or 0)

            user_pct = round(((recent_users - past_users) / past_users) * 100, 1) if past_users > 0 else 0
            doc_pct = round(((recent_docs - past_docs) / past_docs) * 100, 1) if past_docs > 0 else 0

            signals.append(f"3-month trend: active users {user_pct:+}%, documents {doc_pct:+}%")

            # Check exact KB-2914 account risk override threshold (>25% decline in active users)
            if user_pct <= -25:
                signals.append(f"Account risk override trigger: active users declined {abs(user_pct)}% (>25% per KB-2914)")
        except (ValueError, TypeError):
            pass

    # Check for recent support ticket spikes
    recent_tickets = int(recent.get("tickets_opened", 0) or 0)
    if recent_tickets >= 5:
        signals.append(f"Ticket surge: {recent_tickets} tickets opened this month")

    return {
        "found": True,
        "account_id": account_id,
        "latest_month": recent.get("month"),
        "active_users": recent.get("active_users"),
        "documents_processed": recent.get("documents_processed"),
        "login_rate_pct": recent.get("login_rate_pct"),
        "tickets_opened_recent": recent_tickets,
        "usage_signals": signals,
    }


def search_kb(ctx, query: str = "", error_code: str = "", module: str = "") -> list[dict]:
    """Searches knowledge-base articles by error code, module, and keywords.
    Automatically follows deprecated articles to their current superseded target.
    """
    kb = ctx.kb  # List of {"id", "meta", "body"}
    kb_by_id = {a["id"]: a for a in kb}
    matched_articles = []
    seen_ids = set()

    # 1. Exact Error Code matching (Highest Priority)
    if error_code:
        err_clean = error_code.strip()
        for art in kb:
            meta = art["meta"]
            codes = meta.get("error_codes", [])
            if err_clean in codes or err_clean in art["body"]:
                if art["id"] not in seen_ids:
                    seen_ids.add(art["id"])
                    matched_articles.append(art)

    # 2. Keyword matching across title, module, and body
    if query:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        stems = [t[:5] if len(t) >= 5 else t for t in terms]

        # IDF weight: rare words ("bulk") count more than common ones ("purchase")
        n = len(kb)
        weights = []
        for s in stems:
            df = sum(1 for art in kb if s in art["meta"].get("title", "").lower() or s in art["body"].lower())
            weights.append(math.log((n + 1) / (df + 1)))

        for art in kb:
            if art["id"] in seen_ids:
                continue
            meta = art["meta"]
            title_lower = meta.get("title", "").lower()
            mod_lower = str(meta.get("module", "")).lower()
            body_lower = art["body"].lower()

            # Score relevance
            score = 0
            for term, s, w in zip(terms, stems, weights):
                if term in title_lower or s in title_lower:
                    score += 5 * w
                if term in mod_lower:
                    score += 2 * w
                if term in body_lower or s in body_lower:
                    score += 1 * w

            # Boost articles from the ticket's module (module text is free-form, so match loosely)
            mod_q = module.lower().strip()
            if score > 0 and mod_q and mod_q != "all" and (mod_q in mod_lower or (mod_lower and mod_lower in mod_q)):
                score += 3

            if score > 0:
                matched_articles.append((score, art))
                seen_ids.add(art["id"])

        # Sort keyword matches by relevance score
        matched_articles = [a for a in matched_articles if not isinstance(a, tuple)] + [
            a[1] for a in sorted([x for x in matched_articles if isinstance(x, tuple)], key=lambda x: x[0], reverse=True)
        ]

    # 3. Resolve any DEPRECATED articles to their current target (superseded_by)
    resolved = []
    final_seen = set()
    for art in matched_articles:
        meta = art["meta"]
        target = art
        if meta.get("status") == "deprecated" and meta.get("superseded_by"):
            sup_id = meta["superseded_by"]
            if sup_id in kb_by_id:
                target = kb_by_id[sup_id]

        if target["id"] not in final_seen:
            final_seen.add(target["id"])
            # Format clean summary for LLM context
            body_preview = target["body"].strip().replace("\n\n", " ")[:250]
            resolved.append({
                "id": target["id"],
                "title": target["meta"].get("title"),
                "module": target["meta"].get("module"),
                "type": target["meta"].get("type"),
                "status": target["meta"].get("status"),
                "excerpt": body_preview,
            })

    # Return top 5 most relevant articles (short excerpts keep the prompt small)
    return resolved[:5]


def get_kb_article(ctx, article_id: str) -> dict:
    """Returns the full text of one KB article plus its related article ids.
    Lets the agent read handling steps and follow links from one article to another.
    """
    kb_by_id = {a["id"]: a for a in ctx.kb}
    art = kb_by_id.get(article_id.strip())
    if not art:
        return {"found": False, "error": f"No KB article {article_id}"}
    meta = art["meta"]
    return {
        "found": True,
        "id": art["id"],
        "title": meta.get("title"),
        "type": meta.get("type"),
        "status": meta.get("status"),
        "superseded_by": meta.get("superseded_by"),
        "related": meta.get("related", []),
        "body": art["body"].strip()[:3000],  # cap size to keep the prompt small
    }


def check_related_tickets(ctx, account_id: str, current_ticket_id: str = "", module: str = "", error_code: str = "") -> dict:
    """Finds recent and related tickets for an account.
    Detects duplicate tickets within 7 days (KB-5930) and recurring issues.
    """
    df = ctx.tickets
    if not account_id:
        return {"found": False, "error": "No account_id provided"}

    # Filter for this account's tickets, excluding current ticket
    acc_tix = df[df["account_id"] == account_id]
    if current_ticket_id:
        acc_tix = acc_tix[acc_tix["ticket_id"] != current_ticket_id]

    if acc_tix.empty:
        return {"found": True, "duplicate_candidate": None, "recent_tickets": []}

    acc_tix = acc_tix.sort_values("created_at", ascending=False)
    duplicate_candidate = None
    signals = []

    # Check for potential duplicates (open/pending within 7 days per KB-5930)
    for _, t in acc_tix.iterrows():
        t_status = t.get("status", "")
        t_created = t.get("created_at", "")
        t_err = t.get("error_code", "")
        t_mod = str(t.get("module", "")).lower()

        # Parse date and check 7-day window
        try:
            dt = datetime.fromisoformat(t_created[:10])
            days_ago = (AS_OF_DATE - dt).days
        except Exception:
            days_ago = 999

        if 0 <= days_ago <= 7 and t_status in ("open", "pending_customer"):
            # Check if matching error code or module
            same_err = error_code and t_err and (error_code.strip() == t_err.strip())
            same_mod = module and t_mod and (module.lower() in t_mod or t_mod in module.lower())
            # Only flag when the error code or module matches (KB-5930)
            if same_err or same_mod:
                duplicate_candidate = {
                    "ticket_id": t.get("ticket_id"),
                    "subject": t.get("subject"),
                    "status": t_status,
                    "created_at": t_created,
                    "days_ago": days_ago,
                    "error_code": t_err,
                    "customer_priority": t.get("customer_priority") or None,
                }
                signals.append(f"Potential duplicate of {t.get('ticket_id')} (opened {days_ago} days ago, status: {t_status})")
                break

    # Return summary of top 3 recent tickets
    recent_summary = []
    for _, t in acc_tix.head(3).iterrows():
        recent_summary.append({
            "ticket_id": t.get("ticket_id"),
            "created_at": t.get("created_at"),
            "subject": t.get("subject"),
            "status": t.get("status"),
            "error_code": t.get("error_code"),
        })

    return {
        "found": True,
        "account_id": account_id,
        "duplicate_candidate": duplicate_candidate,
        "recent_tickets": recent_summary,
        "ticket_signals": signals,
    }
