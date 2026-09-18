"""
Company research agent.
"""

import json
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

import logging

logger = logging.getLogger(__name__)

load_dotenv()

# -- Config ------------------------------------------------------------------

LLM_MODEL = "openai/gpt-oss-120b"
MAX_SEARCH_RESULTS = 3
MAX_SEARCH_ITERATIONS = 4
N8N_WEBHOOK_URL = "http://localhost:5680/webhook-test/e8ec1971-3497-483f-a611-c9c5f0fec593"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -- Primitives --------------------------------------------------------------

def search_web(query: str) -> str:
    logger.info("Searching: %s", query)
    results = tavily_client.search(query=query, max_results=MAX_SEARCH_RESULTS)
    return "".join(
        f"Title: {r['title']}\nContent: {r['content']}\n\n"
        for r in results["results"]
    )


def ask_llm(messages: list[dict], temperature: float = 0) -> str:
    """temperature=0 by default so identical inputs give stable, repeatable
    outputs run to run. This doesn't fix wrong search results on its own,
    but it stops the LLM steps (planning, filtering, writing) from adding
    their own randomness on top of whatever the search engine returns."""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _parse_json_response(raw: str, fallback: dict) -> dict:
    """LLMs sometimes wrap JSON in ```json fences despite instructions not
    to. Strip that defensively, and never let a parse failure crash the
    pipeline — fall back to a safe default instead."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse JSON from LLM response, using fallback")
        return fallback


# -- Agent steps -------------------------------------------------------------

def identify_company(company_name: str, context_hint: str = "") -> dict:
    """Resolve an ambiguous company name to a single concrete entity BEFORE
    doing any real research. This is the fix for the "wrong company"
    problem you saw: two companies sharing a name can rank differently in
    search results from run to run, and nothing downstream was checking
    which one actually got pulled in. Now everything (queries, filtering,
    report) gets anchored to this resolved entity instead of a bare
    name string.

    context_hint: optional extra info from the caller to disambiguate up
    front, e.g. "fintech, based in Singapore" or a known domain. Cheapest
    and most reliable fix when you already know something about the target.
    """
    logger.info("Step 0 - Identifying entity: %s", company_name)

    query = f"{company_name} official website"
    if context_hint:
        query += f" {context_hint}"
    initial_results = search_web(query)

    messages = [
        {
            "role": "system",
            "content": (
                "You resolve a company name to ONE specific entity using search snippets.\n"
                "If the snippets clearly show multiple different companies sharing this name, "
                "pick the most prominent/likely one given any context hint provided, but flag "
                "the ambiguity and list the other candidates you saw.\n"
                "Output ONLY strict JSON, no markdown, no code fences, no commentary:\n"
                '{"resolved_name": "...", "domain": "...", "industry": "...", '
                '"one_line_description": "...", "ambiguous": true|false, '
                '"other_candidates": ["..."]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company name: {company_name}\n"
                f"Context hint: {context_hint or 'none provided'}\n\n"
                f"Search snippets:\n{initial_results[:3000]}"
            ),
        },
    ]

    raw = ask_llm(messages, temperature=0)
    fallback = {
        "resolved_name": company_name,
        "domain": "",
        "industry": "",
        "one_line_description": "",
        "ambiguous": False,
        "other_candidates": [],
    }
    entity = _parse_json_response(raw, fallback)
    logger.info("Resolved entity: %s", entity)
    return entity


def plan_search_queries(company_name: str, entity: dict) -> list[str]:
    """Ask the LLM to produce 3 targeted, disambiguated search queries."""
    logger.info("Step 1 - Planning searches")

    entity_context = ", ".join(
        filter(None, [entity.get("industry"), entity.get("domain")])
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research planning agent.\n"
                "The target company has been identified as follows:\n"
                f"- Resolved name: {entity.get('resolved_name', company_name)}\n"
                f"- Domain: {entity.get('domain', 'unknown')}\n"
                f"- Industry: {entity.get('industry', 'unknown')}\n"
                f"- Description: {entity.get('one_line_description', 'unknown')}\n\n"
                "Given this, output exactly 3 search queries to research it thoroughly. "
                "Every query MUST include enough disambiguating context (industry, domain, "
                "or location) that results won't be confused with a different company that "
                "happens to share the same or a similar name.\n"
                "Output ONLY a numbered list like this:\n"
                "1. query one\n"
                "2. query two\n"
                "3. query three\n"
                "Nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"Plan research queries for: {company_name}"
            + (f" ({entity_context})" if entity_context else ""),
        },
    ]

    plan = ask_llm(messages, temperature=0)
    logger.info("Search plan: %s", plan)

    queries = [
        line.split(".", 1)[-1].strip()
        for line in plan.strip().splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]

    # Fallback if LLM didn't follow the numbered-list format
    if not queries:
        suffix = f" {entity_context}" if entity_context else ""
        queries = [
            f"{company_name} company overview{suffix}",
            f"{company_name} recent news 2025 2026{suffix}",
            f"{company_name} key people leadership{suffix}",
        ]

    return queries


def run_searches(queries: list[str]) -> str:
    """Execute each query and return aggregated results."""
    all_results = ""

    for i, query in enumerate(queries[:MAX_SEARCH_ITERATIONS], start=2):
        logger.info("Step %s — searching", i)
        result = search_web(query)
        all_results += f"Search: {query}\n{result}\n{'=' * 40}\n"

    return all_results


def filter_relevant_results(entity: dict, raw_results: str) -> str:
    """Drop any search snippet that's clearly about a different company
    sharing the same or a similar name. This is the main guardrail against
    a wrong-company snippet quietly making it into the final report."""
    logger.info("Step — filtering results against resolved entity")

    if not raw_results.strip():
        return raw_results

    messages = [
        {
            "role": "system",
            "content": (
                "The target company is:\n"
                f"- Name: {entity.get('resolved_name', 'unknown')}\n"
                f"- Domain: {entity.get('domain', 'unknown')}\n"
                f"- Industry: {entity.get('industry', 'unknown')}\n"
                f"- Description: {entity.get('one_line_description', 'unknown')}\n\n"
                "You will be given search results made up of Title/Content blocks. "
                "Remove any block that is clearly about a DIFFERENT company that happens "
                "to share the same or a similar name as the target. Keep blocks that are "
                "ambiguous but plausibly about the target. "
                "Return the filtered results in the exact same Title/Content format, "
                "nothing else — no commentary, no summary."
            ),
        },
        {"role": "user", "content": raw_results[:6000]},
    ]

    filtered = ask_llm(messages, temperature=0)
    return filtered if filtered.strip() else raw_results


def maybe_run_extra_search(company_name: str, entity: dict, all_results: str) -> str:
    """Let the LLM decide if a gap-filling search is needed; run it if so."""
    logger.info(" Step %s — evaluating coverage", MAX_SEARCH_ITERATIONS + 2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research quality checker. "
                "Reply with only YES if there is enough information to write a company report, "
                "or NO followed by one more search query needed. "
                "If you include a query, make sure it includes disambiguating context "
                f"(e.g. industry '{entity.get('industry', '')}' or domain "
                f"'{entity.get('domain', '')}') so it targets the right company."
            ),
        },
        {
            "role": "user",
            "content": f"Do I have enough info about {company_name}?\n\nData collected:\n{all_results[:2000]}",
        },
    ]

    evaluation = ask_llm(messages, temperature=0)
    logger.info("Evaluation: %s", evaluation)

    if evaluation.strip().upper().startswith("NO"):
        extra_query = evaluation.replace("NO", "").strip()
        if extra_query:
            logger.info("Running extra search")
            extra_result = search_web(extra_query)
            extra_result = filter_relevant_results(entity, extra_result)
            all_results += f"Search: {extra_query}\n{extra_result}\n"

    return all_results


def write_report(company_name: str, entity: dict, research: str) -> dict:
    """Returns a structured dict instead of one long string. This is the
    fix for the "hard to read" JSON: a single string with embedded \\n
    escapes is correct JSON but ugly to eyeball raw, and awkward to use
    downstream (e.g. in n8n) since you'd have to parse/split the text
    yourself. Structured fields mean n8n (or anything else consuming this)
    can reference report.overview, report.what_they_do[0], etc. directly."""
    logger.info("Final step — writing report")

    ambiguity_note = None
    if entity.get("ambiguous"):
        others = entity.get("other_candidates", [])
        ambiguity_note = (
            f"'{company_name}' may refer to more than one company "
            f"(also found: {', '.join(others) if others else 'other entities'}). "
            f"This report is about {entity.get('resolved_name', company_name)}"
            f"{' (' + entity['domain'] + ')' if entity.get('domain') else ''}."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a business research analyst.\n\n"
                "Write a clean, professional company research report as STRICT JSON.\n\n"
                "IMPORTANT RULES:\n"
                "- Output ONLY a single JSON object, no markdown, no code fences, no commentary.\n"
                "- Use plain ASCII characters only: straight quotes (\" and '), regular hyphens (-), "
                "no smart quotes, no em/en dashes, no bullet characters.\n"
                "- Keep sentences concise and easy to read.\n"
                "- Only include information supported by the research provided. Do not invent facts.\n"
                "- Only include facts about the specific company identified below — ignore any "
                "research snippets that appear to be about a different, similarly named company.\n\n"
                "JSON SHAPE (exact keys):\n"
                "{\n"
                '  "overview": "2-3 sentences",\n'
                '  "what_they_do": ["bullet", "bullet", "bullet"],\n'
                '  "recent_news": ["bullet", "bullet"],\n'
                '  "key_people": ["Name: Role", "Name: Role"],\n'
                '  "why_they_matter": "1-2 sentences"\n'
                "}\n\n"
                "If key_people or recent_news is unavailable, use an empty array []. "
                "Never invent placeholder names."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Company: {company_name}\n"
                f"Identified as: {entity.get('resolved_name', company_name)} "
                f"({entity.get('domain', 'unknown domain')}, "
                f"{entity.get('industry', 'unknown industry')})\n\n"
                f"Write a report using this research:\n\n{research}"
            ),
        },
    ]

    raw = ask_llm(messages, temperature=0)
    fallback = {
        "overview": "",
        "what_they_do": [],
        "recent_news": [],
        "key_people": [],
        "why_they_matter": "",
    }
    report = _parse_json_response(raw, fallback)
    report["ambiguity_note"] = ambiguity_note
    return report


def format_report_as_text(report: dict) -> str:
    """Renders the structured report dict as the plain-text layout you had
    before, for printing to console / logs / anywhere you want a single
    readable string rather than structured fields."""

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- No reliable public information found."

    key_people = bullets(report.get("key_people", []))
    recent_news = bullets(report.get("recent_news", []))

    text = (
        f"COMPANY OVERVIEW\n{report.get('overview', '')}\n\n"
        f"WHAT THEY DO\n{bullets(report.get('what_they_do', []))}\n\n"
        f"RECENT NEWS OR DEVELOPMENTS\n{recent_news}\n\n"
        f"KEY PEOPLE\n{key_people}\n\n"
        f"WHY THEY MATTER\n{report.get('why_they_matter', '')}"
    )

    if report.get("ambiguity_note"):
        text += f"\n\nNOTE: {report['ambiguity_note']}"

    return text


# -- Delivery ----------------------------------------------------------------

def send_to_n8n(company_name: str, report: dict) -> None:
    """Sends the structured report dict, not a pre-formatted string.
    In n8n you can now reference $json.report.overview,
    $json.report.what_they_do[0], etc. directly instead of parsing text."""
    payload = {
        "company": company_name,
        "report": report,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    response = requests.post(N8N_WEBHOOK_URL, json=payload)

    if response.status_code == 200:
        logger.info("Report sent to n8n successfully!")
    else:
        logger.error("Failed to send to n8n: %s", response.status_code)


# -- Orchestrator ------------------------------------------------------------

def run_agent(company_name: str, context_hint: str = "") -> dict:
    """context_hint: optional extra info to disambiguate the company up
    front (industry, city, known domain). Pass it whenever you have it —
    it's the cheapest way to avoid the wrong-company problem entirely."""
    logger.info("=== Researching: %s ===", company_name)

    entity = identify_company(company_name, context_hint)
    if entity.get("ambiguous"):
        logger.warning(
            "Ambiguous company name '%s'. Candidates: %s. Proceeding with best guess: %s",
            company_name,
            entity.get("other_candidates"),
            entity.get("resolved_name"),
        )

    queries = plan_search_queries(company_name, entity)
    logger.info("Generated %s queries", len(queries))

    all_results = run_searches(queries)
    all_results = filter_relevant_results(entity, all_results)

    all_results = maybe_run_extra_search(company_name, entity, all_results)

    logger.info("Generating report...")
    report = write_report(company_name, entity, all_results)

    logger.info("=== FINAL REPORT READY ===")
    logger.debug("Report:\n%s", format_report_as_text(report))

    return report


# -- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    company = input("Enter company name: ")
    hint = input("Any disambiguating context (industry/city/domain, optional): ").strip()
    report = run_agent(company, hint)
    print(format_report_as_text(report))  # human-readable in the console
    send_to_n8n(company, report)          # structured JSON to n8n