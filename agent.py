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

LLM_MODEL = "llama-3.3-70b-versatile"
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


def ask_llm(messages: list[dict]) -> str:
    response = groq_client.chat.completions.create(model=LLM_MODEL, messages=messages)
    return response.choices[0].message.content


# -- Agent steps -------------------------------------------------------------

def plan_search_queries(company_name: str) -> list[str]:
    """Ask the LLM to produce 3 targeted search queries for the company."""
    logger.info("Step 1 - Planning searches")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research planning agent.\n"
                "Given a company name, output exactly 3 search queries to research it thoroughly.\n"
                "Output ONLY a numbered list like this:\n"
                "1. query one\n"
                "2. query two\n"
                "3. query three\n"
                "Nothing else."
            ),
        },
        {"role": "user", "content": f"Plan research queries for: {company_name}"},
    ]

    plan = ask_llm(messages)
    logger.info("Search plan: %s", plan)

    queries = [
        line.split(".", 1)[-1].strip()
        for line in plan.strip().splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]

    # Fallback if LLM didn't follow the numbered-list format
    if not queries:
        queries = [
            f"{company_name} company overview",
            f"{company_name} recent news 2025 2026",
            f"{company_name} key people leadership",
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


def maybe_run_extra_search(company_name: str, all_results: str) -> str:
    """Let the LLM decide if a gap-filling search is needed; run it if so."""
    logger.info(" Step %s — evaluating coverage", MAX_SEARCH_ITERATIONS + 2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research quality checker. "
                "Reply with only YES if there is enough information to write a company report, "
                "or NO followed by one more search query needed."
            ),
        },
        {
            "role": "user",
            "content": f"Do I have enough info about {company_name}?\n\nData collected:\n{all_results[:2000]}",
        },
    ]

    evaluation = ask_llm(messages)
    logger.info("Evaluation: %s", evaluation)

    if evaluation.strip().upper().startswith("NO"):
        extra_query = evaluation.replace("NO", "").strip()
        if extra_query:
            logger.info("Running extra search")
            extra_result = search_web(extra_query)
            all_results += f"Search: {extra_query}\n{extra_result}\n"

    return all_results


def write_report(company_name: str, research: str) -> str:
    logger.info("Final step — writing report")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a business research analyst.\n"
                "Write a clean structured company report using the research provided.\n\n"
                "Format:\n"
                "1. Company Overview (2-3 sentences)\n"
                "2. What They Do (bullet points)\n"
                "3. Recent News or Developments\n"
                "4. Key People (if found)\n"
                "5. Why They Matter (1-2 sentences)\n\n"
                "Be concise and professional."
            ),
        },
        {
            "role": "user",
            "content": f"Write a report on {company_name} using this research:\n\n{research}",
        },
    ]

    return ask_llm(messages)


# -- Delivery ----------------------------------------------------------------

def send_to_n8n(company_name: str, report: str) -> None:
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

def run_agent(company_name: str) -> str:
    logger.info("=== Researching: %s ===", company_name)

    queries = plan_search_queries(company_name)
    logger.info("Generated %s queries", len(queries))

    all_results = run_searches(queries)

    all_results = maybe_run_extra_search(company_name, all_results)

    logger.info("Generating report...")
    report = write_report(company_name, all_results)

    logger.info("=== FINAL REPORT READY ===")
    logger.debug("Report:\n%s", report)

    return report


# -- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    company = input("Enter company name: ")
    run_agent(company)
