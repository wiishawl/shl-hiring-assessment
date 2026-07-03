"""
The complete /chat handler logic. Combines:
  1. llm_router.route()   -- decide intent + extract facets/compare targets
  2. retrieval.retrieve() -- turn facets into real catalog matches
  3. _compare_items()     -- look up named items and generate a grounded comparison

Produces the EXACT response shape the API spec requires:
  {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""
import os
import json
import difflib
from dotenv import load_dotenv
from google import genai
from google.genai import types

import llm_router
import retrieval

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CHAT_MODEL = "gemini-2.5-flash-lite"

_catalog = retrieval._catalog  # reuse the already-loaded catalog


import re

# Pre-extract official abbreviations, but ONLY when the abbreviation
# immediately follows that item's OWN name text (e.g. "Global Skills
# Assessment (GSA)"). This avoids false matches where one item's
# description merely *mentions* another item's name+abbreviation in
# passing (e.g. a "Development Report" whose description explains it's
# based on the "Global Skills Assessment (GSA)").
_ABBREVIATION_MAP = {}
for _item in _catalog:
    _name_escaped = re.escape(_item['name'].rstrip(')').split('(')[0].strip())
    _pattern = _name_escaped + r'\s*\(([A-Z][A-Z0-9\+]{1,6})\)'
    _match = re.search(_pattern, _item.get('description', ''))
    if _match:
        _ABBREVIATION_MAP[_match.group(1)] = _item['name']


def _find_catalog_item_by_name(name_query, cutoff=0.4):
    """Fuzzy-match a user-mentioned assessment name against the real catalog.
    Tries, in order: exact/near name match, substring match, known
    abbreviation match (e.g. 'GSA' -> 'Global Skills Assessment')."""
    all_names = [item["name"] for item in _catalog]

    matches = difflib.get_close_matches(name_query, all_names, n=1, cutoff=cutoff)
    if matches:
        matched_name = matches[0]
        for item in _catalog:
            if item["name"] == matched_name:
                return item

    lowered = name_query.lower()
    for item in _catalog:
        if lowered in item["name"].lower() or item["name"].lower() in lowered:
            return item

    # Abbreviation fallback -- check exact and stripped-punctuation forms
    stripped_query = re.sub(r'[^A-Za-z0-9]', '', name_query).upper()
    for abbr, full_name in _ABBREVIATION_MAP.items():
        if abbr.upper() == name_query.strip().upper() or abbr.upper() == stripped_query:
            for item in _catalog:
                if item["name"] == full_name:
                    return item

    return None


def _format_recommendation(item):
    """Convert an internal catalog item into the exact API recommendation shape."""
    test_type = item.get("test_type", [])
    primary_type = test_type[0] if test_type else ""
    return {
        "name": item["name"],
        "url": item.get("url") or item.get("link"),
        "test_type": primary_type,
    }


import re


def _parse_duration_minutes(duration_str):
    """Extract a number of minutes from strings like '18 minutes'. Returns None if unparseable."""
    if not duration_str:
        return None
    match = re.search(r'(\d+)', duration_str)
    return int(match.group(1)) if match else None


def _apply_constraints(results, constraints, catalog_by_name):
    """Re-rank by seniority match, hard-filter by max duration, using real catalog fields."""
    if not constraints:
        return results

    max_duration = constraints.get("max_duration_minutes")
    seniority = (constraints.get("seniority") or "").lower()

    filtered = []
    for r in results:
        full_item = catalog_by_name.get(r["name"])
        if not full_item:
            filtered.append(r)
            continue
        if max_duration:
            mins = _parse_duration_minutes(full_item.get("duration"))
            if mins is not None and mins > max_duration:
                continue  # hard-filter: exceeds stated duration limit
        filtered.append(r)

    if seniority:
        def seniority_match(r):
            full_item = catalog_by_name.get(r["name"])
            if not full_item:
                return 0
            levels = " ".join(full_item.get("job_levels", [])).lower()
            return 1 if seniority in levels else 0
        filtered.sort(key=seniority_match, reverse=True)

    return filtered


def _handle_recommend_or_refine(facets, intent, constraints=None):
    results = retrieval.retrieve(facets, top_k_per_facet=8, final_top_n=10)
    catalog_by_name = {item["name"]: item for item in _catalog}
    results = _apply_constraints(results, constraints, catalog_by_name)

    if not results:
        return {
            "reply": "I couldn't find catalog assessments matching that (your duration/seniority constraints may be too narrow). Could you loosen them or tell me more?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    verb = "updated the shortlist" if intent == "refine" else "found a shortlist"
    all_names = ", ".join(r["name"] for r in results)
    reply = f"Got it — I {verb} of {len(results)} assessments: {all_names}."
    return {
        "reply": reply,
        "recommendations": [_format_recommendation(r) for r in results],
        "end_of_conversation": False,
    }


def _handle_compare(compare_names):
    found_items = []
    not_found = []
    for name in compare_names:
        item = _find_catalog_item_by_name(name)
        if item:
            found_items.append(item)
        else:
            not_found.append(name)

    if len(found_items) < 2:
        missing = ", ".join(not_found) if not_found else "one of the assessments"
        return {
            "reply": f"I couldn't find {missing} in the SHL catalog. Could you check the name and try again?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Build grounded context from REAL catalog data only
    context_blocks = []
    for item in found_items:
        context_blocks.append(
            f"Name: {item['name']}\n"
            f"Category: {', '.join(item.get('keys', []))}\n"
            f"Duration: {item.get('duration', 'N/A')}\n"
            f"Job levels: {', '.join(item.get('job_levels', []))}\n"
            f"Description: {item.get('description', '')}"
        )
    grounding_text = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "Using ONLY the following real SHL catalog data, write a concise comparison "
        "(3-5 sentences) covering what each assessment measures, how they differ, "
        "and when a recruiter would pick one over the other. Do not add any facts "
        "not present in this data.\n\n" + grounding_text
    )

    response = llm_router._generate_with_retry(
        [types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(temperature=0.2),
    )

    return {
        "reply": response.text.strip(),
        "recommendations": [],
        "end_of_conversation": False,
    }


def handle_chat(messages):
    """
    messages: list of {"role": "user"|"assistant", "content": str}
    Returns the exact API response dict.
    """
    routed = llm_router.route(messages)
    intent = routed["intent"]

    if intent == "clarify":
        return {
            "reply": routed["reply_text"],
            "recommendations": [],
            "end_of_conversation": False,
        }

    if intent == "off_topic":
        return {
            "reply": routed["reply_text"],
            "recommendations": [],
            "end_of_conversation": False,
        }

    if intent in ("recommend", "refine"):
        result = _handle_recommend_or_refine(routed["facets"], intent, routed.get("constraints"))
        if routed.get("task_seems_complete"):
            result["end_of_conversation"] = True
        return result

    if intent == "compare":
        return _handle_compare(routed["compare_items"])

    # Fallback safety net -- should not normally reach here
    return {
        "reply": "Could you tell me more about the role or skills you'd like to assess?",
        "recommendations": [],
        "end_of_conversation": False,
    }


if __name__ == '__main__':
    conversation = [
        {"role": "user", "content": "Hiring a Java developer who works with stakeholders, mid-level, 4 years experience"}
    ]
    result = handle_chat(conversation)
    print(json.dumps(result, indent=2))