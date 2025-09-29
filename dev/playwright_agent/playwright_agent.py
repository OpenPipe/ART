# parallel_eval.py
# Run many (url, question) pairs in parallel with Playwright MCP; judge each answer 0..1.
# deps: pip install mcp openai python-dotenv

import argparse
import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from openpipe.client import AsyncOpenPipe
from panza import limit_concurrency
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4.1")
SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
DEFAULT_CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))

RUNPOD_BASE_URL = "https://pq2dmneot79chc-8000.proxy.runpod.net/v1"
RUNPOD_API_KEY = "sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86"

PLAYWRIGHT_STDIO = StdioServerParameters(
    command="npx", args=["-y", "@playwright/mcp@latest", "--isolated"]
)

RETURN_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "return_answer",
        "description": "Call this exactly once when you are finished. Provide the final, concise answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final answer to the user.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

SUMMARIZE_PROMPT = """
## Goal
Given **one** raw browser/tool response, output a **compact, decision-ready summary** that preserves:
- Key page info (URL, title, status)
- Actionable elements (buttons, inputs, links, forms)
- **Any human-readable content needed for reasoning**, including **longer descriptive text** like job postings, product descriptions, announcements, or article content.
Avoid dumping the full DOM or repeating unchanged states.

## Input
- A single tool response (may include page URL/title, logs, and a DOM tree with refs and text).

## Output (strict)
- Use **YAML**.
- Sections in this order (omit a section if empty):
  1. `page`: `{ url: <string>, title: <string> }`
  2. `status`: short description of the state if notable (e.g., "Login required", "Form error", "Modal open").
  3. `actions`: list of actionable elements with:
     - `{ ref, role, text, href? }` for clickable items
     - `{ ref, role, label|placeholder, type }` for inputs
     - `{ ref, role, label, options[] }` for selects
  4. `content`: **longer readable text** from the page that could help reasoning, such as:
     - Job descriptions
     - Blog post excerpts
     - Product or service summaries
     - Instructions or multi-paragraph text
  5. `notices`: important warnings, errors, or status messages.
  6. `hints`: short suggestions or cues (e.g., “expand menu for more links”).

### Field Rules
- Always include `ref` for listed actionable elements.
- Use simple `role` values: `link`, `button`, `textbox`, `password`, `select`, `checkbox`, `radio`, `form`, `heading`, `banner`, `modal`.
- `text` for visible labels (trimmed). Use `...` if truncated beyond **300 characters** in `actions`.
- In `content`, allow **multi-paragraph text** if clearly human-readable and relevant, but avoid duplicate or irrelevant blocks like repeated footers.
- No styling, coordinates, or raw markup.

### Selection Heuristics
- Always keep key buttons, links, forms, and navigational elements.
- Include headings or section titles that give context to the page.
- Capture **descriptive text** for context: job listings, detailed offers, important announcements, etc.
- If content is very long, summarize repetitive sections and note `(+N more similar)` in `hints`.

### Special Cases
- Auth gate: `status: Login required`; include username/password inputs and login button in `actions`.
- 404/empty: Set `status` accordingly but keep any suggested links.
- Modal/menu open: Indicate in `status`, and list actionable modal/menu elements.
- Redirects: Note destination in `status` or under `actions`.

## Output Template
```yaml
page:
  url: <string>
  title: <string>
status: <string>
actions:
  - ref: <id>
    role: <role>
    text: <string>
    href: <string>  # for navigational elements
content:
  - <long description or human-readable text block>
notices:
  - <short message>
hints:
  - <short hint>
"""

# SUMMARIZE_PROMPT = (
#     "You are a content minimizer for an MCP tool-using agent. "
#     "Given raw tool output, return ONLY what is necessary for the agent's NEXT tool call. "
#     'Prioritize preserving actionable selectors and context such as: element refs (e.g., ref:"..."), '
#     "href URLs, visible labels/text, input placeholders, titles/headers, and small snippets needed to decide the next action. "
#     "Remove styling/CSS, verbose attributes, long code blocks, repeated boilerplate, and media/base64. "
#     "Keep the result short and strictly textual. Do not add analysis or commentary."
# )

# SUMMARIZE_PROMPT = (
#     "Below is the response from the playwright MCP tool call, which will contain the playwright code ran, console messages, and accessible page state. "
#     "Return a comprehensive and clean version of this tool response, with only things that a human would want to know or look at."
#     "If certain things are too repetitive, deduplicate them."
#     "Make sure to keep refs to elements in the page state, since the human might want to take actions based on these refs (like clicking/navigating/filling out)."
#     "Make sure to keep any human readable text on the page, since the human might want to see it to decide what to do next, and understand what is on the page."
#     "Return ONLY the cleaned up tool response."
# )


# --- Keep the ORIGINAL system prompt from your script ---
def build_system_prompt() -> str:
    return (
        "You are a browsing agent that can use Playwright MCP tools to interact with web pages. "
        "You are given a starting URL and a question to answer from the page(s). "
        "Plan steps briefly, then use the tools. When acting on elements, FIRST call `browser_snapshot` "
        "to obtain stable `ref` values. Then pass those refs to `browser_click`, `browser_type`, etc. "
        "Only navigate to relevant pages. Avoid destructive actions (no forms that modify data). "
        "When you have the final answer, you MUST call the `return_answer` tool with the concise answer."
        "You may only use one tool at a time. Do not make multiple parallel tool calls in one step."
    )


# --- helpers ---


def mcp_tool_to_openai_tool(t):
    name = getattr(t, "name", None) or t.get("name")
    desc = getattr(t, "description", None) or t.get("description", "")
    schema = (
        getattr(t, "inputSchema", None)
        or t.get("inputSchema")
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": schema},
    }


async def summarize_tool_response(tool_response: str) -> str:
    async with AsyncOpenAI() as client:
        r = await client.chat.completions.create(
            model=SUMMARIZE_MODEL,
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": tool_response},
            ],
        )
    return r.choices[0].message.content or "(no text)"


async def call_mcp(
    session: ClientSession, name: str, args: dict, summarize_tool_responses: bool
) -> str:
    r = await session.call_tool(name, args)
    content = getattr(r, "content", None) or r.get("content", [])
    out = []
    for item in content or []:
        t = getattr(item, "type", None) or item.get("type")
        if t == "text":
            out.append(getattr(item, "text", None) or item.get("text", ""))
        else:
            out.append(json.dumps(getattr(item, "data", None) or item))
    tool_response = "\n".join(out).strip() or "(no text)"
    if summarize_tool_responses:
        tool_response = await summarize_tool_response(tool_response)
    return tool_response


# --- single browsing run ---
async def run_single(url: str, question: str, summarize_tool_responses: bool) -> str:
    async with stdio_client(PLAYWRIGHT_STDIO) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            t_resp = await session.list_tools()
            mcp_tools = getattr(t_resp, "tools", None) or t_resp.get("tools", [])
            tools = [mcp_tool_to_openai_tool(t) for t in mcp_tools] + [
                RETURN_ANSWER_TOOL
            ]
            msgs = [
                {"role": "system", "content": build_system_prompt()},
                {
                    "role": "user",
                    "content": f"Start at this URL: {url}\n\nQuestion to answer: {question}\n\nTips: Use browser_navigate to go to the URL. Use browser_snapshot to find element refs. You may click or type if needed.",
                },
            ]
            for _ in range(MAX_STEPS):
                async with AsyncOpenAI(
                    base_url=RUNPOD_BASE_URL, api_key=RUNPOD_API_KEY
                ) as client:
                    resp = await client.chat.completions.create(
                        model="Qwen/Qwen2.5-14B-Instruct",
                        messages=msgs,
                        tools=tools,
                        tool_choice="auto",
                        # temperature=0.2,
                    )
                m = resp.choices[0].message
                tcalls = m.tool_calls or []
                if not tcalls:
                    if (m.content or "").strip():
                        answer = m.content.strip()
                        await report_to_openpipe(msgs, answer, url, question)
                        return answer
                    msgs.append(
                        {
                            "role": "user",
                            "content": "Please use the tools and call return_answer when finished.",
                        }
                    )
                    continue
                tc = tcalls[0]
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if name == "return_answer":
                    answer = str(args.get("answer", "")).strip()
                    await report_to_openpipe(msgs, answer, url, question)
                    return answer
                msgs.append(m.model_dump() | {"tool_calls": [tc]})
                try:
                    tool_text = await call_mcp(
                        session, name, args, summarize_tool_responses
                    )
                except Exception as e:
                    tool_text = f"[tool error] {e}"
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": tool_text,
                    }
                )
    return ""


async def report_to_openpipe(
    messages: List[Dict[str, Any]], answer: str, url: str, question: str
) -> None:
    """Report the final trajectory to OpenPipe.

    Sends the chat messages and a minimal completion payload containing the final answer.
    No-op if OPENPIPE_API_KEY is unset.
    """
    api_key = os.getenv("OPENPIPE_PLAYWRIGHT_API_KEY")
    if not api_key:
        print("OPENPIPE_PLAYWRIGHT_API_KEY is not set")
        return

    try:
        op_client = AsyncOpenPipe(api_key=api_key)

        # Minimal OpenAI-like response payload
        resp_payload: Dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        req_payload: Dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "metadata": {
                "project": "playwright_mcp_eval",
                "url": url,
                "question": question,
                "num_messages": str(len(messages)),
            },
        }

        await op_client.report(
            req_payload=req_payload,
            resp_payload=resp_payload,
            status_code=200,
        )
        # Close underlying httpx client
        await op_client.base_client._client_wrapper.httpx_client.aclose()
    except Exception as e:
        print(f"Error reporting to OpenPipe: {e}")
        # Swallow errors to avoid impacting main flow
        pass


# --- judge ---
JUDGE_PROMPT = (
    "You are an exacting evaluator. Given a URL, question, and the agent's answer, score from 0 to 1 with 2 decimals."
    "Rubric (equal weight):"
    "1) Relevance: addresses the exact question."
    "2) Evidence: clearly grounded in the provided page(s) (no hallucination)."
    "3) Specificity: includes the key detail(s) asked (e.g., exact string, number, title)."
    "4) Concision & Clarity: succinct and unambiguous."
    'Return ONLY a JSON object: {"score": <float 0..1>, "justification": <short reason>}.'
)


async def judge(url: str, question: str, answer: str) -> Dict[str, Any]:
    content = f"URL: {url}\nQuestion: {question}\nAnswer: {answer}\nIf insufficient to judge, penalize accordingly."
    async with AsyncOpenAI() as client:
        r = await client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": content},
            ],
        )
    txt = (r.choices[0].message.content or "{}").strip()
    try:
        js = json.loads(txt)
        s = float(js.get("score", 0))
        s = 0 if math.isnan(s) else max(0.0, min(1.0, s))
        return {"score": s, "justification": str(js.get("justification", ""))}
    except Exception:
        return {"score": 0.0, "justification": f"parse_error: {txt[:200]}"}


# --- dataset runner ---
@dataclass
class Item:
    url: str
    question: str


@limit_concurrency(DEFAULT_CONCURRENCY)
async def worker(item: Item, summarize_tool_responses: bool) -> Dict[str, Any]:
    try:
        answer = await run_single(item.url, item.question, summarize_tool_responses)
    except Exception as e:
        answer = f"[agent_error] {e}"
    j = await judge(item.url, item.question, answer)
    return {
        "url": item.url,
        "question": item.question,
        "answer": answer,
        "score": j["score"],
        "justification": j["justification"],
    }


async def run_dataset(
    data: List[Dict[str, str]], summarize_tool_responses: bool
) -> Dict[str, Any]:
    items = [Item(url=d["url"], question=d["question"]) for d in data]
    results = await tqdm_asyncio.gather(
        *[worker(it, summarize_tool_responses) for it in items],
        desc="Running evaluations",
    )
    avg = sum(r["score"] for r in results) / (len(results) or 1)
    return {"results": results, "avg_score": round(avg, 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Parallel Playwright-MCP eval over a JSON list of {url, question}"
    )
    ap.add_argument(
        "--input_json", required=True, help="Path to JSON list of {url, question}"
    )
    ap.add_argument("--summarize_tool_responses", action="store_true")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    MODEL = args.model
    data = json.load(open(args.input_json))
    out = asyncio.run(run_dataset(data, args.summarize_tool_responses))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(
        json.dumps({"avg_score": out["avg_score"], "n": len(out["results"])}, indent=2)
    )
