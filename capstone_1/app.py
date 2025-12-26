import json
import os

import streamlit as st
from openai import OpenAI

from rag import retrieve, format_context, should_offer_ticket
from tools import create_support_ticket, get_company_info
from dotenv import load_dotenv

load_dotenv()

COMPANY_NAME = os.getenv("COMPANY_NAME", "ACME Motors")
COMPANY_SUPPORT_EMAIL = os.getenv("COMPANY_SUPPORT_EMAIL", "support@acme.example")
COMPANY_SUPPORT_PHONE = os.getenv("COMPANY_SUPPORT_PHONE", "+34 600 000 000")

# LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vector DB
CHROMA_DIR = os.getenv("CHROMA_DIR", "chromadb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")

# Issue tracker: GitHub Issues (default)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

st.set_page_config(page_title="Customer Support AI", page_icon="💬", layout="wide")


client = OpenAI(api_key=OPENAI_API_KEY)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket in the issue tracking system when the user asks or when the answer is not found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {"type": "string"},
                    "user_email": {"type": "string"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["user_name", "user_email", "summary", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get company name and contact info (email, phone).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM = """You are a customer support assistant.
You must:
- Answer using the provided context when possible.
- Always cite sources with filename and page when the answer uses documents.
- If the answer is not found in context, say you couldn't find it and suggest creating a support ticket.
- If the user asks to create a support ticket, call the create_support_ticket tool.
- Keep conversation coherent using the chat history.
"""

def render_citations(cites):
    # cites is list of {source, page}
    uniq = []
    for c in cites:
        key = (c.get("source"), c.get("page"))
        if key not in uniq:
            uniq.append(key)
    if not uniq:
        return ""
    lines = []
    for src, page in uniq:
        if page:
            lines.append(f"- {src}, page {page}")
        else:
            lines.append(f"- {src}")
    return "\n".join(lines)

st.title("💬 Customer Support AI (RAG + Tickets)")

with st.sidebar:
    st.header("User details (for tickets)")
    user_name = st.text_input("Name", value=st.session_state.get("user_name", ""))
    user_email = st.text_input("Email", value=st.session_state.get("user_email", ""))
    st.session_state["user_name"] = user_name
    st.session_state["user_email"] = user_email

    info = get_company_info()
    st.markdown("### Company info")
    st.write(info["company_name"])
    st.write(info["support_email"])
    st.write(info["support_phone"])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": SYSTEM}]

# Display conversation
for m in st.session_state["messages"]:
    if m["role"] in ("user", "assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

user_prompt = st.chat_input("Ask a question…")

if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})

    # Retrieve context
    chunks = retrieve(user_prompt, k=6)
    context_text, cites = format_context(chunks)
    not_found = should_offer_ticket(chunks)

    # Add a “hidden” assistant message with context for the LLM
    augmented_user = f"""User question:
{user_prompt}

Company info:
{json.dumps(get_company_info(), ensure_ascii=False)}

Context (may be empty or irrelevant):
{context_text}

Instruction:
If you use the context, include citations as: (Source: filename p.X) in the answer.
If not found, say you couldn't find it and suggest creating a ticket.
"""

    # Call LLM with tools
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state["messages"] + [{"role": "user", "content": augmented_user}],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = resp.choices[0].message

    # If tool call happens
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            if fn == "create_support_ticket":
                # Fill missing user details if possible
                args.setdefault("user_name", st.session_state.get("user_name", ""))
                args.setdefault("user_email", st.session_state.get("user_email", ""))

                result = create_support_ticket(**args)
                st.session_state["messages"].append(
                    {"role": "tool", "tool_call_id": tc.id, "name": fn, "content": json.dumps(result)}
                )

        # Final answer after tool execution
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state["messages"],
            temperature=0.2,
        )
        final = resp2.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": final})

        with st.chat_message("assistant"):
            st.markdown(final)

    else:
        answer = msg.content or ""
        # If heuristic says not found and model didn’t suggest it, add hint
        if not_found and "ticket" not in answer.lower():
            answer += "\n\nI couldn’t find this in the documents. If you want, I can create a support ticket."

        # Add explicit citation block (even if model already cited inline)
        if cites and any(c.get("source") for c in cites):
            answer += "\n\n**Sources:**\n" + render_citations(cites)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
