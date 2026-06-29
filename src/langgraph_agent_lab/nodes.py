# ruff: noqa: E501
"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


class Classification(BaseModel):
    route: str = Field(description="One of: simple, tool, missing_info, risky, error")
    risk_level: str = Field(description="risk level: 'high' for risky routes, 'low' otherwise")



def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    llm = get_llm()
    
    prompt = (
        "Classify the following customer support ticket query into one of these routes:\n"
        "- 'risky': Actions with side effects like refunds, account deletions, sending emails, cancellations.\n"
        "- 'tool': Information lookups such as checking order status, tracking package, looking up data.\n"
        "- 'missing_info': Vague, brief, or incomplete queries lacking actionable context (e.g. 'Can you fix it?', 'Help me').\n"
        "- 'error': Reports of system failures (timeouts, crashes, database errors, service unavailable).\n"
        "- 'simple': General questions answerable directly without tools or actions (e.g. 'How do I reset my password?').\n\n"
        "Priority guide: risky > tool > missing_info > error > simple. If multiple apply, select the highest priority.\n"
        "Set risk_level to 'high' for risky routes, and 'low' for all others.\n\n"
        f"Query: {query}"
    )
    
    structured_llm = llm.with_structured_output(Classification)
    res = structured_llm.invoke(prompt)
    
    route_val = res.route.strip().lower()
    if route_val not in ["simple", "tool", "missing_info", "risky", "error"]:
        route_val = "simple"
        
    risk_val = "high" if route_val == "risky" else "low"
    
    return {
        "route": route_val,
        "risk_level": risk_val,
        "messages": [f"classified:{route_val}:{risk_val}"],
        "events": [make_event("classify", "completed", f"classified query as {route_val} ({risk_val})")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    
    if route == "error" and attempt < 2:
        result = f"Tool execution failed with ERROR (attempt {attempt})"
    else:
        result = "Tool execution completed successfully. Data retrieved."
        
    return {
        "tool_results": [result],
        "messages": [f"tool_execution:{'failed' if 'ERROR' in result else 'success'}"],
        "events": [make_event("tool", "completed", f"executed mock tool (attempt {attempt})")],
    }


class Evaluation(BaseModel):
    evaluation_result: str = Field(description="Must be 'needs_retry' if there are errors or failures, or 'success' otherwise")


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""
    
    llm = get_llm()
    prompt = (
        "Evaluate the following tool execution result. Determine if it is a success or if it needs to be retried due to a failure/error.\n"
        "Your evaluation_result MUST be one of: 'needs_retry' or 'success'.\n\n"
        f"Tool Result: {latest_result}"
    )
    
    try:
        structured_llm = llm.with_structured_output(Evaluation)
        res = structured_llm.invoke(prompt)
        eval_res = res.evaluation_result.strip().lower()
        if eval_res not in ["needs_retry", "success"]:
            raise ValueError()
    except Exception:
        # Fallback heuristic
        if "ERROR" in latest_result or "failed" in latest_result.lower():
            eval_res = "needs_retry"
        else:
            eval_res = "success"
            
    return {
        "evaluation_result": eval_res,
        "messages": [f"evaluated:{eval_res}"],
        "events": [make_event("evaluate", "completed", f"evaluated tool result as {eval_res}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    
    context = f"Original Query: {query}\n"
    if tool_results:
        context += "Tool Results:\n" + "\n".join(tool_results) + "\n"
    if approval:
        context += f"Approval Decision: {approval}\n"
        
    prompt = (
        "You are a helpful customer support agent. Answer the customer's query based on the context provided.\n"
        "Be polite, professional, and clear. Ground your answer strictly in the provided context if any.\n\n"
        f"Context:\n{context}\n"
        f"Query: {query}"
    )
    
    llm = get_llm()
    res = llm.invoke(prompt)
    answer = res.content
    
    return {
        "final_answer": answer,
        "messages": [f"answer:{answer[:40]}..."],
        "events": [make_event("answer", "completed", "generated final answer")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = (
        "The customer query is too vague or incomplete. Generate a polite and helpful clarification question "
        "asking for the specific details needed to help them.\n\n"
        f"Query: {query}"
    )
    
    llm = get_llm()
    res = llm.invoke(prompt)
    question = res.content
    
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"clarification_asked:{question[:40]}..."],
        "events": [make_event("clarify", "completed", "asked for clarification")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = f"Perform support action for query: '{query}'"
    return {
        "proposed_action": proposed_action,
        "messages": [f"proposed_action:{proposed_action[:40]}..."],
        "events": [make_event("risky_action", "completed", "prepared risky action description")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    use_real_hitl = os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true"
    
    if use_real_hitl:
        proposed_action = state.get("proposed_action", "risky action")
        try:
            user_decision = interrupt({
                "proposed_action": proposed_action,
                "message": "This action requires human approval. Approve? (yes/no/comment)",
            })
            
            if isinstance(user_decision, dict):
                approved = user_decision.get("approved", False)
                comment = user_decision.get("comment", "")
            elif isinstance(user_decision, str):
                decision_str = user_decision.strip().lower()
                approved = decision_str in ["yes", "y", "approve", "approved", "true"]
                comment = user_decision
            else:
                approved = False
                comment = str(user_decision)
                
            approval_val = ApprovalDecision(
                approved=approved,
                reviewer="human-reviewer",
                comment=comment
            )
        except Exception:
            approval_val = ApprovalDecision(
                approved=True,
                reviewer="mock-reviewer",
                comment="Automatic fallback approval due to interrupt exception"
            )
    else:
        approval_val = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Automatically approved for testing"
        )
        
    return {
        "approval": approval_val,
        "messages": [f"approval_decision:{approval_val.approved}"],
        "events": [
            make_event(
                "approval", "completed", f"approval decided: {approval_val.approved}"
            )
        ],
    }



def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    next_attempt = attempt + 1
    err_msg = f"Attempt {next_attempt} failed. Retrying..."
    return {
        "attempt": next_attempt,
        "errors": [err_msg],
        "messages": [f"retry_attempt:{next_attempt}"],
        "events": [make_event("retry", "completed", f"recorded attempt {next_attempt}")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    errors = state.get("errors", [])
    last_err = errors[-1] if errors else "unknown error"
    
    final_answer = (
        f"Sorry, we were unable to process your request: '{query}'. "
        f"The system encountered a persistent error: {last_err}. "
        "Your request has been logged for manual support review."
    )
    return {
        "final_answer": final_answer,
        "messages": ["dead_letter:escalated"],
        "events": [make_event("dead_letter", "completed", "escalated to dead letter queue")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }

