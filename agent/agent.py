"""Shared State feature."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
import json
import logging
import os

# Ensure Vertex AI environment variables are set
# These should be configured in your .env file or environment
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false"))
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
import concurrent.futures
import base64
from typing import Dict, List, Any, Optional
from fastapi import FastAPI
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool, ToolContext, AgentTool
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai.types import Content, Part , FunctionDeclaration
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

# Setup logging
log_file = "agent.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProverbsState(BaseModel):
    """List of the proverbs being written."""
    proverbs: list[str] = Field(
        default_factory=list,
        description='The list of already written proverbs',
    )


def set_proverbs(
  tool_context: ToolContext,
  new_proverbs: list[str]
) -> Dict[str, str]:
    """
    Set the list of provers using the provided new list.

    Args:
        "new_proverbs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The new list of proverbs to maintain",
        }

    Returns:
        Dict indicating success status and message
    """
    try:
        # Put this into a state object just to confirm the shape
        new_state = { "proverbs": new_proverbs}
        tool_context.state["proverbs"] = new_state["proverbs"]
        return {"status": "success", "message": "Proverbs updated successfully"}

    except Exception as e:
        return {"status": "error", "message": f"Error updating proverbs: {str(e)}"}



def get_weather(tool_context: ToolContext, location: str) -> Dict[str, str]:
    """Get the weather for a given location. Ensure location is fully spelled out."""
    return {"status": "success", "message": f"The weather in {location} is sunny."}


# Create dedicated agents for built-in tools (following ADK documentation pattern)
search_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='SearchAgent',
    instruction="You're a specialist in Google Search grounding. Use web search to find current, factual information and provide structured findings with source attribution.",
    tools=[GoogleSearchTool()],  # ONLY this tool
)

# Vertex AI configuration helpers for grounding tools
def _build_vertex_model_name(default_model: str = "gemini-2.5-flash") -> str:
    """Return the fully qualified Vertex model name when project/location are set."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if project_id:
        return f"projects/{project_id}/locations/{location}/publishers/google/models/{default_model}"
    return default_model


# Dedicated Maps agent that relies on the official GoogleMapsGroundingTool
maps_agent = LlmAgent(
    model=_build_vertex_model_name(),
    name='MapsAgent',
    instruction=(
        "You are a location research specialist. Answer with concise, fact-based summaries. "
        "Always leverage Google Maps grounding to retrieve up-to-date place details, "
        "addresses, ratings, hours, and contextual insights. Include source references when available."
    ),
    tools=[GoogleMapsGroundingTool()],  # ONLY the built-in Maps grounding tool
)


def _create_runner(agent: BaseGroundedAgent, *, app_name: str, session_service: InMemorySessionService) -> Runner:
    return Runner(
        app_name=app_name,
        agent=agent,
        session_service=session_service,
        artifact_service=InMemoryArtifactService(),
    )


def _ensure_session(
    *,
    app_name: str,
    session_service: InMemorySessionService,
    user_id: str,
    session_id: str,
) -> Session:
    existing_session = session_service.get_session_sync(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    if existing_session:
        return existing_session

    # Create the session using the official API so the runner can retrieve it
    return session_service.create_session_sync(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )


def _build_tool_context(tool_context: ToolContext) -> ToolContext:
    return ToolContext(
        tool_context._invocation_context,
        function_call_id=tool_context.function_call_id,
        event_actions=tool_context.actions,
        tool_confirmation=tool_context.tool_confirmation,
    )


def _run_grounded_agent(
    agent: BaseGroundedAgent,
    *,
    query: str,
    tool_context: ToolContext,
    runner_app_name: str,
    runner_session_service: InMemorySessionService,
    logger_prefix: str,
) -> Dict[str, Any]:
    logger.info(f"{logger_prefix} Running grounded agent for query: '{query}'")
    try:
        state_snapshot = {k: tool_context.state.get(k) for k in tool_context.state.keys()}
        logger.debug(f"{logger_prefix} Tool context state snapshot: {state_snapshot}")
    except Exception as state_error:
        logger.debug(f"{logger_prefix} Unable to snapshot tool context state: {state_error}")

    # Extract session and user info from tool_context.state (CORRECT approach)
    session_id = tool_context.state.get("session:session_id")
    user_id = tool_context.state.get("user:user_id")
    
    # Fallback to user context if direct user_id not available
    if not user_id:
        user_context = tool_context.state.get("user:context", {})
        user_id = user_context.get("user_id")
    
    # If still no user_id, use a default
    if not user_id:
        user_id = "grounding-user"
    
    if not session_id:
        session_id = f"grounding-session-{logger_prefix}-{id(query)}"

    logger.info(f"{logger_prefix} Using user_id: {user_id}, session_id: {session_id}")

    # Create NEW session for grounding agent (don't reuse parent session)
    session = _ensure_session(
        app_name=runner_app_name,
        session_service=runner_session_service,
        user_id=user_id,
        session_id=session_id,
    )

    runner = _create_runner(agent, app_name=runner_app_name, session_service=runner_session_service)

    new_message = Content(role="user", parts=[Part(text=query)])

    # Use basic RunConfig without retrieval_config (not supported)
    run_config = RunConfig()
    logger.info(f"{logger_prefix} Using basic RunConfig for Maps grounding")

    events: List[Event] = []
    try:
        for event in runner.run(
            user_id=user_id,
            session_id=session.id,
            new_message=new_message,
            run_config=run_config,
        ):
            events.append(event)
            logger.info(f"{logger_prefix} Event: {event.author} - {type(event).__name__}")
            try:
                if event.content and getattr(event.content, "parts", None):
                    for idx, part in enumerate(event.content.parts or []):
                        part_summary = {
                            "index": idx,
                            "text": getattr(part, "text", None),
                        }
                        function_call = getattr(part, "function_call", None)
                        if function_call:
                            if hasattr(function_call, "to_dict"):
                                part_summary["function_call"] = function_call.to_dict()
                            else:
                                part_summary["function_call"] = str(function_call)
                        function_response = getattr(part, "function_response", None)
                        if function_response:
                            if hasattr(function_response, "to_dict"):
                                part_summary["function_response"] = function_response.to_dict()
                            else:
                                part_summary["function_response"] = str(function_response)
                        logger.debug(f"{logger_prefix} Event part summary: {part_summary}")
            except Exception as log_error:
                logger.debug(f"{logger_prefix} Failed to log event content: {log_error}")
    except Exception as e:
        logger.error(f"{logger_prefix} Error during runner.run(): {e}")
        return {
            "status": "error",
            "message": f"Runner execution failed: {e}",
            "data": {"response_text": "", "events_count": 0, "tool_events": []},
        }

    logger.info(f"{logger_prefix} Generated {len(events)} events for query: '{query}'")

    response_parts: List[str] = []
    tool_events: List[Any] = []
    for event in events:
        if hasattr(event, "to_dict"):
            tool_events.append({"source": event.author, "payload": event.to_dict()})

        if not event.content:
            continue

        for idx, part in enumerate(event.content.parts or []):
            part_summary = {
                "index": idx,
                "text": getattr(part, "text", None),
            }
            
            # Check for plain text first
            text_content = getattr(part, "text", None)
            if text_content:
                response_parts.append(text_content)
                logger.info(f"{logger_prefix} Found text content: {text_content[:100]}...")
            
            function_response = getattr(part, "function_response", None)
            if function_response:
                try:
                    response_payload = function_response.to_dict()
                except AttributeError:
                    response_payload = {
                        "name": getattr(function_response, "name", ""),
                        "response": getattr(function_response, "response", {}),
                    }

                tool_events.append({
                    "source": "function_response",
                    "payload": response_payload,
                })

                logger.info(
                    f"{logger_prefix} Function response payload: {json.dumps(response_payload, ensure_ascii=False)[:500]}"
                )

                response_body = response_payload.get("response", response_payload)
                try:
                    response_parts.append(json.dumps(response_body, indent=2))
                except (TypeError, ValueError):
                    response_parts.append(str(response_body))

                part_summary["function_response"] = response_payload

            function_call = getattr(part, "function_call", None)
            if function_call:
                if hasattr(function_call, "to_dict"):
                    call_payload = function_call.to_dict()
                else:
                    call_payload = {
                        "name": getattr(function_call, "name", ""),
                        "args": getattr(function_call, "args", {}),
                    }
                tool_events.append({
                    "source": "function_call",
                    "payload": call_payload,
                })
                part_summary["function_call"] = call_payload

                logger.info(
                    f"{logger_prefix} Function call payload: {json.dumps(call_payload, ensure_ascii=False)[:500]}"
                )

            logger.info(f"{logger_prefix} Event part summary: {part_summary}")


    response_text = "\n".join(response_parts).strip()

    if not response_text:
        logger.warning(f"{logger_prefix} No text response produced for query: '{query}'")
        if tool_events:
            logger.info(
                f"{logger_prefix} Tool events present without text response: {json.dumps(tool_events, ensure_ascii=False)[:500]}"
            )
        else:
            logger.warning(
                f"{logger_prefix} No tool events captured for query '{query}'. Vertex AI may not have matching data."
            )
    else:
        logger.info(
            f"{logger_prefix} Final response text (truncated): {response_text[:500]}"
        )

    return {
        "status": "success" if response_text else "error",
        "message": "Grounded response generated" if response_text else "No grounded response",
        "data": {
            "response_text": response_text,
            "events_count": len(events),
            "tool_events": tool_events,
        },
    }


maps_runner_app_name = "maps_grounding_app"
search_runner_app_name = "search_grounding_app"
maps_session_service = InMemorySessionService()
search_session_service = InMemorySessionService()


# Simple wrapper functions that will delegate to AgentTool



def on_before_agent(callback_context: CallbackContext):
    """
    Initialize proverbs state if it doesn't exist.
    """

    if "proverbs" not in callback_context.state:
        # Initialize with default recipe
        default_proverbs =     []
        callback_context.state["proverbs"] = default_proverbs


    return None




# --- Define the Callback Function ---
#  modifying the agent's system prompt to incude the current state of the proverbs list
def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    if agent_name == "ProverbsAgent":
        proverbs_json = "No proverbs yet"
        if "proverbs" in callback_context.state and callback_context.state["proverbs"] is not None:
            try:
                proverbs_json = json.dumps(callback_context.state["proverbs"], indent=2)
            except Exception as e:
                proverbs_json = f"Error serializing proverbs: {str(e)}"
        # --- Modification Example ---
        # Add a prefix to the system instruction
        original_instruction = llm_request.config.system_instruction or Content(role="system", parts=[])
        prefix = f"""You are a helpful assistant for maintaining a list of proverbs.
        This is the current state of the list of proverbs: {proverbs_json}
        When you modify the list of proverbs (wether to add, remove, or modify one or more proverbs), use the set_proverbs tool to update the list."""
        # Ensure system_instruction is Content and parts list exists
        if not isinstance(original_instruction, Content):
            # Handle case where it might be a string (though config expects Content)
            original_instruction = Content(role="system", parts=[Part(text=str(original_instruction))])
        if not original_instruction.parts:
            original_instruction.parts.append(Part(text="")) # Add an empty part if none exist

        # Modify the text of the first part
        modified_text = prefix + (original_instruction.parts[0].text or "")
        original_instruction.parts[0].text = modified_text
        llm_request.config.system_instruction = original_instruction



    return None






# --- Define the Callback Function ---
def simple_after_model_modifier(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Stop the consecutive tool calling of the agent"""
    agent_name = callback_context.agent_name
    # --- Inspection ---
    if agent_name == "ProverbsAgent":
        original_text = ""
        if llm_response.content and llm_response.content.parts:
            # Assuming simple text response for this example
            if  llm_response.content.role=='model' and llm_response.content.parts[0].text:
                original_text = llm_response.content.parts[0].text
                callback_context._invocation_context.end_invocation = True

        elif llm_response.error_message:
            return None
        else:
            return None # Nothing to modify
    return None


proverbs_agent = LlmAgent(
        name="ProverbsAgent",
        model="gemini-2.5-flash",
        instruction=f"""
        When a user asks you to do anything regarding proverbs, you MUST use the set_proverbs tool.

        IMPORTANT RULES ABOUT PROVERBS AND THE SET_PROVERBS TOOL:
        1. Always use the set_proverbs tool for any proverbs-related requests
        2. Always pass the COMPLETE LIST of proverbs to the set_proverbs tool. If the list had 5 proverbs and you removed one, you must pass the complete list of 4 remaining proverbs.
        3. You can use existing proverbs if one is relevant to the user's request, but you can also create new proverbs as required.
        4. Be creative and helpful in generating complete, practical proverbs
        5. After using the tool, provide a brief summary of what you create, removed, or changed        7.

        Examples of when to use the set_proverbs tool:
        - "Add a proverb about soap" → Use tool with an array containing the existing list of proverbs with the new proverb about soap at the end.
        - "Remove the first proverb" → Use tool with an array containing the all of the existing proverbs except the first one"
        - "Change any proverbs about cats to mention that they have 18 lives" → If no proverbs mention cats, do not use the tool. If one or more proverbs do mention cats, change them to mention cats having 18 lives, and use the tool with an array of all of the proverbs, including ones that were changed and ones that did not require changes.

        Do your best to ensure proverbs plausibly make sense.


        IMPORTANT RULES ABOUT WEATHER AND THE GET_WEATHER TOOL:
        1. Only call the get_weather tool if the user asks you for the weather in a given location.
        2. If the user does not specify a location, you can use the location "Everywhere ever in the whole wide world"

        Examples of when to use the get_weather tool:
        - "What's the weather today in Tokyo?" → Use the tool with the location "Tokyo"
        - "Whats the weather right now" → Use the location "Everywhere ever in the whole wide world"
        - Is it raining in London? → Use the tool with the location "London"

        GROUNDING CAPABILITIES (Agent-as-Tool Pattern):
        You now have access to real-time information through specialized research agents:

        SEARCH AGENT:
        - Use SearchAgent for web search, current events, news, or any information requiring real-time data
        - Examples: "What's the latest news about AI?", "Search for information about climate change", "Find recent developments in renewable energy"
        - This agent specializes in Google Search grounding

        MAPS AGENT:
        - Use MapsAgent for location-specific queries, businesses, restaurants, or places
        - Examples: "Find restaurants near me", "What's the address of Central Park?", "Find coffee shops in downtown"
        - This agent specializes in Google Maps grounding

        Always prefer these grounding agents over generic responses when users ask for real-world information.
        The agent-as-tool pattern ensures reliable grounding following ADK best practices.
        """,
        tools=[set_proverbs, get_weather, AgentTool(agent=search_agent), AgentTool(agent=maps_agent)],
        before_agent_callback=on_before_agent,
        before_model_callback=before_model_modifier,
        after_model_callback = simple_after_model_modifier
    )

# Create ADK middleware agent instance
adk_proverbs_agent = ADKAgent(
    adk_agent=proverbs_agent,
    app_name="proverbs_app",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True
)

# Create FastAPI app
app = FastAPI(title="ADK Middleware Proverbs Agent")

# Add the ADK endpoint
add_adk_fastapi_endpoint(app, adk_proverbs_agent, path="/")

if __name__ == "__main__":
    import os
    import uvicorn

    logger.info("🚀 Starting ADK Agent with Grounding Tools")
    logger.info(f"📁 Log file: {os.path.abspath(log_file)}")
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("⚠️  GOOGLE_API_KEY environment variable not set!")
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("   Get a key from: https://makersuite.google.com/app/apikey")
        print()
    else:
        logger.info("✅ GOOGLE_API_KEY found")

    port = int(os.getenv("PORT", 8000))
    logger.info(f"🌐 Starting server on port {port}")
    logger.info("🔧 Available tools: set_proverbs, get_weather, research_current_information, find_places_information")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
