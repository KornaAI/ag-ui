"""
A demo of shared state between the agent and CopilotKit using LangGraph.
"""

import json
from enum import Enum
from typing import Dict, List, Any, Optional
import os

# LangGraph imports
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_core.callbacks.manager import adispatch_custom_event
from langgraph.graph import MessagesState
# OpenAI imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

class SkillLevel(str, Enum):
    """
    The level of skill required for the recipe.
    """
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class SpecialPreferences(str, Enum):
    """
    Special preferences for the recipe.
    """
    HIGH_PROTEIN = "High Protein"
    LOW_CARB = "Low Carb"
    SPICY = "Spicy"
    BUDGET_FRIENDLY = "Budget-Friendly"
    ONE_POT_MEAL = "One-Pot Meal"
    VEGETARIAN = "Vegetarian"
    VEGAN = "Vegan"


class CookingTime(str, Enum):
    """
    The cooking time of the recipe.
    """
    FIVE_MIN = "5 min"
    FIFTEEN_MIN = "15 min"
    THIRTY_MIN = "30 min"
    FORTY_FIVE_MIN = "45 min"
    SIXTY_PLUS_MIN = "60+ min"


GENERATE_RECIPE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_recipe",
        "description": " ".join("""Using the existing (if any) ingredients and instructions, proceed with the recipe to finish it.
        Make sure the recipe is complete. ALWAYS provide the entire recipe, not just the changes.""".split()),
        "parameters": {
            "type": "object",
            "properties": {
                "recipe": {
                    "type": "object",
                    "properties": {
                        "skill_level": {
                            "type": "string",
                            "enum": [level.value for level in SkillLevel],
                            "description": "The skill level required for the recipe"
                        },
                        "special_preferences": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [preference.value for preference in SpecialPreferences]
                            },
                            "description": "A list of special preferences for the recipe"
                        },
                        "cooking_time": {
                            "type": "string",
                            "enum": [time.value for time in CookingTime],
                            "description": "The cooking time of the recipe"
                        },
                        "ingredients": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "icon": {"type": "string", "description": "The icon emoji (not emoji code like '\x1f35e', but the actual emoji like 🥕) of the ingredient"},
                                    "name": {"type": "string"},
                                    "amount": {"type": "string"}
                                }
                            },
                            "description": "Entire list of ingredients for the recipe, including the new ingredients and the ones that are already in the recipe"
                        },
                        "instructions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Entire list of instructions for the recipe, including the new instructions and the ones that are already there"
                        },
                        "changes": {
                            "type": "string",
                            "description": "A description of the changes made to the recipe"
                        }
                    },
                }
            },
            "required": ["recipe"]
        }
    }
}


class AgentState(MessagesState):
    """
    The state of the recipe.
    """
    recipe: Optional[Dict[str, Any]] = None
    tools: List[Any]


async def start_flow(state: Dict[str, Any], config: RunnableConfig):
    """
    This is the entry point for the flow.
    """

    # Initialize recipe if not exists
    if "recipe" not in state or state["recipe"] is None:
        state["recipe"] = {
            "skill_level": SkillLevel.BEGINNER.value,
            "special_preferences": [],
            "cooking_time": CookingTime.FIFTEEN_MIN.value,
            "ingredients": [{"icon": "🍴", "name": "Sample Ingredient", "amount": "1 unit"}],
            "instructions": ["First step instruction"]
        }
        # Emit the initial state to ensure it's properly shared with the frontend
        await adispatch_custom_event(
            "manually_emit_intermediate_state",
            state,
            config=config,
        )
    
    return Command(
        goto="chat_node",
        update={
            "messages": state["messages"],
            "recipe": state["recipe"]
        }
    )


async def chat_node(state: Dict[str, Any], config: RunnableConfig):
    """
    Standard chat node.
    """
    # Create a safer serialization of the recipe
    recipe_json = "No recipe yet"
    if "recipe" in state and state["recipe"] is not None:
        try:
            recipe_json = json.dumps(state["recipe"], indent=2)
        except Exception as e:
            recipe_json = f"Error serializing recipe: {str(e)}"
    
    system_prompt = f"""You are a helpful assistant for creating recipes. 
    This is the current state of the recipe: {recipe_json}
    You can improve the recipe by calling the generate_recipe tool.
    
    IMPORTANT:
    1. Create a recipe using the existing ingredients and instructions. Make sure the recipe is complete.
    2. For ingredients, append new ingredients to the existing ones.
    3. For instructions, append new steps to the existing ones.
    4. 'ingredients' is always an array of objects with 'icon', 'name', and 'amount' fields
    5. 'instructions' is always an array of strings

    If you have just created or modified the recipe, just answer in one sentence what you did. dont describe the recipe, just say what you did.
    """

    # Define the model
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Define config for the model
    if config is None:
        config = RunnableConfig(recursion_limit=25)

    # Use "predict_state" metadata to set up streaming for the write_document tool
    config["metadata"]["predict_state"] = [{
        "state_key": "recipe",
        "tool": "generate_recipe",
        "tool_argument": "recipe"
    }]

    # Bind the tools to the model
    model_with_tools = model.bind_tools(
        [
            *state["tools"],
            GENERATE_RECIPE_TOOL
        ],
        # Disable parallel tool calls to avoid race conditions
        parallel_tool_calls=False,
    )

    # Run the model and generate a response
    response = await model_with_tools.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ], config)

    # Update messages with the response
    messages = state["messages"] + [response]
    
    # Handle tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        
        # Handle tool_call as a dictionary or an object
        if isinstance(tool_call, dict):
            tool_call_id = tool_call["id"]
            tool_call_name = tool_call["name"]
            # Check if args is already a dict or needs to be parsed
            if isinstance(tool_call["args"], dict):
                tool_call_args = tool_call["args"]
            else:
                tool_call_args = json.loads(tool_call["args"])
        else:
            # Handle as an object
            tool_call_id = tool_call.id
            tool_call_name = tool_call.name
            # Check if args is already a dict or needs to be parsed
            if isinstance(tool_call.args, dict):
                tool_call_args = tool_call.args
            else:
                tool_call_args = json.loads(tool_call.args)

        if tool_call_name == "generate_recipe":
            # Update recipe state with tool_call_args
            recipe_data = tool_call_args["recipe"]
            
            # If we have an existing recipe, update it
            if "recipe" in state and state["recipe"] is not None:
                recipe = state["recipe"]
                for key, value in recipe_data.items():
                    if value is not None:  # Only update fields that were provided
                        recipe[key] = value
            else:
                # Create a new recipe
                recipe = {
                    "skill_level": recipe_data.get("skill_level", SkillLevel.BEGINNER.value),
                    "special_preferences": recipe_data.get("special_preferences", []),
                    "cooking_time": recipe_data.get("cooking_time", CookingTime.FIFTEEN_MIN.value),
                    "ingredients": recipe_data.get("ingredients", []),
                    "instructions": recipe_data.get("instructions", [])
                }
            
            # Add tool response to messages
            tool_response = {
                "role": "tool",
                "content": "Recipe generated.",
                "tool_call_id": tool_call_id
            }
            
            messages = messages + [tool_response]
            
            # Explicitly emit the updated state to ensure it's shared with frontend
            state["recipe"] = recipe
            await adispatch_custom_event(
                "manually_emit_intermediate_state",
                state,
                config=config,
            )
            
            # Return command with updated recipe
            return Command(
                goto="start_flow",
                update={
                    "messages": messages,
                    "recipe": recipe
                }
            )

    return Command(
        goto=END,
        update={
            "messages": messages,
            "recipe": state["recipe"]
        }
    )


# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("start_flow", start_flow)
workflow.add_node("chat_node", chat_node)

# Add edges
workflow.set_entry_point("start_flow")
workflow.add_edge(START, "start_flow")
workflow.add_edge("start_flow", "chat_node")
workflow.add_edge("chat_node", END)

# Conditionally use a checkpointer based on the environment
# Check for multiple indicators that we're running in LangGraph dev/API mode
is_fast_api = os.environ.get("LANGGRAPH_FAST_API", "false").lower() == "true"

# Compile the graph
if is_fast_api:
    # For CopilotKit and other contexts, use MemorySaver
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
else:
    # When running in LangGraph API/dev, don't use a custom checkpointer
    graph = workflow.compile()
