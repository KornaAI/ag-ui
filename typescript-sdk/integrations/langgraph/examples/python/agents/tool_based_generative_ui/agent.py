"""
An example demonstrating tool-based generative UI using LangGraph.
"""

from typing import List, Any
import os

# LangGraph imports
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.graph import MessagesState

# OpenAI imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# List of available images (modify path if needed)
IMAGE_LIST = [
  "Osaka_Castle_Turret_Stone_Wall_Pine_Trees_Daytime.jpg",
  "Tokyo_Skyline_Night_Tokyo_Tower_Mount_Fuji_View.jpg",
  "Itsukushima_Shrine_Miyajima_Floating_Torii_Gate_Sunset_Long_Exposure.jpg",
  "Takachiho_Gorge_Waterfall_River_Lush_Greenery_Japan.jpg",
  "Bonsai_Tree_Potted_Japanese_Art_Green_Foliage.jpeg",
  "Shirakawa-go_Gassho-zukuri_Thatched_Roof_Village_Aerial_View.jpg",
  "Ginkaku-ji_Silver_Pavilion_Kyoto_Japanese_Garden_Pond_Reflection.jpg",
  "Senso-ji_Temple_Asakusa_Cherry_Blossoms_Kimono_Umbrella.jpg",
  "Cherry_Blossoms_Sakura_Night_View_City_Lights_Japan.jpg",
  "Mount_Fuji_Lake_Reflection_Cherry_Blossoms_Sakura_Spring.jpg"
]

# This tool generates a haiku on the server.
# The tool call will be streamed to the frontend as it is being generated.
GENERATE_HAIKU_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_haiku",
        "description": "Generate a haiku in Japanese and its English translation. Also select exactly 3 relevant images from the provided list based on the haiku's theme.",
        "parameters": {
            "type": "object",
            "properties": {
                "japanese": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "An array of three lines of the haiku in Japanese"
                },
                "english": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "An array of three lines of the haiku in English"
                },
                "image_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "An array of EXACTLY THREE image filenames from the provided list that are most relevant to the haiku."
                }
            },
            "required": ["japanese", "english", "image_names"]
        }
    }
}

class AgentState(MessagesState):
    tools: List[Any]

async def chat_node(state: AgentState, config: RunnableConfig):
    """
    The main function handling chat and tool calls.
    """
    # Prepare the image list string for the prompt
    image_list_str = "\n".join([f"- {img}" for img in IMAGE_LIST])

    system_prompt = f"""
        You assist the user in generating a haiku.
        When generating a haiku using the 'generate_haiku' tool, you MUST also select exactly 3 image filenames from the following list that are most relevant to the haiku's content or theme. Return the filenames in the 'image_names' parameter.
        
        Available images:
        {image_list_str}
        
        Dont provide the relavent image names in your final response to the user.
    """

    # Define the model
    model = ChatOpenAI(model="gpt-4o")
    
    # Define config for the model
    if config is None:
        config = RunnableConfig(recursion_limit=25)

    # Bind the tools to the model
    model_with_tools = model.bind_tools(
        [GENERATE_HAIKU_TOOL],
        # Disable parallel tool calls to avoid race conditions
        parallel_tool_calls=False,
    )

    # Run the model to generate a response
    response = await model_with_tools.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ], config)

    # Return Command to end with updated messages
    return Command(
        goto=END,
        update={
            "messages": state["messages"] + [response]
        }
    )

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chat_node", chat_node)

# Add edges
workflow.set_entry_point("chat_node")
workflow.add_edge(START, "chat_node")
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


