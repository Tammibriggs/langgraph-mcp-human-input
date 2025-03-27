import os
from typing_extensions import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
from langgraph.graph.message import add_messages

load_dotenv()

global_llm_with_tools = None

# Configure LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

# Configure server parameters
server_params = StdioServerParameters(
    command="python",
    args=["permit_mcp.py"],
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def call_llm(state):
    """Handle LLM calls."""
    response = await global_llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


def route_after_llm(state) -> Literal[END, "human_review_node"]:
    """Route logic after LLM processing."""
    return END if len(state["messages"][-1].tool_calls) == 0 else "human_review_node"


async def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
    """Handle human review process."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]

    high_risk_tools = ['approve_access_request', 'approve_operation_approval']
    if tool_call["name"] not in high_risk_tools:
        return Command(goto="run_tool")

    human_review = interrupt({
        "question": "Do you approve this tool call? (yes/no)",
        "tool_call": tool_call,
    })

    review_action = human_review["action"]

    if review_action == "yes":
        return Command(goto="run_tool")

    return Command(goto='call_llm', update={"messages": [{
        "role": "tool",
        "content": f"The user declined your request to execute the {tool_call.get('name', 'Unknown')} tool, with arguments {tool_call.get('args', 'N/A')}",
        "name": tool_call["name"],
        "tool_call_id": tool_call["id"],
    }]})


async def setup_graph(tools):
    """Set up and configure the state graph."""
    builder = StateGraph(State)
    run_tool = ToolNode(tools)
    builder.add_node(call_llm)
    builder.add_node('run_tool', run_tool)
    builder.add_node(human_review_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", route_after_llm)
    builder.add_edge("run_tool", "call_llm")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


async def stream_responses(graph, config, invokeWith):
    """Streams responses from the graph."""
    async for event in graph.astream(invokeWith, config, stream_mode='updates'):
        for key, value in event.items():
            if key == 'call_llm':
                content = value["messages"][-1].content
                if content:
                    print('\n' + ", ".join(content)
                          if isinstance(content, list) else content)
            elif key == '__interrupt__':
                value = value[0].value
                tool = value['tool_call']
                print(
                    f"\n[Calling tool {tool['name']} with args {tool['args']}]")

                # Get user approval
                user_input = input(f"{value['question']}: ").strip().lower()
                while user_input not in ["yes", "no"]:
                    user_input = input(
                        f"{value['question']}: ").strip().lower()

                # Resume with user input
                await stream_responses(graph, config,
                                       Command(resume={"action": user_input}))


async def chat_loop(graph):
    """Main chat loop."""
    while True:
        try:
            user_input = input("\nQuery: ").strip()
            if user_input in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            sys_m = """
            Always provide the resource instance key during tool calls, as the ReBAC authorization model is being used. To obtain the resource instance key, use the list_resource_instances tool to view available resource instances.
            
            \nAlways parse the provided data before displaying it.
            \nIf the user has initially provides their ID, user that for subsequest tool calls without asking them for it again.
            """

            invokeWith = {"messages": [
                {"role": "user", "content": sys_m + '\n\n' + user_input}]}
            config = {"configurable": {"thread_id": "1"}}

            # Stream initial response
            await stream_responses(graph, config, invokeWith)

        except Exception as e:
            print(f"Error: {e}")


async def main():
    """Main execution function."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            llm_with_tools = llm.bind_tools(tools)
            graph = await setup_graph(tools)

            global global_llm_with_tools
            global_llm_with_tools = llm_with_tools

            with open("workflow_graph.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
            await chat_loop(graph)


if __name__ == "__main__":
    asyncio.run(main())
