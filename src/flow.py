from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.agents import AgentState

class Flow:
    def __init__(self, agents: list, memory: bool = False):
        self.agents = agents

        if memory:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None
            
        self.graph = self._build_graph()

    def _route_start(self, state: AgentState) -> str:
        """
        Private routing logic to decide the entry point.
        """
        messages = state.get("messages", [])
        if not messages:
            return "agent_0"
            
        # Check if the most recent message contains multimodal content (images)
        last_msg = messages[-1]
        if isinstance(last_msg.content, list):
            return "agent_0"
        
        # If it's just text, skip OCR/Vision and go straight to Analysis
        return "agent_1"

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        node_names = []

        for i, agent in enumerate(self.agents):
            # 1. Create a Unique Name for this agent node
            # If it's a single agent, we can just call it "agent_0"
            name = f"agent_{i}"
            node_names.append(name)
            
            # 2. Add the Agent Node (using the public .node method we created)
            # If agent has a specific 'node_func' attribute, use it; otherwise, default
            node_function = getattr(agent, "node_func", agent.vision_node)
            graph_builder.add_node(name, node_function)

            # 3. Handle Tools for this specific agent
            if agent.tools:
                tool_node_name = f"tools_{i}"
                graph_builder.add_node(tool_node_name, ToolNode(agent.tools))
                
                # Setup the ReAct loop for THIS agent
                graph_builder.add_conditional_edges(
                    name, 
                    tools_condition,
                    {
                        "tools": tool_node_name, 
                        "__end__": "__end__" # tools_condition uses this default
                    }
                )
                graph_builder.add_edge(tool_node_name, name)

        # --- ORCHESTRATION (Connecting the nodes) ---

        # Start at the first agent
        #graph_builder.add_edge(START, node_names[0])

        graph_builder.add_conditional_edges(
            START,
            self._route_start,  # Use self._route_start
            {
                "agent_0": "agent_0",
                "agent_1": "agent_1"
            }
        )

        # Connect agent_0 -> agent_1 -> agent_2
        for j in range(len(node_names) - 1):
            graph_builder.add_edge(node_names[j], node_names[j+1])

        # The last agent goes to END
        graph_builder.add_edge(node_names[-1], END)

        return graph_builder.compile(checkpointer=self.checkpointer)
    
    def run(self, user_prompt: str, image_data=None, thread_id="default"):
        # 1. Fix: prepare only returns the state dictionary
        initial_state = self.agents[0].prepare(user_prompt, image_data, "image/jpeg")
        
        # 2. Define the config for the checkpointer/thread
        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer else {}
        
        # 3. Invoke the graph
        result = self.graph.invoke(initial_state, config)

        # 4. Return the content of the very last message in the sequence
        return result["messages"][-1].content

    def stream(
        self,
        user_prompt: str,
        image_data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] = "image/jpeg",
        thread_id: str = "default",
    ):
        from langchain_core.messages import AIMessageChunk

        # 1. Package the messages using the first agent
        initial_state = self.agents[0].prepare(user_prompt, image_data, mime_type)

        # 2. Consistent session management
        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer else {}

        # 3. Stream logic
        # ADDED: Immediate yield to 'anchor' the Streamlit connection
        yield "" 

        for chunk, metadata in self.graph.stream(
            initial_state,
            config,
            stream_mode="messages",
        ):
            # Determine the target node (the last agent in your list)
            target_node = f"agent_{len(self.agents)-1}"
            current_node = metadata.get("langgraph_node")

            # Only process if we are at the target node
            if current_node == target_node:
                if isinstance(chunk, AIMessageChunk):
                    content = chunk.content
                    if content:
                        # Handle standard string content
                        if isinstance(content, str):
                            yield content
                        # Handle list-based content (common in vision/multi-modal models)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    yield item.get("text", "")