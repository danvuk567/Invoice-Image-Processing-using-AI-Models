import base64
import re
from typing import Annotated, List, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import json


# =============================================================================
# AGENT STATE
# =============================================================================

class InvoiceState(TypedDict):
    messages: Annotated[list, add_messages]
    # This will hold your list: [{"category": "...", "amount": "..."}]
    extracted_items: list[dict]


# =============================================================================
# AGENT
# =============================================================================

class Agent:
    """
    A multi-stage Agent supporting Vision, Structured JSON extraction, and Analysis.
    
    Architecture:
    1. Agent 1: Vision (Text extraction from images).
    2. Agent 2: Parser (Structured JSON conversion - No Images).
    3. Agent 3: Analyst (Conversational insights - No Images).

    Args:
        llm_provider (str): Provider name — "OpenAI", "Gemini", "Groq", "HuggingFace".
        model_name (str): Model identifier passed to the provider.
        temperature (float): Sampling temperature (0.0 = deterministic).
        api_key (str, optional): API key for the provider.
        tools (list, optional): LangChain @tool functions available to the agent.
        system_prompt (str, optional): System-level instruction prepended to every call.
        memory (bool): If True, enables multi-turn conversation memory via MemorySaver.
                       Each thread_id maintains its own isolated conversation history.
    """

    def __init__(
        self,
        llm_provider: str = "OpenAI",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str = None,
        tools: list = None,
        system_prompt: str = None,
    ):
        self.llm_provider = llm_provider
        self.model_name   = model_name
        self.temperature  = temperature
        self.api_key      = api_key
        self.tools        = tools or []
        self.system_prompt = system_prompt

        # Initialise the base LLM for the configured provider
        self.llm = self._create_llm()

        # If tools are provided, bind them to the LLM so it knows they exist
        # and can generate tool-call requests in its responses
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)


    # -------------------------------------------------------------------------
    # LLM SETUP
    # -------------------------------------------------------------------------

    def _create_llm(self) -> BaseChatModel:
        """Instantiate the LangChain chat model for the configured provider.

        API keys are passed explicitly rather than relying on environment
        variable naming conventions which differ per provider.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If the provider name is not recognised.
            ImportError: If the required integration package is not installed.
        """
        if self.llm_provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
            )

        elif self.llm_provider == "Google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key,
            )

        elif self.llm_provider == "Groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
            )

        elif self.llm_provider == "HuggingFace":
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name,
                temperature=self.temperature,
                huggingfacehub_api_token=self.api_key,
            )
            return ChatHuggingFace(llm=endpoint)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _get_text_only_history(self, messages: list) -> list:
        """
        Helper to filter out base64 image data from the message history.
        Ensures that downstream agents (Parser and Analyst) stay 
        text-only and cost-efficient.
        """
        processed = []
        for m in messages:
            # If the message content is a list (typical for multimodal/vision inputs)
            if isinstance(m, HumanMessage) and isinstance(m.content, list):
                # We loop through the list and extract ONLY the 'text' parts
                # Ignoring any parts where the type is 'image_url'
                text_part = "".join([
                    part["text"] 
                    for part in m.content 
                    if part.get("type") == "text"
                ])
                # Replace the multimodal message with a simple text-only message
                processed.append(HumanMessage(content=text_part))
            else:
                # If it's already an AIMessage (like the output from Agent 1 or 2)
                # or a simple text HumanMessage, keep it as is.
                processed.append(m)
        
        return processed

    def vision_node(self, state: InvoiceState) -> dict:
        """
        Agent 1 Node: Processes images and populates the structured state.
        """
        messages = state.get("messages", [])
        
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        # 1. Run the LLM
        response = self.llm.invoke(messages)
        
        # 2. Extract the string content
        # response is an object, response.content is the raw [{...}] string
        raw_content = response.content 

        # 3. Parse that string into a real Python list
        extracted_data = []
        try:
            # We look for the JSON array in the text
            json_match = re.search(r"\[\s*\{.*\}\s*\]", raw_content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group(0))
            else:
                # Fallback if the LLM didn't use markdown blocks
                extracted_data = json.loads(raw_content)
        except Exception as e:
            print(f"Parsing failed: {e}")

        # 4. Return BOTH keys
        # 'messages' updates the chat history for your st.write_stream
        # 'extracted_items' updates the data slot for your next Agent
       # return {
       #     "messages": [response], 
       #     "extracted_items": extracted_data
       # }

        return {
          "messages": [AIMessage(content=response.content, name="extraction_result")],
          "extracted_items": extracted_data
       }

    def analyst_node(self, state: InvoiceState) -> dict:
        """
        Agent 2 Node: The Analyst.
        Reads the JSON data from the state and explains it to the user.
        """
        # 1. Access the structured data directly
        data = state.get("extracted_items", [])
        
        # 2. Get the latest user message (their question)
        user_query = state["messages"][-1].content
        
        # 3. Create a specialized prompt that "feeds" the JSON to the LLM
        # so the LLM knows what data it's talking about.
        prompt = f"""
        The following data was extracted from the invoice: {json.dumps(data)}
        Look at the items and summarize the expenses under general categories like groceries, utilities, auto parts, etc...
        Show the total for each category, calculate them from the data above.
        Respond to any user reqests such as showing raw expenses, highest expenses, etc..
        
        The user is asking: {user_query}
        
        Please provide a helpful, non-JSON response.
        """
        
        # 4. Invoke LLM with the text-only history + this data context
        response = self.llm.invoke([SystemMessage(content=prompt)])
        
        # 5. Return a standard message (LangGraph appends this to history)
        #return {"messages": [response]}

        return {
          "messages": [AIMessage(content=response.content, name="analysis_result")]
        }


    # -------------------------------------------------------------------------
    # MESSAGE BUILDERS
    # -------------------------------------------------------------------------

    def _build_image_messages(
        self,
        user_prompt: str,
        image_data: bytes | list[bytes],
        mime_type: str | list[str] = "image/jpeg",
    ) -> list:
        """Build a multimodal HumanMessage containing text and one or more images.

        Normalises single image / single mime_type inputs to lists so the
        same code path handles all cases. Each image is base64-encoded and
        embedded as a data URL — the standard LangChain vision format across
        all supported providers.

        Args:
            user_prompt: Text prompt to send alongside the image(s).
            image_data: Single image bytes or list of image bytes.
            mime_type: Single MIME type string (applied to all images) or a
                       list of MIME types matching each image. Defaults to
                       "image/jpeg".

        Returns:
            A list containing a single multimodal HumanMessage.
        """
        # Normalise to lists so single and multiple images use the same path
        if isinstance(image_data, bytes):
            image_data = [image_data]
        if isinstance(mime_type, str):
            mime_type = [mime_type] * len(image_data)
        elif isinstance(mime_type, list) and len(mime_type) == 1:
            mime_type = mime_type * len(image_data)

        # Start content block with the text prompt
        content = [{"type": "text", "text": user_prompt}]

        # Append each image as a separate base64-encoded content block
        for img, mime in zip(image_data, mime_type):
            image_b64 = base64.b64encode(img).decode("utf-8")
            image_url = f"data:{mime};base64,{image_b64}"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "auto"
                }
            })

        return [HumanMessage(content=content)]

    # -------------------------------------------------------------------------
    # SHARED PIPELINE
    # -------------------------------------------------------------------------

    def prepare(
        self,
        user_prompt: str,
        image_data: bytes | list[bytes] | None,
        mime_type: str | list[str],
    ) -> dict:
        """Build the initial graph state and config shared by run() and stream().

        Centralising this logic means run() and stream() never duplicate
        message building or config construction — they just call _prepare()
        and pass the results to graph.invoke() or graph.stream() respectively.

        Args:
            user_prompt: The user's text prompt.
            image_data: Optional image bytes or list of image bytes.
            mime_type: MIME type(s) for the image(s).
            thread_id: Conversation thread ID for memory isolation.

        Returns:
            A tuple of (initial_state dict, config dict) ready to pass
            directly to graph.invoke() or graph.stream().
        """
        # Build messages with or without images
        if image_data is not None:
            messages = self._build_image_messages(user_prompt, image_data, mime_type)
        else:
            messages = [HumanMessage(content=user_prompt)]

        return {"messages": messages}

       