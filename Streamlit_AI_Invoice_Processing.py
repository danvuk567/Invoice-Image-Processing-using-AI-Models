import streamlit as st
import uuid
import json
from src.agents import Agent
from src.flow import Flow
from src.image_uploader import ImageUploader


# =============================================================================
# HELPERS
# =============================================================================

def load_models(json_path: str) -> dict:
    """
    Parses a JSON configuration file to retrieve AI model metadata.

    Instead of raising exceptions, this function catches them and displays 
    user-friendly error messages in the Streamlit UI before halting execution.

    Args:
        json_path (str): The file path to the models.json configuration file.

    Returns:
        dict: A nested dictionary of model details if successful.
    
    Error Handling:
        - On FileNotFoundError: Displays error and stops the app.
        - On JSONDecodeError: Displays syntax error details and stops the app.
        - On general Exception: Displays the traceback and stops the app.
    """
    try:
        # Use utf-8 to ensure special characters in descriptions load correctly
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
          
    except FileNotFoundError:
        st.error(f"🚨 **Configuration Missing:** The file '{json_path}' was not found.")
        st.stop() 

    except json.JSONDecodeError as e:
        st.error(f"📑 **Syntax Error:** The file '{json_path}' contains invalid JSON.")
        st.info(f"Check line {e.lineno}, column {e.colno}.")
        st.stop() 

    except Exception as e:
        st.error(f"❌ **Unexpected Error:** An issue occurred while loading '{json_path}'.")
        st.exception(e) # Collapsible technical details for debugging
        st.stop()

def get_models_for_provider(models_data: dict, provider: str) -> dict:
    """
    Extracts the subset of models belonging to a specific AI provider.

    This helper function filters the global models dictionary to return only 
    the models associated with the user's selected provider (e.g., 'Google'). 
    It uses a fallback mechanism to prevent the UI from breaking if a 
    provider key is missing.

    Args:
        models_data (dict): The full dictionary loaded from models.json.
        provider (str): The name of the provider to look up (e.g., "OpenAI").

    Returns:
        dict: A dictionary of models for that provider, or an empty 
              dictionary {} if the provider is not found.
    """
    # Using .get() instead of models_data[provider] is a "defensive" move.
    # If 'provider' isn't a key in the dictionary, it returns the second 
    # argument ({}) instead of raising a KeyError.
    return models_data.get(provider, {})

def render_model_description(info: dict, provider: str) -> None:
    """
    Renders a formatted UI block showing model metadata and pricing.

    This function takes a dictionary of model information and displays it 
    using a mix of Streamlit's native components and raw HTML/CSS. It 
    provides the user with the model's originating company, a brief 
    functional description, and the cost per million tokens.

    Args:
        info (dict): The specific model dictionary from models.json 
                     (e.g., {"Company": "Google", "Input": 3.50, ...}).
        provider (str): The fallback provider name if 'Company' is missing.

    Returns:
        None
    """

    # 1. DATA EXTRACTION: Pull the pricing or default to "N/A" to prevent errors.
    input_cost  = info.get("Input",  "N/A")
    output_cost = info.get("Output", "N/A")

    # 2. HEADER: Displays the building emoji and the company name in bold.
    st.caption(f"🏢 **{info.get('Company', provider)}**")

    # 3. BODY: Displays the model's intended use case or description.
    st.write(info.get("Description", "No description available."))

    # 4. FOOTER (HTML): Creates a small, grey, "pro-style" pricing line.
    # Uses HTML entities: &#128176; (Money Bag) and &#36; (Dollar Sign).
    st.markdown(
        f"<p style='color:grey;font-size:0.85em;margin:0'>"
        f"&#128176; Input: &#36;{input_cost} / 1M tokens"
        f" &nbsp;&middot;&nbsp; "
        f"Output: &#36;{output_cost} / 1M tokens"
        f"</p>",
        unsafe_allow_html=True,
    )

def load_css(css_path: str) -> None:
    """Inject a local CSS stylesheet into the Streamlit app."""
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def on_image_delete() -> None:
    """Callback triggered when the user deletes all uploaded images.

    Does not clear chat history — the conversation continues without images.
    Only clears the image hash so the uploader resets cleanly.
    """
    st.session_state.last_image_hash = None


def build_context_prefix(has_images: bool) -> str:
    """Build an optional context prefix to prepend to the user's message.

    Informs the LLM of any images provided so it can
    factor them into its response without the user repeating this context
    in every message. The prefix is sent to the LLM but not shown in the
    chat history — the user only sees their clean question.

    Args:
        has_images: Whether images are currently uploaded.

    Returns:
        A context prefix string, or empty string if no context exists.
    """
    parts = []

    if has_images:
        parts.append("[image(s) attached — please reference them in your response]")

    if parts:
        # Separate context from the user's actual question with a blank line
        return "\n".join(parts) + "\n\n"

    return ""

def model_selector(slot: str, models_data: dict) -> tuple:
    """
    Renders a Streamlit UI component for selecting an LLM provider and model.

    This function handles the state lifecycle for a specific model 'slot'. If the model 
    selection changes, it clears the chat history, regenerates session IDs, and 
    destroys the existing agent instance to ensure the new model starts fresh.

    Args:
        slot (str): Unique identifier for the UI slot (e.g., 'left', 'right', 'a').
        models_data (dict): Nested dictionary containing provider and model metadata.

    Returns:
        tuple: (provider_name, model_name) or (provider_name, None) if no model selected.
    """

    with st.container(border=True):
        # 1. Initialize Provider Selection
        if slot == "1":
            provider = st.selectbox(
                "Image Analysis Provider", 
                PROVIDERS, 
                key=f"provider_{slot}"
            )
        else:
            provider = st.selectbox(
                "Expense Analysis Provider", 
                PROVIDERS, 
                key=f"provider_{slot}"
            )

        # 2. Fetch models for the chosen provider and filter exclusions
        provider_models = get_models_for_provider(models_data, provider)
        all_names = list(provider_models.keys())

        options = ["Select Model..."] + [m for m in all_names]

        # 3. Render Model Selection
        if slot == "1":
            model = st.selectbox(
                "Image Analysis Model", 
                options, 
                key=f"model_{slot}"
            )
        else:
            model = st.selectbox(
                "Expense Analysis Model", 
                options, 
                key=f"model_{slot}"
            )

        # 4. State Management Logic: Detect model changes
        tracker_key = f"model_tracker_{slot}"
        last_val = st.session_state.get(tracker_key)

        if model and model != "Select Model...":

            if model != last_val:

                if last_val != "Select Model..." and last_val is not None:

                    # Wipe the history for this specific slot
                    st.session_state["messages"] = []
                    
                    # KILL the agent so it rebuilds with the new model name/key
                    if f"agent_{slot}" in st.session_state:
                        del st.session_state[f"agent_{slot}"]
                    
                # UPDATE THE TRACKER (Crucial: sync the tracker to the new model)
                st.session_state[tracker_key] = model

                # Generate a fresh thread ID for the new model
                st.session_state["session_id"] = str(uuid.uuid4())
                    
                # RERUN to refresh the UI and clear the chat boxes
                st.rerun()
            

        # 5. UI Feedback: Show model details if a valid selection is made
        if model and model != "Select Model...":
            # Pull metadata from the dictionary subset
            model_info = provider_models.get(model, {})
            render_model_description(model_info, provider)

            return provider, model

    return provider, None

def clear_chat_callback():
    """
    Performs a targeted reset of the conversation state for a specific UI slot.

    This callback ensures that only the data associated with the provided 'slot' 
    identifier is purged, allowing other slots to remain unaffected. It handles 
    message history deletion, session ID rotation, and agent instance destruction.

    Args:
        slot (str): The unique identifier (e.g., 'left', 'right') representing 
                   the conversation stream to be cleared.
    """
    # 1. Clear the UI History
    # We keep index 0 (User Images) and index 1 (AI JSON)
    # These are hidden by your UI loop, but needed for Agent 2 to function.
    if len(st.session_state.messages) >= 2:
        st.session_state.messages = st.session_state.messages[:2]
    
    # 2. Sync the LangGraph Internal Memory
    # This prevents the AI from 'remembering' old questions internally.
    if "flow_1" in st.session_state:
        flow = st.session_state.flow_1
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        # Get the internal state messages
        state = flow.graph.get_state(config)
        internal_msgs = state.values.get("messages", [])
        
        # Keep only the setup messages (User upload + Vision extraction)
        if len(internal_msgs) >= 2:
            clean_history = internal_msgs[:2]
            
            # Update the graph memory
            flow.graph.update_state(
                config,
                {"messages": clean_history},
                as_node="agent_0" # Reset from the vision node's perspective
            )

    # 3. Maintain UI Stage
    st.session_state.image_analysis_stage = True
    st.session_state.extraction_stage = False
    

def api_key_handler(slot: str, provider: str):
    """
    Manages secure API key entry and persistence for a specific model slot.

    This function implements a state-locked 'vault' mechanism. It provides a 
    toggle between a masked input state and an 'Active' state. When a key is 
    updated, it invalidates any existing agents to ensure credentials are 
    refreshed in the backend.

    Args:
        slot (str): Unique identifier for the UI slot ('left', 'right', etc.).
        provider (str): Name of the LLM provider (e.g., 'OpenAI', 'Anthropic').

    Returns:
        str | None: The active API key from the session vault, or None if empty.
    """

    vault_key = f"confirmed_api_key_{slot}"
    widget_key = f"widget_input_{slot}"
    error_key = f"api_key_error_{slot}"

    # 1. THE PERSISTENCE CHECK
    # If a key is in the vault, show the Success message.
    if st.session_state.get(vault_key):
        st.success(f"✅ {provider} API Key Active")
        if st.button(f"Update {provider} Key", key=f"btn_update_{slot}"):
            st.session_state[vault_key] = "" # Clear vault
            if f"agent_{slot}" in st.session_state:
                del st.session_state[f"agent_{slot}"]
            st.rerun()
        return st.session_state[vault_key]

    # 2. THE INPUT UI: Rendered only if no valid key exists in the vault
    st.markdown(f"**Enter {provider} API Key**")
    
    api_key_input = st.text_input(
        label="API Key Input",
        type="password",
        placeholder="Paste key here...",
        value=st.session_state.get(vault_key, ""), # Pull from vault
        key=widget_key,
        label_visibility="collapsed",
    )

    # 3. VALIDATION & SAVE LOGIC
    if st.button("Enter", use_container_width=True, key=f"btn_save_key_{slot}"):
        if not api_key_input.strip():
            st.session_state[error_key] = "⚠️ API key cannot be blank."
        else:
            # Commit the key to the vault and clear any previous errors
            st.session_state[vault_key] = api_key_input.strip()
            st.session_state[error_key] = "" 

            # Kill the agent instance to force a fresh connection with the new key
            if f"agent_{slot}" in st.session_state:
                del st.session_state[f"agent_{slot}"]

        # Rerun to switch from the Input UI to the Success UI
        st.rerun()

    # 4. ERROR DISPLAY: Persistent error messaging tied to the slot
    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])

    return st.session_state.get(vault_key)

def stream_response(
    flow: Flow,
    user_message: str,
    thread_id: str,
    image_data: list[bytes] | None = None,
    mime_type: str | list[str] = "image/jpeg",
) -> str:
    # 1. Define the generator
    # We don't necessarily need the nonlocal full_response anymore 
    # because st.write_stream returns the full string for us.
    def response_generator():
        for chunk in flow.stream(
            user_prompt=user_message,
            image_data=image_data,
            mime_type=mime_type,
            thread_id=thread_id,
        ):
            yield chunk
    
    # 2. Consume the stream and CAPTURE the result
    # st.write_stream handles the iteration and returns the final text block.
    #full_response = st.write_stream(response_generator())
    full_response = "".join(response_generator())
    
    # 3. Return the captured string to be saved to session_state.messages
    return full_response


# =============================================================================
# INITIALISATION
# =============================================================================

def init_session_state() -> None:
    """
    Initializes and synchronizes all session-state variables on the first run.

    This function acts as the state manager for the application. It ensures that 
    all necessary keys exist in st.session_state before the UI renders. It also 
    handles the re-attachment of non-serializable objects (like callbacks) that 
    Streamlit drops between execution cycles.

    Note:
        Must be called at the very top of the script's entry point.
    """

    # Component State Persistence
    # Initializing the custom ImageUploader class. We check for existence so we 
    # don't overwrite existing uploads during a standard rerun.
    if "uploader" not in st.session_state:
        st.session_state.uploader = ImageUploader(on_delete=on_image_delete)

    # Callback Re-synchronization
    # Re-attach callback every rerun — Streamlit does not serialise callables
    st.session_state.uploader.on_delete = on_image_delete

    if "process_requested" not in st.session_state:
        st.session_state.process_requested = False

    # Stores the list of chat dictionaries {role, content}
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    # Unique identifier for the conversation thread (useful for LangChain/Agent logging)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    # Multi-Slot Architecture Initialization
    # Using a loop to standardize the setup for dual-agent comparison (Slots 1 and 2).
    for slot in ["1", "2"]:
        # Monitors if the user changed the model in the dropdown    
        if f"model_tracker_{slot}" not in st.session_state:
            st.session_state[f"model_tracker_{slot}"] = "Select Model..."

        # The 'Vault' key for API credentials
        if f"confirmed_api_key_{slot}" not in st.session_state:
            st.session_state[f"confirmed_api_key_{slot}"] = ""

        # Tracks the previous provider to detect provider-level changes
        if f"last_provider_{slot}" not in st.session_state:
            st.session_state[f"last_provider_{slot}"] = None

    # Global Vision Logic
    # Used to detect if the set of uploaded images has changed since the last LLM call
    if "last_image_hash" not in st.session_state:
        st.session_state.last_image_hash = None

    # Holds the temporary stream output from the LLM before it's committed to history
    if "current_full_message" not in st.session_state:
        st.session_state.current_full_message = ""

    # Stores the processed base64 strings or file objects currently staged for the LLM
    if "current_images_to_send" not in st.session_state:
        st.session_state.current_images_to_send = None

    if "extraction_stage" not in st.session_state:
        st.session_state.extraction_stage = None

    if "image_analysis_stage" not in st.session_state:
        st.session_state.image_analysis_stage = None


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🤖 Invoice Image Processing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Setup Constants
PROVIDERS = ["OpenAI", "Google", "Groq", "HuggingFace"]

SYSTEM_PROMPT_1 = """You are an Expert Receipt Parser.

Instructions:
    1. ANALYZE the image and extract the raw text.
    2. ANALYZE the raw text to find the items and amounts.
    3. Ignore the total unless it is the only information available in which case use it and give it an intuitive category name.
    4. If the text is messy, use your intelligence to fix typos (e.g., 'T0TAL' -> 'Total').
    5. Your ONLY goal is to take that raw text, extract the items and amounts, and label them into categories.
    5. Output the final data in a structured JSON format.
   
    Output Example:
    {
        {"category": "category type",
        "amount": "Price"
        }
    }
'"""

# Stylesheet now handles all styling including the diagnosis text area —
# no inline CSS needed in this file.
load_css("static/style.css")
init_session_state()

# 3. Load Model Data (JSON)
models_data = load_models("models.json")

st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <h1 style="margin: 0;">🤖 Invoice Image Processing</h1>
    </div>
""", unsafe_allow_html=True)


# =============================================================================
# UI — INPUT PANEL (diagnosis + images)
# =============================================================================

# Visually distinct input panel above the chat area.
# Styling (border, background) is handled by style.css.
with st.sidebar:
    # Aesthetically centered instruction for the user
    st.markdown(
    "<p style='text-align: center; color: #aaaaaa;'>Upload your invoice images.</p>",
    unsafe_allow_html=True
    )

    # Render the custom uploader component (handles internal state/preview)
    st.session_state.uploader.render()
    #img_data = st.session_state.uploader.get_images()

    if st.button("🚀 Complete Upload", use_container_width=True, key="btn_complete_upload_1", type="primary"):
        st.session_state.process_requested = True
        
        # 0. CHECK IF PROCESSING IS ACTUALLY NEEDED
        if st.session_state.get("process_requested", False):
            
            # 1. DATA GATHERING & STATE ANALYSIS
            image_bytes = st.session_state.uploader.get_images()
            image_hash = st.session_state.uploader.get_hash()
            image_types = st.session_state.uploader.get_types()

            has_images = len(image_bytes) > 0
            images_changed = image_hash != st.session_state.last_image_hash

            # 2. CONTEXT PREPARATION
            st.session_state.current_full_message = build_context_prefix(has_images)
            st.session_state.messages.append({"role": "user", "content": st.session_state.current_full_message})

            # 3. INTELLIGENT IMAGE SENDING LOGIC
            if has_images and images_changed:
                st.session_state.current_images_to_send = image_bytes
                
                # --- MIME-TYPE NORMALIZATION LOGIC ---
                # Handle cases where image_types might be a single string or a list
                if isinstance(image_types, str):
                    # If it's just one string, multiply it by the number of images
                    st.session_state.current_mime_types = [image_types] * len(image_bytes)
                else:
                    # If it's a list, check if the length matches the number of images
                    if len(image_types) == len(image_bytes):
                        st.session_state.current_mime_types = image_types
                    else:
                        # Fallback: Use the first type for all images if there's a mismatch
                        st.session_state.current_mime_types = [image_types[0]] * len(image_bytes)

                st.session_state.last_image_hash = image_hash
            else:
                st.session_state.current_images_to_send = None
                st.session_state.current_mime_types = []

            # IMPORTANT: Reset the flag so the next rerun doesn't enter this block again
            st.session_state.process_requested = False 

            st.session_state.extraction_stage = True
            
            # Now trigger the rerun to update the UI with the new messages/state
            st.rerun()

    st.header("Image Analysis AI Model")

    # 1. MODEL #1 MODEL & CREDENTIAL INITIALIZATION
    prov_1, mod_1 = model_selector("1", models_data)

    if mod_1 and mod_1 != "Select Model...":
        key_1 = api_key_handler("1", prov_1)

        # Lazy initialization of the Agent: only builds if a key exists and agent is missing
        if key_1 and "agent_1" not in st.session_state:
            st.session_state.agent_1 = Agent(
                llm_provider=prov_1,
                model_name=mod_1,
                temperature=0.0,
                api_key=key_1,
                system_prompt=SYSTEM_PROMPT_1,
            )
            st.session_state.agent_1.node_func = st.session_state.agent_1.vision_node

    st.header("Expense Analysis AI Model")

    # 2. MODEL #2 MODEL & CREDENTIAL INITIALIZATION
    prov_2, mod_2 = model_selector("2", models_data)

    if mod_2 and mod_2 != "Select Model...":
        key_2 = api_key_handler("2", prov_2)

        # Lazy initialization of the Agent: only builds if a key exists and agent is missing
        if key_2 and "agent_2" not in st.session_state:
            st.session_state.agent_2 = Agent(
                llm_provider=prov_2,
                model_name=mod_2,
                temperature=0.0,
                api_key=key_2,
                system_prompt=SYSTEM_PROMPT_1,
            )
            st.session_state.agent_2.node_func = st.session_state.agent_2.analyst_node

    # 3. Initialize the Flow (The Orchestrator)
    # This only builds if we have all 3 agents ready
    if all(k in st.session_state for k in ["agent_1", "agent_2"]) and "flow_1" not in st.session_state:
        st.session_state.flow_1 = Flow(
            agents=[st.session_state.agent_1, st.session_state.agent_2], 
            memory=True
        )


# =============================================================================
# UI — CHAT AREA
# =============================================================================

# CHAT INTERFACE CONTAINER
with st.container(height=400, border=True):
    # --- Render existing chat history ---
    #for msg in st.session_state.messages:
        #with st.chat_message(msg["role"]):
            #st.markdown(msg["content"])

    # TRIGGER INFERENCE (Flow Orchestration Logic)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        if st.session_state.get("extraction_stage") or st.session_state.get("image_analysis_stage"):
            if "flow_1" in st.session_state:
                with st.chat_message("assistant"):
                    current_images = st.session_state.get("current_images_to_send")
                    current_types = st.session_state.get("current_mime_types")

                    if isinstance(current_images, list) and isinstance(current_types, str):
                        current_types = [current_types] * len(current_images)

                    # We now pass the FLOW to the streamer
                    full_res = stream_response(
                        flow=st.session_state.flow_1, # Target the flow engine
                        user_message=st.session_state.current_full_message,
                        image_data=current_images,
                        mime_type=current_types,
                        thread_id=st.session_state.session_id
                    )

                if full_res:
                    st.write(full_res)

                    if st.session_state.extraction_stage:
                        name_tag = "extraction_result"
                    else:
                        name_tag = "analysis_result"

                    st.session_state.messages.append({"role": "assistant", "content": full_res, "name": name_tag})

                    st.session_state.current_images_to_send = None
                    st.session_state.current_mime_types = None
                    st.session_state.image_analysis_stage = True
                    
                    # Now rerun safely
                    #st.rerun()

    if st.session_state.image_analysis_stage:
        # 1. THE PAST (Always at the top of the block)
        #with st.container(height=300, border=True):
        for i, msg in enumerate(st.session_state.messages):
            if i < 1: continue # Hide the setup context
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 2. THE PRESENT (The Input Box)
        # Note: st.chat_input naturally "pins" to the bottom of the screen
        if prompt := st.chat_input("Ask about the invoice..."):
            # Add the human message to history immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # We redraw the user's message so they see it while the AI "thinks"
            with st.chat_message("user"):
                st.markdown(prompt)

            # 3. THE FUTURE (The AI Logic)
            with st.chat_message("assistant"):
                # Stream the response live
                response = stream_response(
                    flow=st.session_state.flow_1,
                    user_message=prompt,
                    thread_id=st.session_state.session_id
                )
                # Append the final result to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to move the "Present" message into the "Past" container
            st.rerun()

        # 4. UTILITIES
        st.button("🗑️ Clear AI Model Chat", 
                key="btn_clear_1", 
                use_container_width=True, 
                on_click=clear_chat_callback)


# GLOBAL RERUN HANDLER
# This block is essential for the 'Streamlit Flow'. Because we appended messages 
# to the state AFTER the containers were drawn, we must rerun once to make 
# those assistant messages visible to the user immediately.
if st.session_state.get("needs_rerun"):
    st.session_state.needs_rerun = False
    st.rerun()