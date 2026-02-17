"""Streamlit frontend for MCP Client.

This module provides a chat interface for interacting with the MCP client,
featuring markdown rendering, streaming responses, and multi-modal support.
"""
import asyncio
import json
import streamlit as st
from pathlib import Path
from typing import Optional
from loguru import logger

# Import business logic - use new architecture
from agents.chat_agent import ChatAgent

# Page configuration
st.set_page_config(
    page_title="MCP Chat Client",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional chat appearance
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e8f4f8;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .image-preview {
        max-width: 200px;
        border-radius: 8px;
        margin: 5px;
    }
    .tool-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .tool-header {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .tool-section {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .tool-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    .streaming-indicator {
        color: #6c757d;
        font-style: italic;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = None
    
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []
    
    if "mcp_config_content" not in st.session_state:
        st.session_state.mcp_config_content = None
    
    if "mcp_config_path" not in st.session_state:
        st.session_state.mcp_config_path = "mcp.json"
    
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    
    # Initialize provider/model selections from environment
    import os
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = os.getenv("LLM_MODEL", "gpt-oss:latest")
    
    if "embed_provider" not in st.session_state:
        st.session_state.embed_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def initialize_client(mcp_config: str, model_name: Optional[str] = None):
    """Initialize the chat agent."""
    try:
        with st.spinner("Initializing chat agent..."):
            agent = ChatAgent(model_name=model_name, mcp_config_path=mcp_config)
            st.session_state.chat_agent = agent
            logger.info("Chat agent initialized successfully")
            
            # Show validation warnings if any
            warnings = agent.get_validation_warnings()
            if warnings:
                for warning in warnings:
                    st.warning(f"🔧 MCP Tool Issue:\n\n{warning}")
            
            return True
    except Exception as e:
        st.error(f"Failed to initialize chat agent: {e}")
        logger.error(f"Agent initialization error: {e}")
        return False


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.title("Configuration")
        
        # MCP Config editor
        with st.expander("MCP Config Editor", expanded=False):
            # Config path input
            config_path = st.text_input(
                "Config File Path",
                value=st.session_state.mcp_config_path,
                key="config_path_input",
                help="Path to mcp.json configuration file"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Config"):
                    try:
                        config_file = Path(config_path)
                        if config_file.exists():
                            with open(config_file, 'r') as f:
                                st.session_state.mcp_config_content = f.read()
                            st.session_state.mcp_config_path = config_path
                            st.success(f"Loaded {config_path}")
                        else:
                            st.error(f"File not found: {config_path}")
                    except Exception as e:
                        st.error(f"Error loading config: {e}")
            
            with col2:
                if st.button("Save Config"):
                    try:
                        if st.session_state.mcp_config_content:
                            # Validate JSON before saving
                            json.loads(st.session_state.mcp_config_content)
                            with open(config_path, 'w') as f:
                                f.write(st.session_state.mcp_config_content)
                            st.success(f"Saved to {config_path}")
                        else:
                            st.warning("No config content to save")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"Error saving config: {e}")
            
            # JSON editor
            if st.session_state.mcp_config_content is None:
                # Try to load default config on first render
                try:
                    config_file = Path(config_path)
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            st.session_state.mcp_config_content = f.read()
                except:
                    st.session_state.mcp_config_content = '{\n  "servers": {}\n}'
            
            edited_config = st.text_area(
                "MCP Configuration (JSON)",
                value=st.session_state.mcp_config_content,
                height=300,
                help="Edit your MCP server configuration in JSON format",
                key="config_editor"
            )
            
            # Update session state when user edits
            if edited_config != st.session_state.mcp_config_content:
                st.session_state.mcp_config_content = edited_config
            
            # Validate JSON if content exists
            if st.session_state.mcp_config_content:
                try:
                    json.loads(st.session_state.mcp_config_content)
                    st.success("✓ Valid JSON")
                except json.JSONDecodeError as e:
                    st.error(f"✗ Invalid JSON: {e}")
        
        st.divider()
        
        # Provider and Model selection
        st.subheader("🤖 LLM Configuration")
        
        # Import provider functions
        try:
            from model_providers import (
                get_available_llm_providers,
                get_available_llm_models,
                get_available_embedding_providers,
                get_available_embedding_models
            )
            
            # LLM Provider selection
            llm_providers = get_available_llm_providers()
            current_llm_provider = st.session_state.get("llm_provider", "ollama")
            
            llm_provider = st.selectbox(
                "LLM Provider",
                options=llm_providers,
                index=llm_providers.index(current_llm_provider) if current_llm_provider in llm_providers else 0,
                key="llm_provider_select",
                help="Select the LLM provider (ollama, azure, grok, groq)"
            )
            
            # Update session state
            if llm_provider != st.session_state.get("llm_provider"):
                st.session_state.llm_provider = llm_provider
                # Refresh available models
                st.rerun()
            
            # LLM Model selection
            llm_models = get_available_llm_models(llm_provider)
            current_llm_model = st.session_state.get("llm_model", llm_models[0] if llm_models else "")
            
            llm_model = st.selectbox(
                "LLM Model",
                options=llm_models,
                index=llm_models.index(current_llm_model) if current_llm_model in llm_models else 0,
                key="llm_model_select",
                help="Select the LLM model to use"
            )
            st.session_state.llm_model = llm_model
            
            st.divider()
            st.subheader("📊 Embedding Configuration")
            
            # Embedding Provider selection
            embed_providers = get_available_embedding_providers()
            current_embed_provider = st.session_state.get("embed_provider", "ollama")
            
            embed_provider = st.selectbox(
                "Embedding Provider",
                options=embed_providers,
                index=embed_providers.index(current_embed_provider) if current_embed_provider in embed_providers else 0,
                key="embed_provider_select",
                help="Select the embedding provider"
            )
            
            # Update session state
            if embed_provider != st.session_state.get("embed_provider"):
                st.session_state.embed_provider = embed_provider
                # Refresh available models
                st.rerun()
            
            # Embedding Model selection
            embed_models = get_available_embedding_models(embed_provider)
            current_embed_model = st.session_state.get("embed_model", embed_models[0] if embed_models else "")
            
            embed_model = st.selectbox(
                "Embedding Model",
                options=embed_models,
                index=embed_models.index(current_embed_model) if current_embed_model in embed_models else 0,
                key="embed_model_select",
                help="Select the embedding model to use"
            )
            st.session_state.embed_model = embed_model
            
        except ImportError as e:
            st.warning(f"Provider module not available: {e}")
            # Fallback to text input
            model_name = st.text_input(
                "Model Name (Optional)",
                value="",
                help="Override default model from environment"
            )
        
        st.divider()
        
        # Streaming toggle
        st.session_state.streaming_enabled = st.checkbox(
            "Enable Streaming",
            value=st.session_state.streaming_enabled,
            help="Show streaming thoughts and tool calls in real-time"
        )
        
        # Initialize button
        if st.button("Initialize Client", type="primary"):
            # Set environment variables for selected providers and models
            import os
            if st.session_state.get("llm_provider"):
                os.environ["LLM_PROVIDER"] = st.session_state.llm_provider
            if st.session_state.get("llm_model"):
                os.environ["LLM_MODEL"] = st.session_state.llm_model
            if st.session_state.get("embed_provider"):
                os.environ["EMBEDDING_PROVIDER"] = st.session_state.embed_provider
            if st.session_state.get("embed_model"):
                os.environ["EMBEDDING_MODEL"] = st.session_state.embed_model
            
            # Initialize with selected model
            model_override = st.session_state.get("llm_model", "")
            if initialize_client(st.session_state.mcp_config_path, model_override or None):
                st.success(f"✓ Client initialized with {st.session_state.get('llm_provider', 'default')} / {st.session_state.get('llm_model', 'default')}!")
        
        # Agent status
        st.divider()
        st.subheader("Status")
        if st.session_state.chat_agent:
            st.success("🟢 Agent Ready")
            
            # Show MCP servers and their tools
            with st.expander("🔧 MCP Servers & Tools", expanded=False):
                if st.button("🔄 Refresh Server Info"):
                    with st.spinner("Fetching server information..."):
                        server_info = get_server_info(st.session_state.chat_agent)
                        st.session_state.server_info = server_info
                
                if 'server_info' not in st.session_state:
                    st.info("Click 'Refresh Server Info' to load server details")
                elif st.session_state.server_info:
                    for idx, server in enumerate(st.session_state.server_info, 1):
                        # Create a collapsible section for each server
                        server_label = f"{server['server_type'].upper()} - {server['server_id']}"
                        with st.expander(f"**Server {idx}:** {server_label}", expanded=False):
                            st.caption(f"Type: `{server['server_type']}`")
                            st.caption(f"ID: `{server['server_id']}`")
                            
                            if server['tools']:
                                st.markdown(f"**{len(server['tools'])} Tool(s) Available:**")
                                for tool in server['tools']:
                                    st.markdown(f"• **{tool['name']}**")
                                    st.caption(f"  {tool['description']}")
                            else:
                                st.warning("No tools available")
                else:
                    st.warning("No MCP servers connected")
            
            # Show usage stats if available
            if hasattr(st.session_state.chat_agent, 'usage') and st.session_state.chat_agent.usage:
                usage = st.session_state.chat_agent.usage
                
                # Calculate cost in NOK (Norwegian Kroner)
                # Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
                input_token_price_nok = 23 / 1e6
                cached_input_token_price_nok = 6 / 1e6
                output_token_price_nok = 90 / 1e6
                
                # Calculate individual costs
                regular_input_tokens = (usage.input_tokens or 0) - (usage.cache_read_tokens or 0)
                cached_input_tokens = usage.cache_read_tokens or 0
                output_tokens = usage.output_tokens or 0
                
                input_cost = regular_input_tokens * input_token_price_nok
                cached_input_cost = cached_input_tokens * cached_input_token_price_nok
                output_cost = output_tokens * output_token_price_nok
                total_cost = input_cost + cached_input_cost + output_cost
                
                with st.expander("Token Usage & Cost"):
                    st.write(f"**Requests:** {usage.requests}")
                    st.write(f"**Total Tokens:** {usage.total_tokens:,}")
                    
                    if usage.input_tokens:
                        st.write(f"**Input Tokens:** {usage.input_tokens:,}")
                        if usage.cache_read_tokens:
                            st.write(f"  - Regular: {regular_input_tokens:,} (kr {input_cost:.4f})")
                            st.write(f"  - Cached: {cached_input_tokens:,} (kr {cached_input_cost:.4f})")
                        else:
                            st.write(f"  Cost: kr {input_cost:.4f}")
                    
                    if usage.output_tokens:
                        st.write(f"**Output Tokens:** {output_tokens:,}")
                        st.write(f"  Cost: kr {output_cost:.4f}")
                    
                    st.divider()
                    st.write(f"**Total Cost:** kr {total_cost:.4f} NOK")
        else:
            st.warning("Agent Not Initialized")
        
        # Image upload section
        st.divider()
        st.subheader("Attach Images")
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            help="Attach images to your next message"
        )
        
        if uploaded_files:
            st.session_state.uploaded_images = []
            cols = st.columns(2)
            for idx, file in enumerate(uploaded_files):
                # Save to temp directory
                temp_path = Path(f"temp_{file.name}")
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                st.session_state.uploaded_images.append(str(temp_path))
                
                # Show preview
                with cols[idx % 2]:
                    st.image(file, caption=file.name, width=150)
        
        # Clear images button
        if st.session_state.uploaded_images:
            if st.button("Clear Images"):
                st.session_state.uploaded_images = []
                st.rerun()
        
        # Clear chat button
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                # Also clear the agent's conversation history
                if st.session_state.chat_agent:
                    st.session_state.chat_agent.message_history = []
                st.rerun()
        
        with col2:
            if st.button("Reset Agent"):
                st.session_state.chat_agent = None
                st.session_state.messages = []
                st.rerun()


def render_message(role: str, content: str, tool_calls: list = None, thoughts: list = None, thinking: str = None, msg_idx: int = 0):
    """Render a chat message with markdown support, tool calls, thoughts, and reasoning."""
    with st.chat_message(role):
        # Display thinking/reasoning if available
        if thinking:
            with st.expander("🧠 Model Reasoning", expanded=False):
                st.markdown(f"*{thinking}*")
        
        # Display tool calls if any
        if tool_calls and len(tool_calls) > 0:
            with st.expander(f"⚙️ Tool Calls ({len(tool_calls)})", expanded=False):
                for i, call in enumerate(tool_calls, 1):
                    st.markdown(f'<div class="tool-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="tool-header">Tool {i}: {call["name"]}</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="tool-section">', unsafe_allow_html=True)
                        st.markdown('<div class="tool-label">Input</div>', unsafe_allow_html=True)
                        st.json(call.get('args', {}))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="tool-section">', unsafe_allow_html=True)
                        st.markdown('<div class="tool-label">Output</div>', unsafe_allow_html=True)
                        result_text = str(call.get('result', 'No result'))
                        if len(result_text) > 500:
                            st.text_area("", result_text, height=150, disabled=True, label_visibility="collapsed", key=f"tool_result_msg_{msg_idx}_call_{i}")
                        else:
                            st.code(result_text, language="text")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if i < len(tool_calls):
                        st.divider()
        
        # Render markdown content
        st.markdown(content, unsafe_allow_html=True)


async def process_query_async_streaming(agent: ChatAgent, query: str, image_paths: Optional[list[str]] = None, container=None):
    """Asynchronously process a query with streaming support."""
    tool_calls = []
    thoughts = []
    full_text = ""
    all_thinking = []  # Track multiple thinking sections
    current_thinking = ""
    final_result = None
    
    # Keep track of placeholders for dynamic updates
    thinking_sections = []
    tool_call_sections = []
    text_placeholder = None
    
    try:
        if not query:
            return {"output": "Error: No query provided.", "tool_calls": [], "thoughts": [], "thinking": ""}
        
        # Create separate containers for proper ordering: thinking -> tools -> text
        thinking_container = None
        tools_container = None
        
        if container and st.session_state.streaming_enabled:
            thinking_container = container.container()
            tools_container = container.container()
            text_placeholder = container.empty()
        
        # Consume the entire async generator to ensure proper cleanup
        stream_gen = agent.run_query_stream(query, image_paths)
        
        try:
            async for chunk in stream_gen:
                chunk_type = chunk.get("type")
                
                if chunk_type == "text_delta":
                    # Accumulate text chunks
                    text_chunk = chunk.get("content", "")
                    full_text += text_chunk
                    
                    # Update the text placeholder with accumulated text
                    if text_placeholder:
                        text_placeholder.markdown(full_text)
                
                elif chunk_type == "thinking_delta":
                    # Accumulate thinking/reasoning chunks
                    thinking_chunk = chunk.get("content", "")
                    current_thinking += thinking_chunk
                    
                    # Create or update thinking expander
                    if thinking_container:
                        # If this is new thinking section, create new expander
                        if not thinking_sections or len(current_thinking) < 10:
                            thinking_expander = thinking_container.expander(
                                f"🧠 Model Reasoning {len(thinking_sections) + 1}" if thinking_sections else "🧠 Model Reasoning",
                                expanded=True
                            )
                            thinking_placeholder = thinking_expander.empty()
                            thinking_sections.append(thinking_placeholder)
                        
                        # Update the current thinking section
                        thinking_sections[-1].markdown(f"*{current_thinking}*")
                
                elif chunk_type == "tool_call":
                    # Save current thinking section if we have one
                    if current_thinking:
                        all_thinking.append(current_thinking)
                        current_thinking = ""
                    
                    # Add tool call
                    tool_call = {
                        'name': chunk.get("tool_name", "Unknown"),
                        'args': chunk.get("tool_args", {}),
                        'result': None
                    }
                    tool_calls.append(tool_call)
                    
                    # Create live tool call display
                    if tools_container:
                        tool_expander = tools_container.expander(
                            f"⚙️ Tool Call {len(tool_calls)}: {tool_call['name']}",
                            expanded=False
                        )
                        tool_placeholder = tool_expander.empty()
                        tool_call_sections.append(tool_placeholder)
                        
                        # Show tool input immediately
                        with tool_placeholder.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Input:**")
                                st.json(tool_call['args'])
                            with col2:
                                st.markdown("**Output:**")
                                st.info("⏳ Running...")
                    
                elif chunk_type == "tool_result":
                    tool_name = chunk.get("tool_name")
                    result_content = chunk.get("content", "N/A")
                    
                    # Find matching tool call and update result
                    for i, tool in enumerate(reversed(tool_calls)):
                        if tool['result'] is None:
                            tool['result'] = result_content
                            
                            # Update the tool display with result
                            if i < len(tool_call_sections):
                                idx = len(tool_calls) - 1 - i
                                with tool_call_sections[idx].container():
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Input:**")
                                        st.json(tool['args'])
                                    with col2:
                                        st.markdown("**Output:**")
                                        result_str = str(result_content)
                                        if len(result_str) > 500:
                                            st.text_area("", result_str, height=150, disabled=True, label_visibility="collapsed", key=f"stream_tool_result_{idx}")
                                        else:
                                            st.code(result_str, language="text")
                            break
                            
                elif chunk_type == "final":
                    # Save last thinking section
                    if current_thinking:
                        all_thinking.append(current_thinking)
                    
                    final_content = chunk.get("content", full_text)
                    final_thinking = chunk.get("thinking", " ".join(all_thinking))
                    usage_info = chunk.get("usage", {})
                    final_result = {
                        'output': final_content or full_text,
                        'tool_calls': tool_calls,
                        'thoughts': thoughts,
                        'thinking': final_thinking or " ".join(all_thinking),
                        'usage': usage_info
                    }
                    # Continue consuming to ensure generator completes
                    
                elif chunk_type == "error":
                    error_msg = chunk.get("content", "Unknown error")
                    final_result = {
                        'output': f"Error: {error_msg}",
                        'tool_calls': tool_calls,
                        'thoughts': thoughts,
                        'thinking': " ".join(all_thinking)
                    }
                    # Continue consuming to ensure generator completes
        finally:
            # Ensure the generator is properly closed
            await stream_gen.aclose()
        
        # Clear streaming UI and show final organized view
        if container and st.session_state.streaming_enabled and final_result:
            container.empty()
        
        # Return final result or what we collected
        if final_result:
            return final_result
        
        # Save last thinking if not saved
        if current_thinking:
            all_thinking.append(current_thinking)
        
        return {
            'output': full_text or "No response received",
            'tool_calls': tool_calls,
            'thoughts': thoughts,
            'thinking': " ".join(all_thinking)
        }
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return {
            "output": f"Error: {str(e)}",
            "tool_calls": [],
            "thoughts": [],
            "thinking": ""
        }
async def process_query_async(agent: ChatAgent, query: str, image_paths: Optional[list[str]] = None):
    """Asynchronously process a query using the chat agent (non-streaming)."""
    try:
        result = await agent.run_query(query, image_paths)
        
        # Extract output text
        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = result.data if hasattr(result, 'data') else str(result)
        
        return {'output': output, 'tool_calls': [], 'thoughts': []}
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return {'output': f"Error: {str(e)}", 'tool_calls': [], 'thoughts': []}


def process_query(agent: ChatAgent, query: str, image_paths: Optional[list[str]] = None, container=None):
    """Synchronously process a query (wrapper for async function)."""
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if st.session_state.streaming_enabled:
            result = loop.run_until_complete(process_query_async_streaming(agent, query, image_paths, container))
        else:
            result = loop.run_until_complete(process_query_async(agent, query, image_paths))
        return result
    finally:
        loop.close()


async def get_server_info_async(agent: ChatAgent):
    """Asynchronously get server information."""
    try:
        return await agent.get_server_info()
    except Exception as e:
        logger.error(f"Error getting server info: {e}")
        return []


def get_server_info(agent: ChatAgent):
    """Synchronously get server info (wrapper for async function)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(get_server_info_async(agent))
        return result
    finally:
        loop.close()


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat interface
    st.title("MCP Chat Client")
    st.caption("Multi-modal AI assistant with Model Context Protocol integration")
    
    # Check if agent is initialized
    if not st.session_state.chat_agent:
        st.info("Please initialize the chat agent from the sidebar to start chatting")
        return
    
    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        tool_calls = message.get("tool_calls", [])
        thoughts = message.get("thoughts", [])
        thinking = message.get("thinking", "")
        render_message(message["role"], message["content"], tool_calls, thoughts, thinking, msg_idx)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        render_message("user", prompt)
        
        # Display attached images if any
        if st.session_state.uploaded_images:
            with st.chat_message("user"):
                st.caption(f"{len(st.session_state.uploaded_images)} image(s) attached")
        
        # Process query with assistant
        with st.chat_message("assistant"):
            # Create container for streaming updates
            streaming_container = st.container()
            
            with st.spinner("Processing..."):
                result = process_query(
                    st.session_state.chat_agent,
                    prompt,
                    st.session_state.uploaded_images if st.session_state.uploaded_images else None,
                    container=streaming_container
                )
            
            # Clear streaming container
            streaming_container.empty()
            
            # Display thinking if available
            if result.get('thinking'):
                with st.expander("🧠 Model Reasoning", expanded=False):
                    st.markdown(f"*{result['thinking']}*")
            
            # Display tool calls if any
            if result.get('tool_calls'):
                with st.expander(f"⚙️ Tool Calls ({len(result['tool_calls'])})", expanded=False):
                    for i, call in enumerate(result['tool_calls'], 1):
                        st.markdown(f"**Tool {i}: {call['name']}**")
                        with st.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Input:**")
                                st.json(call['args'])
                            with col2:
                                st.markdown("**Output:**")
                                result_str = str(call['result'])
                                if len(result_str) > 500:
                                    st.text_area("", result_str, height=150, disabled=True, label_visibility="collapsed", key=f"final_tool_result_{i}")
                                else:
                                    st.code(result_str, language="text")
                        if i < len(result['tool_calls']):
                            st.divider()
            
            # Render final response
            response_text = result.get('output', str(result))
            st.markdown(response_text, unsafe_allow_html=True)
        
        # Add assistant response to history
        response_text = result.get('output', str(result))
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "tool_calls": result.get('tool_calls', []),
            "thoughts": result.get('thoughts', []),
            "thinking": result.get('thinking', "")
        })
        
        # Clear uploaded images after sending
        if st.session_state.uploaded_images:
            # Clean up temp files
            for img_path in st.session_state.uploaded_images:
                try:
                    Path(img_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {img_path}: {e}")
            
            st.session_state.uploaded_images = []
        
        # Rerun to update chat
        st.rerun()


if __name__ == "__main__":
    main()
