"""Streamlit frontend for MCP Client.

This module provides a chat interface for interacting with the MCP client,
featuring markdown rendering, streaming responses, and multi-modal support.
"""

import asyncio
import json
import os
import streamlit as st
from pathlib import Path
from typing import Optional
from loguru import logger

# Import business logic - use new architecture
from agents.chat_agent import ChatAgent

# ---------------------------------------------------------------------------
# SVG icon helpers (inline, no emoji)
# ---------------------------------------------------------------------------

ICON_TERMINAL = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>'
ICON_CPU = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>'
ICON_BRAIN = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a7 7 0 0 0-7 7c0 3 2 5.5 4 7l3 3 3-3c2-1.5 4-4 4-7a7 7 0 0 0-7-7z"/><path d="M12 2v10"/><path d="M8 6h8"/></svg>'
ICON_WRENCH = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>'
ICON_KEY = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>'
ICON_SERVER = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>'
ICON_SETTINGS = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
ICON_IMAGE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>'
ICON_REFRESH = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'
ICON_TRASH = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'
ICON_ACTIVITY = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
ICON_ZAP = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
ICON_PLAY = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>'
ICON_FILE = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>'
ICON_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
ICON_X = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
ICON_CIRCLE = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="#22c55e" stroke="none"><circle cx="12" cy="12" r="10"/></svg>'
ICON_CIRCLE_OFF = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="#71717a" stroke="none"><circle cx="12" cy="12" r="10"/></svg>'
ICON_LOADER = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/></svg>'
ICON_BAR_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>'


def _icon(svg: str, label: str = "") -> str:
    """Return an inline HTML snippet with an SVG icon and optional label."""
    gap = f"&nbsp;&nbsp;{label}" if label else ""
    return f'<span style="display:inline-flex;align-items:center;gap:4px;vertical-align:middle;">{svg}{gap}</span>'


# Page configuration
st.set_page_config(
    page_title="MCP Client",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS -- Professional black & white dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* --- Import professional font --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- Root variables --- */
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #141414;
        --bg-elevated: #1a1a1a;
        --bg-surface: #1e1e1e;
        --border-subtle: #262626;
        --border-default: #333333;
        --border-strong: #444444;
        --text-primary: #f0f0f0;
        --text-secondary: #a0a0a0;
        --text-muted: #666666;
        --accent: #ffffff;
        --accent-dim: #888888;
        --success: #22c55e;
        --warning: #eab308;
        --error: #ef4444;
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }

    /* --- Global overrides --- */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* --- Hide Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(10,10,10,0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    [data-testid="stSidebar"] .stDivider {
        border-color: var(--border-subtle) !important;
    }

    /* --- Section headers in sidebar --- */
    .sidebar-section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        padding: 8px 0 4px 0;
        margin-top: 4px;
        border: none;
    }
    .sidebar-section-header svg {
        opacity: 0.6;
    }

    /* --- Chat messages --- */
    [data-testid="stChatMessage"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* --- Chat input --- */
    [data-testid="stChatInput"] {
        border-color: var(--border-default) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.925rem !important;
        color: var(--text-primary) !important;
    }

    /* --- Buttons --- */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.8125rem !important;
        letter-spacing: 0.01em;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-default) !important;
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        transition: all 0.15s ease !important;
        padding: 0.4rem 1rem !important;
    }
    .stButton > button:hover {
        background: var(--border-default) !important;
        border-color: var(--border-strong) !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #e0e0e0 !important;
        border-color: #e0e0e0 !important;
    }

    /* --- Expanders --- */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-surface) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--text-primary) !important;
    }

    /* --- Inputs --- */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8125rem !important;
        background: var(--bg-primary) !important;
        border-color: var(--border-default) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-dim) !important;
        box-shadow: 0 0 0 1px var(--border-strong) !important;
    }

    /* --- Select boxes --- */
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-primary) !important;
        border-color: var(--border-default) !important;
        border-radius: var(--radius-sm) !important;
        font-size: 0.8125rem !important;
    }

    /* --- Code blocks --- */
    [data-testid="stCode"], .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* --- Tool call cards --- */
    .tool-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 0.875rem;
        margin: 0.375rem 0;
    }
    .tool-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8125rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.625rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .tool-card-section {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 0.625rem;
        margin: 0.25rem 0;
    }
    .tool-card-label {
        font-size: 0.6875rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.375rem;
    }

    /* --- Reasoning / Thinking --- */
    .reasoning-block {
        background: var(--bg-primary);
        border-left: 2px solid var(--border-strong);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 0.75rem 1rem;
        margin: 0.375rem 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-style: italic;
    }

    /* --- Status badges --- */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 100px;
        letter-spacing: 0.02em;
    }
    .status-ready {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .status-offline {
        background: rgba(113, 113, 122, 0.1);
        color: #71717a;
        border: 1px solid rgba(113, 113, 122, 0.2);
    }

    /* --- Streaming indicator --- */
    .streaming-pulse {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--text-muted);
        font-size: 0.8125rem;
        font-style: italic;
    }
    .streaming-pulse::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--text-muted);
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    /* --- Main title area --- */
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 4px;
        margin-bottom: 2px;
    }
    .app-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        line-height: 1;
    }
    .app-header-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* --- Info/warning/success/error overrides --- */
    [data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        font-size: 0.85rem !important;
    }

    /* --- File uploader --- */
    [data-testid="stFileUploader"] {
        border-radius: var(--radius-md) !important;
    }
    [data-testid="stFileUploader"] section {
        border-color: var(--border-default) !important;
        border-radius: var(--radius-md) !important;
    }

    /* --- Dividers --- */
    hr {
        border-color: var(--border-subtle) !important;
    }

    /* --- Scrollbar --- */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-default);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-strong);
    }

    /* --- Token usage table --- */
    .usage-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .usage-table td {
        padding: 4px 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .usage-table td:first-child {
        color: var(--text-muted);
        font-weight: 500;
    }
    .usage-table td:last-child {
        text-align: right;
        color: var(--text-primary);
    }
    .usage-total td {
        border-top: 1px solid var(--border-default);
        font-weight: 600;
        padding-top: 6px;
    }

    /* --- Image preview --- */
    .image-preview {
        max-width: 180px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-subtle);
        margin: 4px;
    }

    /* --- Sidebar brand --- */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 12px 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 12px;
    }
    .sidebar-brand-text {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text-primary);
    }
    .sidebar-brand-version {
        font-size: 0.65rem;
        color: var(--text-muted);
        background: var(--bg-surface);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
    }

    /* Hide default streamlit title styling */
    [data-testid="stSidebar"] h1 {
        display: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


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
        with st.spinner("Initializing agent..."):
            agent = ChatAgent(model_name=model_name, mcp_config_path=mcp_config)
            st.session_state.chat_agent = agent
            logger.info("Chat agent initialized successfully")

            # Show validation warnings if any
            warnings = agent.get_validation_warnings()
            if warnings:
                for warning in warnings:
                    st.warning(f"MCP Tool Issue: {warning}")

            return True
    except Exception as e:
        st.error(f"Failed to initialize chat agent: {e}")
        logger.error(f"Agent initialization error: {e}")
        return False


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        # Brand header
        st.markdown(
            f"""<div class="sidebar-brand">
                {ICON_TERMINAL}
                <span class="sidebar-brand-text">MCP Client</span>
                <span class="sidebar-brand-version">v0.1.0</span>
            </div>""",
            unsafe_allow_html=True,
        )

        # --- MCP Config Section ---
        st.markdown(
            f'<div class="sidebar-section-header">{ICON_FILE} Configuration</div>',
            unsafe_allow_html=True,
        )

        with st.expander("MCP Config Editor", expanded=False):
            config_path = st.text_input(
                "Config File Path",
                value=st.session_state.mcp_config_path,
                key="config_path_input",
                help="Path to mcp.json configuration file",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", key="load_config_btn"):
                    try:
                        config_file = Path(config_path)
                        if config_file.exists():
                            with open(config_file, "r") as f:
                                st.session_state.mcp_config_content = f.read()
                            st.session_state.mcp_config_path = config_path
                            st.success(f"Loaded {config_path}")
                        else:
                            st.error(f"File not found: {config_path}")
                    except Exception as e:
                        st.error(f"Error loading config: {e}")

            with col2:
                if st.button("Save", key="save_config_btn"):
                    try:
                        if st.session_state.mcp_config_content:
                            json.loads(st.session_state.mcp_config_content)
                            with open(config_path, "w") as f:
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
                try:
                    config_file = Path(config_path)
                    if config_file.exists():
                        with open(config_file, "r") as f:
                            st.session_state.mcp_config_content = f.read()
                except Exception:
                    st.session_state.mcp_config_content = '{\n  "servers": {}\n}'

            edited_config = st.text_area(
                "MCP Configuration (JSON)",
                value=st.session_state.mcp_config_content,
                height=300,
                help="Edit your MCP server configuration in JSON format",
                key="config_editor",
            )

            if edited_config != st.session_state.mcp_config_content:
                st.session_state.mcp_config_content = edited_config

            if st.session_state.mcp_config_content:
                try:
                    json.loads(st.session_state.mcp_config_content)
                    st.markdown(
                        f"{_icon(ICON_CHECK, 'Valid JSON')}",
                        unsafe_allow_html=True,
                    )
                except json.JSONDecodeError as e:
                    st.markdown(
                        f"{_icon(ICON_X, f'Invalid JSON: {e}')}",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # --- LLM Configuration Section ---
        st.markdown(
            f'<div class="sidebar-section-header">{ICON_CPU} LLM Configuration</div>',
            unsafe_allow_html=True,
        )

        try:
            from model_providers import (
                get_available_llm_providers,
                get_available_llm_models,
                get_available_embedding_providers,
                get_available_embedding_models,
            )

            # LLM Provider selection
            llm_providers = get_available_llm_providers()
            current_llm_provider = st.session_state.get("llm_provider", "ollama")

            llm_provider = st.selectbox(
                "LLM Provider",
                options=llm_providers,
                index=llm_providers.index(current_llm_provider)
                if current_llm_provider in llm_providers
                else 0,
                key="llm_provider_select",
                help="Select the LLM provider",
            )

            if llm_provider != st.session_state.get("llm_provider"):
                st.session_state.llm_provider = llm_provider
                st.rerun()

            # LLM Model selection
            llm_models = get_available_llm_models(llm_provider)
            current_llm_model = st.session_state.get(
                "llm_model", llm_models[0] if llm_models else ""
            )

            llm_model = st.selectbox(
                "LLM Model",
                options=llm_models,
                index=llm_models.index(current_llm_model)
                if current_llm_model in llm_models
                else 0,
                key="llm_model_select",
                help="Select the LLM model to use",
            )
            st.session_state.llm_model = llm_model

            st.divider()

            # --- Embedding Configuration Section ---
            st.markdown(
                f'<div class="sidebar-section-header">{ICON_BAR_CHART} Embedding Configuration</div>',
                unsafe_allow_html=True,
            )

            embed_providers = get_available_embedding_providers()
            current_embed_provider = st.session_state.get("embed_provider", "ollama")

            embed_provider = st.selectbox(
                "Embedding Provider",
                options=embed_providers,
                index=embed_providers.index(current_embed_provider)
                if current_embed_provider in embed_providers
                else 0,
                key="embed_provider_select",
                help="Select the embedding provider",
            )

            if embed_provider != st.session_state.get("embed_provider"):
                st.session_state.embed_provider = embed_provider
                st.rerun()

            embed_models = get_available_embedding_models(embed_provider)
            current_embed_model = st.session_state.get(
                "embed_model", embed_models[0] if embed_models else ""
            )

            embed_model = st.selectbox(
                "Embedding Model",
                options=embed_models,
                index=embed_models.index(current_embed_model)
                if current_embed_model in embed_models
                else 0,
                key="embed_model_select",
                help="Select the embedding model to use",
            )
            st.session_state.embed_model = embed_model

        except ImportError as e:
            st.warning(f"Provider module not available: {e}")
            model_name = st.text_input(
                "Model Name (Optional)",
                value="",
                help="Override default model from environment",
            )

        st.divider()

        # --- Provider Credentials ---
        st.markdown(
            f'<div class="sidebar-section-header">{ICON_KEY} Provider Credentials</div>',
            unsafe_allow_html=True,
        )
        current_provider = st.session_state.get("llm_provider", "ollama")
        current_embed_provider = st.session_state.get("embed_provider", "ollama")

        needs_gcp = (
            current_provider in ("google", "vertex-gemini", "vertex-claude")
            or current_embed_provider == "google"
        )
        if needs_gcp:
            if current_provider == "google" or current_embed_provider == "google":
                gcp_api_key = st.text_input(
                    "GCP API Key",
                    value=os.getenv("GCP_API_KEY", ""),
                    type="password",
                    key="gcp_api_key_input",
                    help="Google API key for Gemini",
                )
                if gcp_api_key:
                    os.environ["GCP_API_KEY"] = gcp_api_key

            if current_provider in ("vertex-gemini", "vertex-claude"):
                gcp_project = st.text_input(
                    "GCP Project",
                    value=os.getenv("GCP_PROJECT", ""),
                    key="gcp_project_input",
                    help="Google Cloud project ID",
                )
                if gcp_project:
                    os.environ["GCP_PROJECT"] = gcp_project

                gcp_location = st.text_input(
                    "GCP Location",
                    value=os.getenv(
                        "GCP_LOCATION",
                        "us-central1"
                        if current_provider == "vertex-gemini"
                        else "us-east5",
                    ),
                    key="gcp_location_input",
                    help="Google Cloud region",
                )
                if gcp_location:
                    os.environ["GCP_LOCATION"] = gcp_location

        # Streaming toggle
        st.session_state.streaming_enabled = st.checkbox(
            "Enable Streaming",
            value=st.session_state.streaming_enabled,
            help="Show streaming thoughts and tool calls in real-time",
        )

        st.divider()

        # --- Initialize Button ---
        if st.button("Initialize Client", type="primary", use_container_width=True):
            import os

            if st.session_state.get("llm_provider"):
                os.environ["LLM_PROVIDER"] = st.session_state.llm_provider
            if st.session_state.get("llm_model"):
                os.environ["LLM_MODEL"] = st.session_state.llm_model
            if st.session_state.get("embed_provider"):
                os.environ["EMBEDDING_PROVIDER"] = st.session_state.embed_provider
            if st.session_state.get("embed_model"):
                os.environ["EMBEDDING_MODEL"] = st.session_state.embed_model

            model_override = st.session_state.get("llm_model", "")
            if initialize_client(
                st.session_state.mcp_config_path, model_override or None
            ):
                provider_label = st.session_state.get("llm_provider", "default")
                model_label = st.session_state.get("llm_model", "default")
                st.success(f"Initialized: {provider_label} / {model_label}")

        st.divider()

        # --- Status ---
        st.markdown(
            f'<div class="sidebar-section-header">{ICON_ACTIVITY} Status</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.chat_agent:
            st.markdown(
                f'<span class="status-badge status-ready">{ICON_CIRCLE} Agent Ready</span>',
                unsafe_allow_html=True,
            )

            # MCP Servers & Tools
            with st.expander("MCP Servers & Tools", expanded=False):
                if st.button("Refresh Server Info", key="refresh_servers"):
                    with st.spinner("Fetching..."):
                        server_info = get_server_info(st.session_state.chat_agent)
                        st.session_state.server_info = server_info

                if "server_info" not in st.session_state:
                    st.caption("Click refresh to load server details")
                elif st.session_state.server_info:
                    for idx, server in enumerate(st.session_state.server_info, 1):
                        server_label = (
                            f"{server['server_type'].upper()} / {server['server_id']}"
                        )
                        with st.expander(
                            f"Server {idx}: {server_label}", expanded=False
                        ):
                            st.caption(f"Type: `{server['server_type']}`")
                            st.caption(f"ID: `{server['server_id']}`")

                            if server["tools"]:
                                st.markdown(
                                    f"**{len(server['tools'])} tool(s) available:**"
                                )
                                for tool in server["tools"]:
                                    st.markdown(f"- **{tool['name']}**")
                                    st.caption(f"  {tool['description']}")
                            else:
                                st.caption("No tools available")
                else:
                    st.caption("No MCP servers connected")

            # Usage stats
            if (
                hasattr(st.session_state.chat_agent, "usage")
                and st.session_state.chat_agent.usage
            ):
                usage = st.session_state.chat_agent.usage

                # Cost calculation (NOK)
                input_token_price_nok = 23 / 1e6
                cached_input_token_price_nok = 6 / 1e6
                output_token_price_nok = 90 / 1e6

                regular_input_tokens = (usage.input_tokens or 0) - (
                    usage.cache_read_tokens or 0
                )
                cached_input_tokens = usage.cache_read_tokens or 0
                output_tokens = usage.output_tokens or 0

                input_cost = regular_input_tokens * input_token_price_nok
                cached_input_cost = cached_input_tokens * cached_input_token_price_nok
                output_cost = output_tokens * output_token_price_nok
                total_cost = input_cost + cached_input_cost + output_cost

                with st.expander("Token Usage & Cost"):
                    rows = f"<tr><td>Requests</td><td>{usage.requests}</td></tr>"
                    rows += (
                        f"<tr><td>Total Tokens</td><td>{usage.total_tokens:,}</td></tr>"
                    )

                    if usage.input_tokens:
                        rows += f"<tr><td>Input Tokens</td><td>{usage.input_tokens:,}</td></tr>"
                        if usage.cache_read_tokens:
                            rows += f"<tr><td>&nbsp;&nbsp;Regular</td><td>{regular_input_tokens:,} (kr {input_cost:.4f})</td></tr>"
                            rows += f"<tr><td>&nbsp;&nbsp;Cached</td><td>{cached_input_tokens:,} (kr {cached_input_cost:.4f})</td></tr>"
                        else:
                            rows += f"<tr><td>&nbsp;&nbsp;Cost</td><td>kr {input_cost:.4f}</td></tr>"

                    if usage.output_tokens:
                        rows += (
                            f"<tr><td>Output Tokens</td><td>{output_tokens:,}</td></tr>"
                        )
                        rows += f"<tr><td>&nbsp;&nbsp;Cost</td><td>kr {output_cost:.4f}</td></tr>"

                    rows += f'<tr class="usage-total"><td>Total Cost</td><td>kr {total_cost:.4f} NOK</td></tr>'

                    st.markdown(
                        f'<table class="usage-table">{rows}</table>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                f'<span class="status-badge status-offline">{ICON_CIRCLE_OFF} Not Initialized</span>',
                unsafe_allow_html=True,
            )

        # --- Image Upload ---
        st.divider()
        st.markdown(
            f'<div class="sidebar-section-header">{ICON_IMAGE} Attachments</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            help="Attach images to your next message",
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.session_state.uploaded_images = []
            cols = st.columns(2)
            for idx, file in enumerate(uploaded_files):
                temp_path = Path(f"temp_{file.name}")
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                st.session_state.uploaded_images.append(str(temp_path))

                with cols[idx % 2]:
                    st.image(file, caption=file.name, width=140)

        if st.session_state.uploaded_images:
            if st.button("Clear Images", key="clear_images_btn"):
                st.session_state.uploaded_images = []
                st.rerun()

        # --- Actions ---
        st.divider()
        st.markdown(
            f'<div class="sidebar-section-header">{ICON_SETTINGS} Actions</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True, key="clear_chat_btn"):
                st.session_state.messages = []
                if st.session_state.chat_agent:
                    st.session_state.chat_agent.message_history = []
                st.rerun()

        with col2:
            if st.button(
                "Reset Agent", use_container_width=True, key="reset_agent_btn"
            ):
                st.session_state.chat_agent = None
                st.session_state.messages = []
                st.rerun()


def render_message(
    role: str,
    content: str,
    tool_calls: list = None,
    thoughts: list = None,
    thinking: str = None,
    msg_idx: int = 0,
):
    """Render a chat message with markdown support, tool calls, thoughts, and reasoning."""
    with st.chat_message(role):
        # Display thinking/reasoning if available
        if thinking:
            with st.expander("Reasoning", expanded=False):
                st.markdown(
                    f'<div class="reasoning-block">{thinking}</div>',
                    unsafe_allow_html=True,
                )

        # Display tool calls if any
        if tool_calls and len(tool_calls) > 0:
            with st.expander(f"Tool Calls ({len(tool_calls)})", expanded=False):
                for i, call in enumerate(tool_calls, 1):
                    st.markdown(
                        f"""<div class="tool-card">
                            <div class="tool-card-header">
                                {ICON_WRENCH}
                                <span>{call["name"]}</span>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            '<div class="tool-card-label">INPUT</div>',
                            unsafe_allow_html=True,
                        )
                        st.json(call.get("args", {}))

                    with col2:
                        st.markdown(
                            '<div class="tool-card-label">OUTPUT</div>',
                            unsafe_allow_html=True,
                        )
                        result_text = str(call.get("result", "No result"))
                        if len(result_text) > 500:
                            st.text_area(
                                "",
                                result_text,
                                height=150,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"tool_result_msg_{msg_idx}_call_{i}",
                            )
                        else:
                            st.code(result_text, language="text")

                    if i < len(tool_calls):
                        st.divider()

        # Render markdown content
        st.markdown(content, unsafe_allow_html=True)


async def process_query_async_streaming(
    agent: ChatAgent,
    query: str,
    image_paths: Optional[list[str]] = None,
    container=None,
):
    """Asynchronously process a query with streaming support."""
    tool_calls = []
    thoughts = []
    full_text = ""
    all_thinking = []
    current_thinking = ""
    final_result = None

    thinking_sections = []
    tool_call_sections = []
    text_placeholder = None

    try:
        if not query:
            return {
                "output": "Error: No query provided.",
                "tool_calls": [],
                "thoughts": [],
                "thinking": "",
            }

        thinking_container = None
        tools_container = None

        if container and st.session_state.streaming_enabled:
            thinking_container = container.container()
            tools_container = container.container()
            text_placeholder = container.empty()

        stream_gen = agent.run_query_stream(query, image_paths)

        try:
            async for chunk in stream_gen:
                chunk_type = chunk.get("type")

                if chunk_type == "text_delta":
                    text_chunk = chunk.get("content", "")
                    full_text += text_chunk

                    if text_placeholder:
                        text_placeholder.markdown(full_text)

                elif chunk_type == "thinking_delta":
                    thinking_chunk = chunk.get("content", "")
                    current_thinking += thinking_chunk

                    if thinking_container:
                        if not thinking_sections or len(current_thinking) < 10:
                            label = (
                                f"Reasoning {len(thinking_sections) + 1}"
                                if thinking_sections
                                else "Reasoning"
                            )
                            thinking_expander = thinking_container.expander(
                                label, expanded=True
                            )
                            thinking_placeholder = thinking_expander.empty()
                            thinking_sections.append(thinking_placeholder)

                        thinking_sections[-1].markdown(
                            f'<div class="reasoning-block">{current_thinking}</div>',
                            unsafe_allow_html=True,
                        )

                elif chunk_type == "tool_call":
                    if current_thinking:
                        all_thinking.append(current_thinking)
                        current_thinking = ""

                    tool_call = {
                        "name": chunk.get("tool_name", "Unknown"),
                        "args": chunk.get("tool_args", {}),
                        "result": None,
                    }
                    tool_calls.append(tool_call)

                    if tools_container:
                        tool_expander = tools_container.expander(
                            f"Tool Call {len(tool_calls)}: {tool_call['name']}",
                            expanded=False,
                        )
                        tool_placeholder = tool_expander.empty()
                        tool_call_sections.append(tool_placeholder)

                        with tool_placeholder.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(
                                    '<div class="tool-card-label">INPUT</div>',
                                    unsafe_allow_html=True,
                                )
                                st.json(tool_call["args"])
                            with col2:
                                st.markdown(
                                    '<div class="tool-card-label">OUTPUT</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<div class="streaming-pulse">Running...</div>',
                                    unsafe_allow_html=True,
                                )

                elif chunk_type == "tool_result":
                    tool_name = chunk.get("tool_name")
                    result_content = chunk.get("content", "N/A")

                    for i, tool in enumerate(reversed(tool_calls)):
                        if tool["result"] is None:
                            tool["result"] = result_content

                            if i < len(tool_call_sections):
                                idx = len(tool_calls) - 1 - i
                                with tool_call_sections[idx].container():
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(
                                            '<div class="tool-card-label">INPUT</div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.json(tool["args"])
                                    with col2:
                                        st.markdown(
                                            '<div class="tool-card-label">OUTPUT</div>',
                                            unsafe_allow_html=True,
                                        )
                                        result_str = str(result_content)
                                        if len(result_str) > 500:
                                            st.text_area(
                                                "",
                                                result_str,
                                                height=150,
                                                disabled=True,
                                                label_visibility="collapsed",
                                                key=f"stream_tool_result_{idx}",
                                            )
                                        else:
                                            st.code(result_str, language="text")
                            break

                elif chunk_type == "final":
                    if current_thinking:
                        all_thinking.append(current_thinking)

                    final_content = chunk.get("content", full_text)
                    final_thinking = chunk.get("thinking", " ".join(all_thinking))
                    usage_info = chunk.get("usage", {})
                    final_result = {
                        "output": final_content or full_text,
                        "tool_calls": tool_calls,
                        "thoughts": thoughts,
                        "thinking": final_thinking or " ".join(all_thinking),
                        "usage": usage_info,
                    }

                elif chunk_type == "error":
                    error_msg = chunk.get("content", "Unknown error")
                    final_result = {
                        "output": f"Error: {error_msg}",
                        "tool_calls": tool_calls,
                        "thoughts": thoughts,
                        "thinking": " ".join(all_thinking),
                    }
        finally:
            await stream_gen.aclose()

        if container and st.session_state.streaming_enabled and final_result:
            container.empty()

        if final_result:
            return final_result

        if current_thinking:
            all_thinking.append(current_thinking)

        return {
            "output": full_text or "No response received",
            "tool_calls": tool_calls,
            "thoughts": thoughts,
            "thinking": " ".join(all_thinking),
        }

    except Exception as e:
        return {
            "output": f"Stream Error: {str(e)}",
            "tool_calls": [],
            "thoughts": [],
            "thinking": "",
        }


async def process_query_async(
    agent: ChatAgent, query: str, image_paths: Optional[list[str]] = None
):
    """Asynchronously process a query using the chat agent (non-streaming)."""
    try:
        result = await agent.run_query(query, image_paths)

        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = result.data if hasattr(result, "data") else str(result)

        return {"output": output, "tool_calls": [], "thoughts": []}
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return {"output": f"Error: {str(e)}", "tool_calls": [], "thoughts": []}


def _reset_mcp_toolsets(agent: ChatAgent):
    """Reset MCP toolset internal async state for the current event loop.

    MCPServer objects hold an asyncio.Lock (_enter_lock) that is bound to the
    event loop on which it was created.  Since process_query() creates a fresh
    event loop for every request, the old lock becomes invalid.  Calling
    __post_init__() re-creates the lock (and resets _running_count /
    _exit_stack) so the toolsets work correctly on the new loop.
    """
    for toolset in getattr(agent, "toolsets", []):
        # Unwrap ToolFilterWrapper to reach the underlying MCPServer
        inner = getattr(toolset, "wrapped_toolset", toolset)
        if hasattr(inner, "__post_init__"):
            inner.__post_init__()


def process_query(
    agent: ChatAgent,
    query: str,
    image_paths: Optional[list[str]] = None,
    container=None,
):
    """Synchronously process a query (wrapper for async function)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Reset MCP toolset async state (locks, streams) so they work on this
        # new event loop.  Without this, the asyncio.Lock inside each
        # MCPServer is still bound to the previous (now-closed) loop, causing
        # "Session terminated" / McpError on the first request.
        _reset_mcp_toolsets(agent)

        if st.session_state.streaming_enabled:
            result = loop.run_until_complete(
                process_query_async_streaming(agent, query, image_paths, container)
            )
        else:
            result = loop.run_until_complete(
                process_query_async(agent, query, image_paths)
            )
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
        _reset_mcp_toolsets(agent)
        result = loop.run_until_complete(get_server_info_async(agent))
        return result
    finally:
        loop.close()


def main():
    """Main application entry point."""
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main chat interface -- header
    st.markdown(
        f"""<div class="app-header">
            {ICON_TERMINAL}
            <div>
                <div class="app-header-title">MCP Client</div>
                <div class="app-header-subtitle">Multi-modal AI assistant with Model Context Protocol integration</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Check if agent is initialized
    if not st.session_state.chat_agent:
        st.markdown("---")
        st.markdown(
            "Initialize the chat agent from the sidebar to begin.",
        )
        return

    # Display chat history
    for msg_idx, message in enumerate(st.session_state.messages):
        tool_calls = message.get("tool_calls", [])
        thoughts = message.get("thoughts", [])
        thinking = message.get("thinking", "")
        render_message(
            message["role"], message["content"], tool_calls, thoughts, thinking, msg_idx
        )

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        render_message("user", prompt)

        # Display attached images if any
        if st.session_state.uploaded_images:
            with st.chat_message("user"):
                st.caption(f"{len(st.session_state.uploaded_images)} image(s) attached")

        # Process query with assistant
        with st.chat_message("assistant"):
            streaming_container = st.container()

            with st.spinner("Processing..."):
                result = process_query(
                    st.session_state.chat_agent,
                    prompt,
                    st.session_state.uploaded_images
                    if st.session_state.uploaded_images
                    else None,
                    container=streaming_container,
                )

            # Clear streaming container
            streaming_container.empty()

            # Display thinking if available
            if result.get("thinking"):
                with st.expander("Reasoning", expanded=False):
                    st.markdown(
                        f'<div class="reasoning-block">{result["thinking"]}</div>',
                        unsafe_allow_html=True,
                    )

            # Display tool calls if any
            if result.get("tool_calls"):
                with st.expander(
                    f"Tool Calls ({len(result['tool_calls'])})",
                    expanded=False,
                ):
                    for i, call in enumerate(result["tool_calls"], 1):
                        st.markdown(
                            f"""<div class="tool-card">
                                <div class="tool-card-header">
                                    {ICON_WRENCH}
                                    <span>{call["name"]}</span>
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        with st.container():
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(
                                    '<div class="tool-card-label">INPUT</div>',
                                    unsafe_allow_html=True,
                                )
                                st.json(call["args"])
                            with col2:
                                st.markdown(
                                    '<div class="tool-card-label">OUTPUT</div>',
                                    unsafe_allow_html=True,
                                )
                                result_str = str(call["result"])
                                if len(result_str) > 500:
                                    st.text_area(
                                        "",
                                        result_str,
                                        height=150,
                                        disabled=True,
                                        label_visibility="collapsed",
                                        key=f"final_tool_result_{i}",
                                    )
                                else:
                                    st.code(result_str, language="text")
                        if i < len(result["tool_calls"]):
                            st.divider()

            # Render final response
            response_text = result.get("output", str(result))
            st.markdown(response_text, unsafe_allow_html=True)

        # Add assistant response to history
        response_text = result.get("output", str(result))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_calls": result.get("tool_calls", []),
                "thoughts": result.get("thoughts", []),
                "thinking": result.get("thinking", ""),
            }
        )

        # Clear uploaded images after sending
        if st.session_state.uploaded_images:
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
