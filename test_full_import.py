# test_full_import.py
import sys
import types

# Apply patch
dummy = types.ModuleType('langchain_core.messages.content')
dummy.FileContentBlock = type('FileContentBlock', (), {})
dummy.ImageContentBlock = type('ImageContentBlock', (), {})
dummy.TextContentBlock = type('TextContentBlock', (), {})
dummy.create_file_block = lambda x, **kw: x
dummy.create_image_block = lambda x, **kw: x
dummy.create_text_block = lambda x, **kw: x
sys.modules['langchain_core.messages.content'] = dummy

# Try all your imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    from langchain.tools import tool
    from pydantic import BaseModel, Field
    from typing_extensions import Annotated
    from langchain_google_genai import ChatGoogleGenerativeAI
    from dotenv import load_dotenv
    from langchain_core.messages import (HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage)
    from langgraph.graph.message import add_messages
    import asyncio, random
    from playwright.async_api import async_playwright
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("You can now run your full code.")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()