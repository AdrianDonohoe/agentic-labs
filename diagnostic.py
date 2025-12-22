import sys
print(f"Python: {sys.version}")

# Try to patch first
import types
dummy = types.ModuleType('langchain_core.messages.content')
dummy.FileContentBlock = type('FileContentBlock', (), {})
dummy.ImageContentBlock = type('ImageContentBlock', (), {})
dummy.TextContentBlock = type('TextContentBlock', (), {})
dummy.create_file_block = lambda x, **kw: x
dummy.create_image_block = lambda x, **kw: x
dummy.create_text_block = lambda x, **kw: x
sys.modules['langchain_core.messages.content'] = dummy

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    print("✓ SUCCESS: Imported MultiServerMCPClient")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    
    # Try to see what langchain_core actually has
    import langchain_core
    print(f"\nlangchain-core version: {langchain_core.__version__}")
    
    import langchain_core.messages as m
    print("\nAvailable in langchain_core.messages:")
    for attr in dir(m):
        if 'content' in attr.lower() or 'block' in attr.lower() or 'File' in attr or 'Image' in attr or 'Text' in attr:
            print(f"  - {attr}")
