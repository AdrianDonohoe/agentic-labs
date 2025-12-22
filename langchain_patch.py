"""
Patch for langchain-mcp-adapters compatibility with langchain-core >= 0.3.x
"""
import sys
import types

def apply_langchain_patch():
    """Apply compatibility patch for langchain-mcp-adapters."""
    
    # Check if patch is already applied
    if 'langchain_core.messages.content' in sys.modules:
        return
    
    # Create dummy module
    content_module = types.ModuleType('langchain_core.messages.content')
    
    # Add required attributes
    content_module.FileContentBlock = type('FileContentBlock', (), {})
    content_module.ImageContentBlock = type('ImageContentBlock', (), {})
    content_module.TextContentBlock = type('TextContentBlock', (), {})
    
    # Add required functions
    content_module.create_file_block = lambda content, **kwargs: {"type": "file", "content": content, **kwargs}
    content_module.create_image_block = lambda content, **kwargs: {"type": "image", "content": content, **kwargs}
    content_module.create_text_block = lambda content, **kwargs: {"type": "text", "content": content, **kwargs}
    
    # Install the module
    sys.modules['langchain_core.messages.content'] = content_module
    
    print("✓ Applied langchain-mcp-adapters compatibility patch")
    
    return content_module

# Apply automatically when imported
apply_langchain_patch()