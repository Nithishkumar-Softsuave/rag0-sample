"""Week 9: this app's own MCP servers.

Each module is a standalone MCP server process -- it offers tools, it never
runs a model and has no idea which AI is calling it. The AI runs on the
host side (agents/mcp_host.py + agents/agent.py). See docs/week9-mcp.md.

    menu_server.py        -- our capability: search menus, convert prices
    ingredient_server.py  -- Track B bolt-on: the ingredient database
"""
