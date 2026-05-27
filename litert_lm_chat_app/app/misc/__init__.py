"""Optional local extensions to the litert chat app.

Modules in this package are NOT required for the core voice loop. They are
loaded opportunistically by `app.main` (wrapped in try/except). If `misc/`
is missing, the app still runs with the registered web-search tools only.
"""
