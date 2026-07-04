"""Search-replace edit parsing and transactional application."""

from repo_surgeon.editing.search_replace import ApplyResult, apply_patches, parse_search_replace

__all__ = ["ApplyResult", "apply_patches", "parse_search_replace"]
