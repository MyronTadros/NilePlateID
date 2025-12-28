<INSTRUCTIONS>
- Always prefer simple CLI entrypoints in `src/` with `argparse`.
- Don’t hardcode absolute paths; use repo-relative defaults.
- Every script must have: `--help`, clear logging, and deterministic outputs.
- Add safety: never overwrite outputs without `--force`.
- Use type hints where reasonable.

Reference:
- Codex custom instructions (AGENTS.md): https://developers.openai.com/codex/guides/agents-md
</INSTRUCTIONS>
