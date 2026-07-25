"""Quick check: what does Python actually interpret the DIFF_TOOL_PATH as?"""
from agentFour.config import DIFF_TOOL_PATH

print(f"Configured path: {DIFF_TOOL_PATH!r}")
print(f"Path length: {len(DIFF_TOOL_PATH)}")
print(f"Hex bytes: {DIFF_TOOL_PATH.encode('utf-8').hex(' ')}")
