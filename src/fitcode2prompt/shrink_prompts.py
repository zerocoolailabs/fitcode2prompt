"""Compression prompt templates for different tiers of code summarization."""

from typing import Dict


# Tier-specific compression prompts for code files
# Each tier represents a target compression percentage
CODE_PROMPTS: Dict[int, str] = {
    # Tier 100: No compression - original code preserved
    100: "{code}",
    
    # Tier 95: Minimal compression (5%) - remove only non-essential elements
    95: """Compress the following code by 5%. The output MUST be shorter than the input.

Remove ONLY:
- Import statements
- ALL blank lines and excessive whitespace
- Commented-out code (code that is commented out, NOT regular comments)
- print() statements used for debugging

KEEP:
- ALL regular comments (explanatory comments, TODOs, warnings)
- ALL docstrings
- ALL actual code
- Function and class definitions unchanged

    Respond only with the compressed code, no additional text.
{code}""",
    
    # Tier 85: Light compression (15%) - remove redundant elements while preserving structure
    85: """Compress the following code by 15%. The output MUST be 85% of the original size.

Remove:
- Import statements and other non-critical information at the top of the file
- ALL logging statements (unless they log errors or critical events)
- ALL blank lines and excessive whitespace
- ALL commented-out code
- ALL redundant comments (keep only critical warnings/security notes)
- ALL docstrings except those documenting complex algorithms

Try to keep in original form:
- All function and class definitions
- All actual implementation code
- Critical comments (security warnings, complex algorithm explanations)
- Business logic and core functionality

    Respond only with the compressed code, no additional text.
{code}""",
    
    # Tier 50: Medium compression (50%) - significant reduction while keeping key logic
    50: """Compress the following code by 50%. The output MUST be approximately half the size.

Remove:
- Import statements and other non-critical information at the top of the file
- ALL logging statements (unless they log errors or critical events)
- ALL blank lines and excessive whitespace
- ALL commented-out code
- ALL redundant comments (keep only critical warnings/security notes)
- ALL docstrings except those documenting complex algorithms
- All getter/setter methods

Then REPLACE:
- Trivial one-liner functions that are canonical or self-explanatory → just leave the function signature only
- Functions longer than 10 lines that aren't complex → Replace function body with one to three lines describing what it does

Try to Keep as actual code:
- Complex algorithms or business logic
- Critical or important operations such as external calls and state changes
- Security/auth checks
- Non-obvious implementations

    Respond only with the compressed code, no additional text.
{code}""",
    
    # Tier 10: Heavy compression (90%) - skeleton with key signatures only
    10: """Compress the following code by 90%. The output MUST be 10% of original size.

Replace the ENTIRE file with:
- A summary at the top of the file's purpose and functionality
- List of key functions/classes with signatures with a one to three line descriptions for non-obvious or complex functions
- When possible, preserve actual code for important or complex logic

    Respond only with the compressed code, no additional text.
{code}""",
    
    # Tier 0: Maximum compression - brief textual summary only
    0: """Summarize the following code in one to three sentences.
    Respond only with the compressed code, no additional text.
{code}"""
}

def get_doc_prompt(percent: int) -> str:
    """Generate documentation compression prompt for given percentage.
    
    Args:
        percent: Compression percentage (5, 15, 50, 90)
        
    Returns:
        Formatted prompt string
    """
    return f"""Compress the following documentation by {percent}%.

Start by removing whitespace, then eliminate redundant statements or contents. Then summarize less important details. Continue to summarize more and more as needed to reach the target compression level, trying to retain as much detail about the critical points as possible.

    Respond only with the compressed code, no additional text.
{{code}}"""

# Documentation-specific prompts for markdown, text, and other doc files
DOC_PROMPTS: Dict[int, str] = {
    # Tier 100: No compression
    100: "{code}",
    
    # Tier 95: Minimal compression (5%)
    95: get_doc_prompt(5),
    
    # Tier 85: Light compression (15%)
    85: get_doc_prompt(15),
    
    # Tier 50: Medium compression (50%)
    50: get_doc_prompt(50),
    
    # Tier 10: Heavy compression (90%)
    10: get_doc_prompt(90),
    
    # Tier 0: Maximum compression - brief summary
    0: """Summarize the following documentation in one to three sentences, capturing its main purpose and key points.
    Respond only with the compressed code, no additional text.
{code}"""
}

# Tier descriptions for documentation/UI
TIER_DESCRIPTIONS: Dict[int, str] = {
    0: "Max compression ~100% - one sentence description",
    10: "Heavy compression 90% - bullet point summary",
    50: "Medium compression 50% - structured summary format",
    85: "Light compression 15% - remove redundant comments only",
    95: "Trim compression 5% - remove imports and whitespace",
    100: "No compression 0% - original code preserved"
}

# Compression percentages for each tier
TIER_PERCENTAGES: Dict[int, int] = {
    100: 0,   # No compression
    95: 5,    # 5% compression
    85: 15,   # 15% compression
    50: 50,   # 50% compression
    10: 90,   # 90% compression
    0: 100    # ~100% compression
}
