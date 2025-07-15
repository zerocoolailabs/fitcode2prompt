"""Strategic compression planning module."""

import json
import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()

# Suppress litellm's verbose logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Tier names mapping
TIER_NAMES: Dict[int, str] = {
    100: "none", 95: "trim", 85: "light", 
    50: "medium", 10: "heavy", 0: "max"
}

# Minimum token size for compressed files
MIN_COMPRESSED_TOKENS = 100


def get_planner_prompt(file_count: int, total_tokens: int, budget: int, fixed_section: str, file_list: str) -> str:
    """Generate the planner prompt with the given parameters.
    
    Args:
        file_count: Total number of files
        total_tokens: Total token count
        budget: Token budget
        fixed_section: Section describing fixed compression files
        file_list: List of files to plan for
        
    Returns:
        Formatted prompt string
    """
    return f"""You are a code compression strategist. Create a compression plan that fits within the token budget.

Input:
- Total files: {file_count}
- Total tokens: {total_tokens:,}
- Budget: {budget:,} tokens maximum
{fixed_section}
Files to plan compression for:
{file_list}

Compression levels:
- 100: No compression
- 95: Trim compression (minimal)
- 50: Medium compression
- 10: Heavy compression
- 0: Replace file with 1-3 sentence description

First, explain your compression strategy in 1-2 sentences. Consider file importance based on paths (core files vs tests vs interfaces).

Then create your compression plan using the validate_budget function. The function will return:
- true: Plan fits within budget, you're done!
- false: Plan exceeds budget - use more aggressive compression

Keep adjusting and validating until you get true."""


class Planner:
    """Strategic planner for compression strategies."""

    def __init__(self, model: str):
        """
        Initialize the planner.

        Args:
            model: LLM model to use for planning (required)
        """
        self.model = model
    
    def make_plan(
        self,
        files: List[Tuple[str, int]],
        budget: int,
        buffer_percent: int = 10,
        verbose: bool = False,
        fixed_files: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a compression plan that fits the codebase within token budget.

        Args:
            files: List of (filepath, token_count) tuples
            budget: Maximum allowed tokens in final output
            buffer_percent: Percentage to reduce budget by for safety margin (default: 10%)
            verbose: Enable verbose logging
            fixed_files: List of files with fixed compression (dict with path, original_tokens, tier)

        Returns:
            Dictionary containing the compression plan
        """
        fixed_files = fixed_files or []
        
        # Step 1: Calculate effective budget with buffer
        effective_budget = self._apply_buffer_to_budget(budget, buffer_percent)
        
        # Step 2: Process fixed compression files
        fixed_tokens, remaining_budget = self._process_fixed_files(fixed_files, effective_budget)
        
        # Step 3: Log planning details if verbose
        if verbose:
            self._log_planning_details(files, fixed_files, budget, effective_budget, fixed_tokens, remaining_budget, buffer_percent)
        
        # Step 4: Generate prompt for LLM
        prompt = self._generate_planner_prompt(files, fixed_files, fixed_tokens, remaining_budget, budget, effective_budget)
        
        # Step 5: Get compression plan from LLM
        result = self._get_plan_from_llm(prompt, effective_budget, files, fixed_files, fixed_tokens, buffer_percent, budget, verbose)
        
        return result
    
    def _calculate_fixed_tokens(self, fixed_files: List[Dict[str, Any]]) -> int:
        """Calculate tokens used by fixed compression files.
        
        Args:
            fixed_files: List of files with fixed compression
            
        Returns:
            Total tokens for fixed files
        """
        return sum(
            self._estimate_file_tokens(f['original_tokens'], f['tier'])
            for f in fixed_files
        )
    
    def _estimate_file_tokens(self, original_tokens: int, tier: int) -> int:
        """Estimate compressed tokens for a file at given tier.
        
        Args:
            original_tokens: Original file token count
            tier: Compression tier
            
        Returns:
            Estimated token count after compression
        """
        if tier == 0:
            # Max compression: minimum of 100 or original size
            return min(MIN_COMPRESSED_TOKENS, original_tokens)
        elif tier == 100:
            # No compression
            return original_tokens
        else:
            # Calculate with minimum threshold
            calculated = int(original_tokens * tier / 100)
            return min(max(MIN_COMPRESSED_TOKENS, calculated), original_tokens)
    
    def _build_fixed_section(self, fixed_files: List[Dict[str, Any]], fixed_tokens: int, remaining_budget: int) -> str:
        """Build the fixed files section for the prompt.
        
        Args:
            fixed_files: List of files with fixed compression
            fixed_tokens: Total tokens for fixed files
            remaining_budget: Remaining budget after fixed files
            
        Returns:
            Formatted string for fixed files section
        """
        if not fixed_files:
            return ""
        
        lines = [
            f"\n- Fixed compression files: {len(fixed_files)} files using {fixed_tokens:,} tokens",
            f"- Remaining budget for planning: {remaining_budget:,} tokens",
            "\nFiles with user-specified compression (DO NOT include in your plan):"
        ]
        
        for f in fixed_files:
            tier_name = TIER_NAMES.get(f['tier'], f"tier-{f['tier']}")
            lines.append(f"{f['path']} - {f['original_tokens']:,} tokens → {tier_name} compression")
        
        return "\n".join(lines) + "\n"
    
    def _calculate_estimated_tokens(self, files_plan: List[Dict[str, Any]]) -> Tuple[int, List[Tuple[Dict[str, Any], int]]]:
        """Calculate estimated tokens for a compression plan.
        
        Args:
            files_plan: List of file plans
            
        Returns:
            Tuple of (total_estimated, file_estimates)
        """
        file_estimates = []
        total_estimated = 0
        
        for file_plan in files_plan:
            estimated = self._estimate_file_tokens(
                file_plan["original_tokens"], 
                file_plan["tier"]
            )
            total_estimated += estimated
            file_estimates.append((file_plan, estimated))
        
        return total_estimated, file_estimates
    
    
    def _build_validation_tools(self):
        """Define the validation tool for the planner."""
        return [{
            "type": "function",
            "function": {
                "name": "validate_budget",
                "description": "Validate that the compression plan fits within budget. Automatically applies 100 token minimum to all compressed files. Returns true if within budget, false if over budget",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "original_tokens": {"type": "integer"},
                                    "tier": {"type": "integer", "enum": [100, 95, 85, 50, 10, 0]}
                                },
                                "required": ["path", "original_tokens", "tier"]
                            }
                        }
                    },
                    "required": ["files"]
                }
            }
        }]
    
    def _process_tool_response(
        self, 
        response: Any, 
        effective_budget: int, 
        files: List[Tuple[str, int]], 
        fixed_files: List[Dict[str, Any]], 
        fixed_tokens: int, 
        buffer_percent: int, 
        budget: int
    ) -> Tuple[Dict[str, Any], bool, Any, Any]:
        """Process the LLM's tool call response and build the result.
        
        Args:
            response: LLM response
            effective_budget: Budget after buffer
            files: Files for planner
            fixed_files: Files with fixed compression
            fixed_tokens: Tokens for fixed files
            buffer_percent: Buffer percentage
            budget: Original budget
            
        Returns:
            Tuple of (result, is_valid, tool_call, response)
        """
        if not response.choices[0].message.tool_calls:
            logger.warning("Model did not use validation tool")
            return self._create_error_result(
                "Model did not generate a proper plan",
                response.choices[0].message.content
            ), False, None, response
        
        tool_call = response.choices[0].message.tool_calls[0]
        plan_data = json.loads(tool_call.function.arguments)
        
        # Calculate the validation ourselves
        total_estimated, _ = self._calculate_estimated_tokens(plan_data["files"])
        
        # Build result
        result = self._build_plan_result(
            plan_data, files, fixed_files, fixed_tokens, 
            total_estimated, budget, effective_budget, 
            buffer_percent, response.choices[0].message.content
        )
        
        return result, True, tool_call, response
    
    def _create_error_result(self, error: str, reasoning: Optional[str] = None) -> Dict[str, Any]:
        """Create an error result.
        
        Args:
            error: Error message
            reasoning: Optional reasoning from model
            
        Returns:
            Error result dictionary
        """
        return {
            "valid": False,
            "error": error,
            "reasoning": reasoning or "No reasoning provided"
        }
    
    def _build_plan_result(
        self,
        plan_data: Dict[str, Any],
        files: List[Tuple[str, int]],
        fixed_files: List[Dict[str, Any]],
        fixed_tokens: int,
        total_estimated: int,
        budget: int,
        effective_budget: int,
        buffer_percent: int,
        reasoning: Optional[str]
    ) -> Dict[str, Any]:
        """Build the plan result dictionary.
        
        Args:
            plan_data: Plan data from LLM
            files: Files for planner
            fixed_files: Files with fixed compression
            fixed_tokens: Tokens for fixed files
            total_estimated: Estimated tokens for plan
            budget: Original budget
            effective_budget: Budget after buffer
            buffer_percent: Buffer percentage
            reasoning: Reasoning from model
            
        Returns:
            Plan result dictionary
        """
        # Create file lookup
        file_dict = {f[0]: f[1] for f in files}
        
        result = {
            "valid": True,
            "total_estimated": total_estimated + fixed_tokens,
            "budget": budget,
            "effective_budget": effective_budget,
            "buffer_percent": buffer_percent,
            "reasoning": reasoning or "No reasoning provided",
            "files": []
        }
        
        # Add planner files
        for file_plan in plan_data["files"]:
            result["files"].append({
                "path": file_plan["path"],
                "original_tokens": file_dict.get(file_plan["path"], file_plan["original_tokens"]),
                "tier": file_plan["tier"]
            })
        
        # Add fixed files
        result["files"].extend(fixed_files)
        
        return result

    def _apply_buffer_to_budget(self, budget: int, buffer_percent: int) -> int:
        """Apply buffer percentage to budget for safety margin."""
        return int(budget * (1 - buffer_percent / 100))
    
    def _process_fixed_files(self, fixed_files: List[Dict], effective_budget: int) -> Tuple[int, int]:
        """Process fixed compression files and calculate remaining budget."""
        fixed_tokens = self._calculate_fixed_tokens(fixed_files)
        remaining_budget = effective_budget - fixed_tokens
        return fixed_tokens, remaining_budget
    
    def _log_planning_details(self, files, fixed_files, budget, effective_budget, fixed_tokens, remaining_budget, buffer_percent):
        """Log planning details for verbose mode."""
        logger.info(f"Planning compression for {len(files)} files")
        logger.info(f"Total tokens: {sum(f[1] for f in files):,}")
        logger.info(f"Budget: {budget:,} tokens (effective: {effective_budget:,} with {buffer_percent}% buffer)")
        if fixed_files:
            logger.info(f"Fixed compression files: {len(fixed_files)} using {fixed_tokens:,} tokens")
            logger.info(f"Remaining budget for planner: {remaining_budget:,} tokens")
    
    def _generate_planner_prompt(
        self, 
        files: List[Tuple[str, int]], 
        fixed_files: List[Dict[str, Any]], 
        fixed_tokens: int, 
        remaining_budget: int, 
        budget: int, 
        effective_budget: int
    ) -> str:
        """Generate the prompt for the LLM planner.
        
        Args:
            files: List of (path, tokens) tuples for planner
            fixed_files: List of files with fixed compression
            fixed_tokens: Total tokens for fixed files
            remaining_budget: Remaining budget after fixed files
            budget: Original budget
            effective_budget: Budget after applying buffer
            
        Returns:
            Formatted prompt string
        """
        # Build fixed files section if any
        fixed_section = self._build_fixed_section(fixed_files, fixed_tokens, remaining_budget)
        
        # Build the file list string (only files for planner)
        file_list = "\n".join(f"{path} - {tokens:,} tokens" for path, tokens in files)
        
        # Calculate total tokens (including fixed files)
        total_all_files = sum(f[1] for f in files) + sum(f['original_tokens'] for f in fixed_files)
        total_file_count = len(files) + len(fixed_files)
        
        # Generate prompt with parameters
        return get_planner_prompt(
            file_count=total_file_count,
            total_tokens=total_all_files,
            budget=effective_budget,
            fixed_section=fixed_section,
            file_list=file_list
        )
    
    def _get_plan_from_llm(
        self, 
        prompt: str, 
        effective_budget: int, 
        files: List[Tuple[str, int]], 
        fixed_files: List[Dict[str, Any]], 
        fixed_tokens: int, 
        buffer_percent: int, 
        budget: int, 
        verbose: bool
    ) -> Dict[str, Any]:
        """Get compression plan from LLM.
        
        Args:
            prompt: Prompt for LLM
            effective_budget: Budget after buffer
            files: Files for planner
            fixed_files: Files with fixed compression
            fixed_tokens: Tokens for fixed files
            buffer_percent: Buffer percentage
            budget: Original budget
            verbose: Enable verbose logging
            
        Returns:
            Plan result dictionary
        """
        tools = self._build_validation_tools()
        messages = [
            {"role": "system", "content": "You are a code compression strategist."},
            {"role": "user", "content": prompt}
        ]
        
        if verbose:
            logger.info(f"Calling {self.model} for planning...")
        
        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            result, _, _, _ = self._process_tool_response(
                response, effective_budget, files, fixed_files, 
                fixed_tokens, buffer_percent, budget
            )
            
            if verbose:
                self._log_plan_result(result, budget)
            
            return result
                
        except Exception as e:
            logger.error(f"Error during planning: {e}")
            raise
    
    def _log_plan_result(self, result: Dict[str, Any], budget: int) -> None:
        """Log plan result for verbose mode.
        
        Args:
            result: Plan result
            budget: Original budget
        """
        logger.info(f"Plan generated. Valid: {result.get('valid', False)}")
        logger.info(f"Total estimated: {result.get('total_estimated', 0):,} / {budget:,} tokens")
    
