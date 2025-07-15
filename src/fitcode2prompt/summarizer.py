"""Core Summarizer class for Summarizely."""

import asyncio
import json
import logging
import platform
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, NamedTuple, Union

from .file_discovery import FileDiscovery
from .tokenizer import Tokenizer
from .planner import Planner
from .async_processor import AsyncProcessor

# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# Legacy color constants for backward compatibility
CYAN = Colors.CYAN
GREEN = Colors.GREEN
YELLOW = Colors.YELLOW
RED = Colors.RED
BOLD = Colors.BOLD
RESET = Colors.RESET

# Tier names mapping
TIER_NAMES: Dict[int, str] = {
    100: "none", 95: "trim", 85: "light",
    50: "medium", 10: "heavy", 0: "max"
}


class FileStats(NamedTuple):
    """File statistics for processing."""
    files_with_tokens: List[Tuple[str, int]]
    fixed_compression_files: List[Dict[str, Any]]
    planner_files: List[Tuple[str, int]]
    total_tokens: int


class Summarizer:
    """Main class for summarizing codebases within token budgets."""

    def __init__(
        self,
        path: str,
        llm_model_planner: str,
        llm_model_summarizer: str,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        strict_glob: bool = False,
        budget: int = None,
        model: str = "cl100k_base",
        output_dir: str = ".",
        verbose: bool = False,
        use_planner: bool = False,
        buffer_percent: int = 10,
        default_compression: int = 95,
        respect_gitignore: bool = True,
        compression_config: Dict[str, int] = None,
        line_number_patterns: List[str] = None,
        no_clipboard: bool = False,
        return_results: bool = False
    ):
        """
        Initialize the Summarizer.

        Args:
            path: Base path to search for files
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
            strict_glob: Use standard glob patterns without automatic recursion
            budget: Maximum token budget
            model: Tiktoken encoding for token counting
            llm_model_planner: LLM model for compression planning
            llm_model_summarizer: LLM model for summarization
            output_dir: Directory for output files (default: current directory)
            verbose: Enable verbose logging
            use_planner: Use AI planner for compression strategy (default: False)
            buffer_percent: Percentage buffer for planner budget (default: 10%)
            default_compression: Default compression tier when not using planner (default: 95%)
            respect_gitignore: Whether to respect .gitignore patterns (default: True)
            compression_config: Dict mapping glob patterns to compression tiers
            line_number_patterns: List of glob patterns for files to add line numbers
            no_clipboard: Whether to skip copying output to clipboard (default: False)
            return_results: Whether to return the summary text instead of exit code (default: False)
        """
        self.path = path
        self.include_patterns = include_patterns or ['**/*']
        self.exclude_patterns = exclude_patterns or []
        self.strict_glob = strict_glob
        self.budget = budget
        self.model = model
        self.llm_model_planner = llm_model_planner
        self.llm_model_summarizer = llm_model_summarizer
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.use_planner = use_planner
        self.buffer_percent = buffer_percent
        self.default_compression = default_compression
        self.respect_gitignore = respect_gitignore
        self.compression_config = compression_config or {}
        self.line_number_patterns = line_number_patterns or []
        self.no_clipboard = no_clipboard
        self.return_results = return_results

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(message)s' if not verbose else '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Initialize components
        self.file_discovery = FileDiscovery()
        # Use the same model for tokenization that we'll use for summarization
        self.tokenizer = Tokenizer(llm_model_summarizer)

    def run(self) -> Union[int, str, None]:
        """
        Run the full summarization process.

        Returns:
            If return_results=False: Exit code (0 for success)
            If return_results=True: String containing the summary text, or None on error
        """
        start_time = time.time()
        
        try:
            # Step 1: Prepare patterns
            include_patterns, exclude_patterns = self._prepare_patterns()
            
            # Step 2: Discover all files
            files, errors = self.file_discovery.find_files(
                self.path, include_patterns, exclude_patterns, self.respect_gitignore
            )
            
            # Report errors
            for error in errors:
                logging.warning(error)
                
            if not files:
                logging.error("No valid files found matching the provided patterns")
                if self.return_results:
                    return None
                return 1
                
            print(f"Found {len(files)} files to process")
            
            # Step 3: Build user compression mapping
            user_compression_mapping = self._build_user_compression_mapping(files)
            
            # Step 4: Process user-specified compressions first
            user_results = []
            user_actual_tokens = 0
            planner_files = []
            total_original_tokens = 0
            
            # Show user compression summary if any
            if user_compression_mapping:
                print(f"\n{BOLD}{CYAN}=== USER-SPECIFIED COMPRESSIONS ==={RESET}")
                tier_counts = {}
                for tier in user_compression_mapping.values():
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                for tier, count in sorted(tier_counts.items()):
                    print(f"  {TIER_NAMES.get(tier, f'tier-{tier}')}: {count} files")
            
            for file in files:
                filepath = str(file)
                
                # Read file content and count tokens
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    original_tokens = self.tokenizer.count_tokens(content)
                    total_original_tokens += original_tokens
                    
                    if filepath in user_compression_mapping:
                        # This is a user-specified file - compress it now
                        tier = user_compression_mapping[filepath]
                        
                        # Create a mini-plan for this single file
                        mini_plan = {
                            'files': [{
                                'path': filepath,
                                'original_tokens': original_tokens,
                                'tier': tier
                            }]
                        }
                        
                        # Run compression on this file
                        result = self._run_async_summarization(mini_plan)[0]
                        user_results.append(result)
                        user_actual_tokens += self.tokenizer.count_tokens(result['summary'])
                    else:
                        # This file goes to the planner
                        planner_files.append((filepath, original_tokens))
                        
                except UnicodeDecodeError:
                    logging.warning(f"Cannot decode {file}")
                    continue
            
            # Report user compression results
            if user_results:
                print(f"\n{GREEN}✓ Completed {len(user_results)} user-specified compressions{RESET}")
                print(f"Actual tokens used: {user_actual_tokens:,}")
                
                # Check budget even if no planner files
                if self.budget and user_actual_tokens > self.budget:
                    print(f"\n{RED}ERROR: User-specified compressions ({user_actual_tokens:,} tokens) exceed budget ({self.budget:,} tokens){RESET}")
                    if self.return_results:
                        return None
                    return 1
            
            # Step 5: Now handle planner files if needed
            planner_results = []
            
            if planner_files and self.budget:
                # Calculate precise remaining budget
                remaining_budget = self.budget - user_actual_tokens
                
                print(f"\n{BOLD}{CYAN}=== PLANNING REMAINING FILES ==={RESET}")
                print(f"User-specified compressions: {len(user_results)} files using {user_actual_tokens:,} actual tokens")
                print(f"Remaining budget for planner: {remaining_budget:,} tokens")
                print(f"Files for planner: {len(planner_files)}")
                
                if remaining_budget <= 0:
                    print(f"\n{RED}ERROR: User-specified compressions ({user_actual_tokens:,} tokens) exceed budget ({self.budget:,} tokens){RESET}")
                    if self.return_results:
                        return None
                    return 1
                
                # Create planner plan with adjusted budget
                planner_plan = self._create_planner_plan(planner_files, remaining_budget)
                if not planner_plan:
                    if self.return_results:
                        return None
                    return 1
                    
                # Execute planner compressions
                planner_results = self._run_async_summarization(planner_plan)
            elif planner_files:
                # No budget specified, use default compression
                print(f"\n{BOLD}=== APPLYING DEFAULT COMPRESSION ==={RESET}")
                default_plan = {
                    'files': [
                        {
                            'path': filepath,
                            'original_tokens': tokens,
                            'tier': self.default_compression
                        }
                        for filepath, tokens in planner_files
                    ]
                }
                planner_results = self._run_async_summarization(default_plan)
            
            # Step 6: Combine all results and output
            all_results = user_results + planner_results
            
            # Create a combined plan for output statistics
            combined_plan = {
                'files': [r for r in all_results],
                'budget': self.budget,
                'total_estimated': sum(self.tokenizer.count_tokens(r['summary']) for r in all_results)
            }
            
            return self._finalize_output(all_results, total_original_tokens, combined_plan, time.time() - start_time)
            
        except Exception as e:
            logging.error(f"Unexpected error during summarization: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            if self.return_results:
                return None
            return 1
    
    
    def _finalize_output(
        self, 
        results: List[Dict[str, Any]], 
        total_tokens: int, 
        plan: Dict[str, Any], 
        total_time: float
    ) -> Union[int, str, None]:
        """Finalize output by writing files and displaying results.
        
        Args:
            results: Processing results
            total_tokens: Total original tokens
            plan: Compression plan used
            total_time: Total execution time
            
        Returns:
            If return_results=False: 0 for success
            If return_results=True: String containing the summary text, or None on error
        """
        self._write_output(results, total_tokens, plan, total_time)
        
        if self.return_results:
            # Read the output file content and return it
            output_path = self.output_dir / "fitcode2prompt.out"
            try:
                with open(output_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Failed to read output file: {e}")
                return None
        
        return 0

    def _create_planner_plan(self, planner_files: List[Tuple[str, int]], remaining_budget: int) -> Optional[Dict[str, Any]]:
        """
        Create a compression plan using the planner for the remaining files.
        
        Args:
            planner_files: List of (filepath, tokens) tuples for files not user-specified
            remaining_budget: Budget remaining after user-specified compressions
            
        Returns:
            Compression plan or None if planning fails
        """
        # Apply buffer to the remaining budget
        planner_budget = int(remaining_budget * (1 - self.buffer_percent / 100))
        
        print(f"\n{CYAN}Submitting to planner ({self.llm_model_planner})...{RESET}")
        print(f"Planner budget: {planner_budget:,} tokens (with {self.buffer_percent}% buffer)")
        
        planner = Planner(model=self.llm_model_planner)
        
        try:
            plan = planner.make_plan(
                files=planner_files,
                budget=planner_budget,
                buffer_percent=0,  # Buffer already applied
                verbose=False,
                fixed_files=[]  # No fixed files for planner
            )
        except Exception as e:
            logging.error(f"Planner failed: {e}")
            return None
            
        # Pretty print the reasoning
        self._print_planner_reasoning(plan)
        
        # Calculate compression percentage
        total_estimated = plan.get('total_estimated', 0)
        usage_pct = (total_estimated / planner_budget * 100) if planner_budget > 0 else 0
        
        print(f"\n{BOLD}=== COMPRESSION RESULT ==={RESET}")
        print(f"Estimated output: {total_estimated:,} tokens")
        print(f"Target budget: {planner_budget:,} tokens")
        print(f"Usage: {usage_pct:.1f}% of budget")
        
        # Check if plan is valid
        if not plan.get('valid', False):
            print(f"\n{RED}✗ Planner failed to create valid plan{RESET}")
            return None
            
        if total_estimated > planner_budget:
            print(f"\n{RED}✗ Plan exceeds budget{RESET}")
            return None
            
        print(f"\n{GREEN}✓ Plan fits within budget{RESET}")
        
        # Print tier distribution
        if plan.get('files'):
            tier_counts = {}
            for f in plan.get('files', []):
                tier = f.get('tier', 100)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            if tier_counts:
                print(f"\n{BOLD}Tier distribution:{RESET}")
                for tier in sorted(tier_counts.keys(), reverse=True):
                    tier_name = TIER_NAMES.get(tier, f"tier-{tier}")
                    count = tier_counts[tier]
                    print(f"  {tier_name:<8}: {count:3} files")
        
        return plan
    
    def _build_user_compression_mapping(self, files: List[Path]) -> Dict[str, int]:
        """
        Build a mapping of files to user-specified compression levels.
        
        This method processes all compression levels in order and assigns files
        to their specified compression tier. Conflicts are resolved by taking
        the highest compression level (lowest tier value).
        
        Args:
            files: List of all discovered files
            
        Returns:
            Dict mapping file paths to compression tiers
        """
        file_to_compression = {}
        
        # Process each compression level
        # Lower tier values = higher compression, so process in ascending order
        # This way if a file appears in multiple levels, highest compression wins
        # Get all unique tiers from compression_config
        all_tiers = sorted(set(self.compression_config.values()))
        for tier in all_tiers:
            # Get patterns for this tier from compression_config
            patterns_for_tier = [
                pattern for pattern, level in self.compression_config.items() 
                if level == tier
            ]
            
            if not patterns_for_tier:
                continue
                
            # Find all files matching patterns for this tier
            for file in files:
                filepath = str(file)
                
                # Skip if already assigned (conflict resolution: first match wins)
                if filepath in file_to_compression:
                    continue
                    
                # Check if file matches any pattern for this tier
                for pattern in patterns_for_tier:
                    if self._file_matches_pattern(filepath, pattern):
                        file_to_compression[filepath] = tier
                        break
        
        return file_to_compression
    
    def _file_matches_pattern(self, filepath: str, pattern: str) -> bool:
        """
        Check if a file matches a compression pattern.
        
        Args:
            filepath: Full path to the file
            pattern: Pattern to match (may include '::' for content search)
            
        Returns:
            True if file matches the pattern
        """
        from fnmatch import fnmatch
        from pathlib import Path
        
        # Get relative path for matching
        try:
            base = Path(self.path).resolve()
            file_path = Path(filepath).resolve()
            
            # If base is a file, use its parent directory
            if base.is_file():
                base = base.parent
                
            rel_path = str(file_path.relative_to(base))
        except ValueError:
            # Not relative to base, use just the filename
            rel_path = Path(filepath).name
            
        # Handle content-based patterns
        if '::' in pattern:
            glob_part, search_term = pattern.split('::', 1)
            
            # Check glob pattern first
            if not self._matches_glob(rel_path, glob_part):
                return False
                
            # Then check content
            try:
                import re
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return bool(re.search(search_term, content, re.MULTILINE | re.IGNORECASE))
            except Exception:
                return False
        else:
            # Regular glob pattern
            return self._matches_glob(rel_path, pattern)
    
    def _matches_glob(self, path: str, pattern: str) -> bool:
        """
        Check if a path matches a glob pattern.
        
        Args:
            path: Path to check (relative)
            pattern: Glob pattern
            
        Returns:
            True if path matches pattern
        """
        from fnmatch import fnmatch
        from pathlib import Path
        
        # Special handling for patterns like **/oracles/**
        # This should match any file within the oracles directory tree
        if pattern.startswith('**/') and pattern.endswith('/**'):
            # Extract the directory name
            dir_name = pattern[3:-3]  # Remove **/ and /**
            # Check if the path contains this directory
            return f'/{dir_name}/' in f'/{path}' or path.startswith(f'{dir_name}/')
        
        # Use Path.match for patterns with **, fnmatch for others
        if '**' in pattern:
            # Try matching with Path.match
            if Path(path).match(pattern):
                return True
            # Also try with /* appended for directory matching
            if pattern.endswith('/**') and Path(path).match(pattern + '*'):
                return True
            # Try without leading **/ for relative paths
            if pattern.startswith('**/') and Path(path).match(pattern[3:]):
                return True
        else:
            # Check full path
            if fnmatch(path, pattern):
                return True
            # Also check basename
            if fnmatch(Path(path).name, pattern):
                return True
        return False
    
    def _make_recursive(self, pattern: str) -> str:
        """
        Transform a pattern to be recursive if it doesn't already have **.

        Examples:
            *.py -> **/*.py
            src/*.py -> src/**/*.py
            **/*.py -> **/*.py (unchanged)
            *.py::term -> **/*.py::term (transforms glob part only)
        """
        # Handle :: patterns - only transform the glob part
        if '::' in pattern:
            glob_part, search_term = pattern.split('::', 1)
            transformed_glob = self._make_recursive(glob_part)
            return f"{transformed_glob}::{search_term}"

        if '**' in pattern:
            # Already has recursive wildcard
            return pattern

        # Split on last slash to handle paths
        if '/' in pattern:
            parts = pattern.rsplit('/', 1)
            return f"{parts[0]}/**/{parts[1]}"
        else:
            # No path, just add **/ prefix
            return f"**/{pattern}"



    def count_tokens(self) -> int:
        """
        Count tokens for all discovered files.

        Returns:
            Exit code (0 for success)
        """
        # Transform patterns if not using strict glob
        if not self.strict_glob:
            include_patterns = [self._make_recursive(p) for p in self.include_patterns]
            exclude_patterns = [self._make_recursive(p) for p in self.exclude_patterns]
        else:
            include_patterns = self.include_patterns
            exclude_patterns = self.exclude_patterns

        # Discover files
        files, errors = self.file_discovery.find_files(
            self.path, include_patterns, exclude_patterns, self.respect_gitignore
        )

        # Report errors
        for error in errors:
            logging.warning(error)

        if not files:
            logging.error("No valid files found matching the provided patterns")
            return 1

        # Count tokens for each file
        total_tokens = 0
        file_tokens = []

        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                tokens = self.tokenizer.count_tokens(content)
                file_tokens.append({
                    "file": str(file),
                    "tokens": tokens
                })
                total_tokens += tokens
                logging.info(f"{file}: {tokens:,} tokens")
            except UnicodeDecodeError:
                logging.warning(f"Cannot decode {file}")

        # Output summary
        print(f"\nTotal files: {len(file_tokens)}")
        print(f"Total tokens: {total_tokens:,}")
        if self.budget:
            print(f"Budget: {self.budget:,} tokens")
            print(f"Usage: {(total_tokens / self.budget * 100):.1f}% of budget")

        return 0

    def _print_planner_reasoning(self, plan: Dict[str, Any]) -> None:
        """Pretty print the planner's reasoning.
        
        Args:
            plan: Plan dictionary with reasoning
        """
        print(f"\n{Colors.BOLD}=== PLANNER REASONING ==={Colors.RESET}")
        
        reasoning = plan.get('reasoning', '')
        if not reasoning:
            return
        
        # Clean reasoning text
        reasoning = self._clean_reasoning_text(reasoning)
        
        # Format and print paragraphs
        for paragraph in reasoning.split('\n\n'):
            if paragraph.strip():
                wrapped = textwrap.fill(paragraph, width=80)
                print(wrapped)
                print()
    
    def _clean_reasoning_text(self, reasoning: str) -> str:
        """Clean reasoning text by removing JSON blocks.
        
        Args:
            reasoning: Raw reasoning text
            
        Returns:
            Cleaned reasoning text
        """
        if '```json' in reasoning:
            reasoning = reasoning.split('```json')[0].strip()
        
        # Remove any lines that look like JSON
        lines = reasoning.split('\n')
        cleaned_lines = [
            line for line in lines 
            if not (line.strip().startswith('{') or line.strip().startswith('['))
        ]
        
        return '\n'.join(cleaned_lines).strip()

    def _print_tier_summary(self, plan: Dict[str, Any]) -> None:
        """Print summary of tier distribution.
        
        Args:
            plan: Plan dictionary with files
        """
        if 'files' not in plan:
            return
        
        # Count files by tier
        tier_counts = self._count_files_by_tier(plan['files'])
        
        print(f"\n{Colors.BOLD}Tier distribution:{Colors.RESET}")
        for tier, count in sorted(tier_counts.items(), reverse=True):
            name = TIER_NAMES.get(tier, f"tier-{tier}")
            print(f"  {name:8s}: {count:3d} files")
    
    def _count_files_by_tier(self, files: List[Dict[str, Any]]) -> Dict[int, int]:
        """Count files grouped by compression tier.
        
        Args:
            files: List of file dictionaries with tier info
            
        Returns:
            Dictionary mapping tier to file count
        """
        tier_counts: Dict[int, int] = {}
        for f in files:
            tier = f['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        return tier_counts

    def _print_file_assignments(self, plan: Dict[str, Any]) -> None:
        """Print file-by-file breakdown of compression assignments.
        
        Args:
            plan: Plan dictionary with file assignments
        """
        print(f"\n{Colors.BOLD}=== FILE ASSIGNMENTS ==={Colors.RESET}")
        base_path = Path(self.path).resolve()
        
        # Sort files by estimated tokens (largest first)
        sorted_files = sorted(
            plan['files'], 
            key=lambda f: self._get_estimated_tokens(f), 
            reverse=True
        )
        
        for f in sorted_files:
            self._print_single_file_assignment(f, base_path)
    
    def _get_estimated_tokens(self, file_info: Dict[str, Any]) -> int:
        """Get estimated token count for a file.
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Estimated token count
        """
        if 'estimated_tokens' in file_info:
            return file_info['estimated_tokens']
        
        original = file_info['original_tokens']
        tier = file_info['tier']
        
        if tier == 0:
            return min(100, original)
        elif tier == 100:
            return original
        else:
            calculated = int(original * tier / 100)
            return min(max(100, calculated), original)
    
    def _print_single_file_assignment(
        self, 
        file_info: Dict[str, Any], 
        base_path: Path
    ) -> None:
        """Print assignment for a single file.
        
        Args:
            file_info: File information dictionary
            base_path: Base path for relative path calculation
        """
        original = file_info['original_tokens']
        tier = file_info['tier']
        estimated = self._get_estimated_tokens(file_info)
        
        # Get relative path
        try:
            rel_path = Path(file_info['path']).relative_to(base_path)
        except ValueError:
            rel_path = file_info['path']
        
        # Get color for tier
        color = self._get_tier_color(tier)
        tier_name = TIER_NAMES.get(tier, f"tier-{tier}")
        
        print(
            f"{color}{str(rel_path):60s} {tier_name:8s} - "
            f"{original:6,} → {estimated:6,} tokens{Colors.RESET}"
        )
    
    def _get_tier_color(self, tier: int) -> str:
        """Get ANSI color code for a compression tier.
        
        Args:
            tier: Compression tier
            
        Returns:
            ANSI color code string
        """
        tier_colors = {
            100: Colors.GREEN,
            95: Colors.CYAN,
            85: Colors.BLUE,
            50: Colors.YELLOW,
            10: Colors.MAGENTA,
            0: Colors.RED
        }
        return tier_colors.get(tier, Colors.RESET)

    def _write_output(
        self, 
        results: List[Dict[str, Any]], 
        total_tokens: int, 
        plan: Dict[str, Any], 
        total_time: float
    ) -> None:
        """
        Write the summarization output to files and clipboard.

        Args:
            results: List of summarization results
            total_tokens: Total original tokens
            plan: The compression plan used
            total_time: Total execution time in seconds
        """
        # Print summary statistics
        self._print_completion_summary(results, total_tokens, total_time)
        
        # Generate timestamp for output files
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Write main output file
        output_path = self._write_summary_file(results, timestamp)
        print(f"\n{Colors.GREEN}Output written to: {output_path}{Colors.RESET}")
        
        # Copy to clipboard if requested
        if not self.no_clipboard:
            self._copy_to_clipboard(output_path)
        
        # Write plan file if planner was used
        if self.use_planner:
            self._write_plan_file(plan, timestamp)
    
    def _print_completion_summary(
        self, 
        results: List[Dict[str, Any]], 
        total_tokens: int, 
        total_time: float
    ) -> None:
        """Print completion summary with statistics.
        
        Args:
            results: Processing results
            total_tokens: Total original tokens
            total_time: Total execution time
        """
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{Colors.BOLD}=== SUMMARIZATION COMPLETE ==={Colors.RESET}")
        print(f"{Colors.GREEN}✓ Success: {len(successful)} files{Colors.RESET}")
        
        if failed:
            print(f"{Colors.RED}✗ Failed: {len(failed)} files{Colors.RESET}")
            for f in failed:
                print(f"  - {f['path']}: {f.get('error', 'Unknown error')}")
        
        # Calculate compression statistics
        total_compressed = sum(r.get('compressed_tokens', 0) for r in successful)
        compression_ratio = (1 - total_compressed / total_tokens) * 100 if total_tokens > 0 else 0
        
        print(f"\nInitial size: {total_tokens:,} tokens")
        print(f"Final size: {total_compressed:,} tokens")
        print(f"Actual compression: {compression_ratio:.1f}%")
        
        if self.budget:
            budget_usage = (total_compressed / self.budget * 100) if self.budget > 0 else 0
            print(f"Budget requested: {self.budget:,} tokens")
            print(f"Budget usage: {budget_usage:.1f}%")
        
        # Display execution time
        self._print_execution_time(total_time)
    
    def _print_execution_time(self, total_time: float) -> None:
        """Print formatted execution time.
        
        Args:
            total_time: Total time in seconds
        """
        if total_time < 60:
            print(f"Total time: {total_time:.1f}s")
        else:
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(f"Total time: {minutes}m {seconds}s")
    
    def _write_summary_file(
        self, 
        results: List[Dict[str, Any]], 
        timestamp: str
    ) -> Path:
        """Write the main summary file.
        
        Args:
            results: Processing results
            timestamp: Timestamp for filename
            
        Returns:
            Path to written file
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "fitcode2prompt.out"
        
        successful = [r for r in results if r['success']]
        base_path = Path(self.path).resolve()
        
        with open(output_path, 'w') as f:
            for i, result in enumerate(successful):
                self._write_file_section(f, result, base_path)
                
                # Add separator between files
                if i < len(successful) - 1:
                    f.write("\n\n---\n\n")
        
        return output_path
    
    def _write_file_section(
        self, 
        file_handle, 
        result: Dict[str, Any], 
        base_path: Path
    ) -> None:
        """Write a single file's section to the output.
        
        Args:
            file_handle: Open file handle
            result: File processing result
            base_path: Base path for relative paths
        """
        # Get relative path
        try:
            rel_path = Path(result['path']).relative_to(base_path)
        except ValueError:
            rel_path = Path(result['path']).name
        
        # Write header
        file_handle.write(f"## {rel_path}\n")
        
        # Write statistics
        original = result['original_tokens']
        compressed = result.get('compressed_tokens', 0)
        tier = result['tier']
        
        if tier < 100:
            actual_compression = ((original - compressed) / original * 100) if original > 0 else 0
            tier_name = TIER_NAMES.get(tier, f"tier-{tier}")
            
            file_handle.write(
                f"**Original:** {original:,} tokens | **Compressed:** {compressed:,} tokens "
                f"({actual_compression:.1f}% actual compression, {tier_name})\n\n"
            )
        else:
            file_handle.write(f"**Original:** {original:,} tokens | **Preserved as-is (none)**\n\n")
        
        # Write content
        file_handle.write(result['summary'])
    
    def _write_plan_file(self, plan: Dict[str, Any], timestamp: str) -> None:
        """Write the compression plan to a JSON file.
        
        Args:
            plan: Compression plan
            timestamp: Timestamp for filename
        """
        plan_path = self.output_dir / "fitcode2prompt_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)

    def _copy_to_clipboard(self, file_path: Path) -> None:
        """Copy file contents to clipboard if supported.
        
        Args:
            file_path: Path to file to copy
        """
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Get clipboard command for platform
            system = platform.system()
            clipboard_cmd = self._get_clipboard_command(system)
            
            if not clipboard_cmd:
                logging.debug(f"No clipboard command available for {system}")
                return
            
            # Execute clipboard command
            self._execute_clipboard_command(clipboard_cmd, content, system)
            print(f"{Colors.GREEN}✓ Output copied to clipboard{Colors.RESET}")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.warning(f"Failed to copy to clipboard: {e}")
        except Exception as e:
            logging.debug(f"Clipboard operation failed: {e}")
    
    def _get_clipboard_command(self, system: str) -> Optional[List[str]]:
        """Get clipboard command for the current platform.
        
        Args:
            system: Platform system name
            
        Returns:
            Clipboard command list or None
        """
        commands = {
            'Darwin': ['pbcopy'],
            'Linux': ['xclip', '-selection', 'clipboard'],
            'Windows': ['clip']
        }
        return commands.get(system)
    
    def _execute_clipboard_command(
        self, 
        cmd: List[str], 
        content: str, 
        system: str
    ) -> None:
        """Execute clipboard command with content.
        
        Args:
            cmd: Command list
            content: Content to copy
            system: Platform system name
        """
        encoding = 'utf-16' if system == 'Windows' else 'utf-8'
        
        try:
            subprocess.run(cmd, input=content.encode(encoding), check=True)
        except FileNotFoundError:
            # Linux fallback to xsel
            if system == 'Linux':
                subprocess.run(
                    ['xsel', '--clipboard', '--input'], 
                    input=content.encode('utf-8'), 
                    check=True
                )

    def _prepare_patterns(self) -> Tuple[List[str], List[str]]:
        """Prepare include and exclude patterns based on glob mode.
        
        Returns:
            Tuple of (include_patterns, exclude_patterns)
        """
        if not self.strict_glob:
            include_patterns = [self._make_recursive(p) for p in self.include_patterns]
            exclude_patterns = [self._make_recursive(p) for p in self.exclude_patterns]
        else:
            include_patterns = self.include_patterns[:]
            exclude_patterns = self.exclude_patterns[:]
        
        return include_patterns, exclude_patterns

    def _run_async_summarization(self, plan):
        """Run async summarization with progress tracking."""
        print(f"\n{BOLD}{CYAN}=== STARTING SUMMARIZATION ==={RESET}")

        # Inform about line numbers if requested
        if self.line_number_patterns:
            print(f"{YELLOW}Note: Line numbers will only be added to uncompressed code files{RESET}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run async summarization
        processor = AsyncProcessor(
            model=self.llm_model_summarizer,
            max_concurrent=50,
            line_number_patterns=self.line_number_patterns
        )

        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            processor.process_files_with_plan(plan['files'], self.path)
        )
        loop.close()

        return results