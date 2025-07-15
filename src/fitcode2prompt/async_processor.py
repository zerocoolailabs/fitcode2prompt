"""Async processor for parallel summarization with litellm."""

import os
import warnings

# Suppress litellm warnings  
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="litellm")
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

import asyncio
import logging
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Set

from litellm import acompletion

from .shrink_prompts import CODE_PROMPTS, DOC_PROMPTS
from .tokenizer import Tokenizer

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# File extension categories
DOC_EXTENSIONS: Set[str] = {
    '.md', '.markdown', '.rst', '.txt', 
    '.adoc', '.asciidoc', '.org', '.pod'
}

# Tier display names
TIER_NAMES: Dict[int, str] = {
    100: "none", 95: "trim", 85: "light", 
    50: "medium", 10: "heavy", 0: "max"
}



class AsyncProcessor:
    """Process files concurrently with rate limit handling."""
    
    def __init__(
        self, 
        model: str, 
        max_concurrent: int = 50, 
        line_number_patterns: List[str] = None
    ) -> None:
        """
        Initialize async processor.
        
        Args:
            model: LLM model to use (required)
            max_concurrent: Maximum concurrent requests (default 50)
            line_number_patterns: Patterns for files to add line numbers (only for uncompressed files)
        """
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.line_number_patterns = line_number_patterns or []
        self.tokenizer = Tokenizer(model)
    
    def _should_add_line_numbers(self, filepath: str, base_path: str) -> bool:
        """Check if a file should have line numbers added (only for uncompressed files).
        
        Args:
            filepath: Path to the file
            base_path: Base path for relative path calculation
            
        Returns:
            True if line numbers should be added
        """
        if not self.line_number_patterns:
            return False
        
        rel_path, basename = self._get_relative_path_parts(filepath, base_path)
        
        return any(
            fnmatch(rel_path, pattern) or fnmatch(basename, pattern)
            for pattern in self.line_number_patterns
        )
    
    def _get_relative_path_parts(self, filepath: str, base_path: str) -> tuple[str, str]:
        """Get relative path and basename for a file.
        
        Args:
            filepath: Path to the file
            base_path: Base path for relative path calculation
            
        Returns:
            Tuple of (relative_path, basename)
        """
        try:
            base = Path(base_path).resolve()
            if base.is_file():
                base = base.parent
            file_path = Path(filepath).resolve()
            rel_path = str(file_path.relative_to(base))
        except ValueError:
            rel_path = filepath
        
        return rel_path, Path(filepath).name
    
    def _add_line_numbers(self, content: str) -> str:
        """Add line numbers to content.
        
        Args:
            content: File content to add line numbers to
            
        Returns:
            Content with line numbers added
        """
        lines = content.split('\n')
        max_line_num = len(lines)
        line_num_width = len(str(max_line_num))
        
        return '\n'.join(
            f"{i:>{line_num_width}}│ {line}"
            for i, line in enumerate(lines, 1)
        )
        
    async def process_files_with_plan(
        self, 
        files_plan: List[Dict[str, any]],
        base_path: str
    ) -> List[Dict[str, any]]:
        """
        Process files according to the compression plan.
        
        Args:
            files_plan: List of file plans from planner
            base_path: Base path for reading files
            
        Returns:
            List of results with summaries
        """
        start_time = time.time()
        
        # Create all tasks at once
        tasks = [
            asyncio.create_task(self._process_single_file(file_info, base_path))
            for file_info in files_plan
        ]
        
        print(f"\n🚀 Started {len(tasks)} async compression tasks (model: {self.model})...")
        
        # Process results as they complete
        results = []
        completed = 0
        
        async for task in self._process_tasks_with_progress(tasks, start_time):
            result = await task
            completed += 1
            self._print_progress(result, completed, len(tasks), time.time() - start_time)
            results.append(result)
        
        return results
    
    async def _process_tasks_with_progress(
        self, 
        tasks: List[asyncio.Task], 
        start_time: float
    ):
        """Yield tasks as they complete.
        
        Args:
            tasks: List of tasks to process
            start_time: Start time for elapsed calculation
            
        Yields:
            Completed tasks
        """
        for task in asyncio.as_completed(tasks):
            yield task
    
    def _print_progress(
        self, 
        result: Dict[str, any], 
        completed: int, 
        total: int, 
        elapsed: float
    ) -> None:
        """Print progress for a completed task.
        
        Args:
            result: Task result
            completed: Number of completed tasks
            total: Total number of tasks
            elapsed: Elapsed time in seconds
        """
        rel_path = Path(result['path']).name
        tier = result.get('tier', 100)
        tier_name = TIER_NAMES.get(tier, f"{tier}%")
        
        if result['success']:
            # Success - show in green
            status = f"\033[92m✓ [{completed}/{total}] {rel_path} [{tier_name}] ({elapsed:.1f}s)\033[0m"
        else:
            # Failure - show in red
            status = f"\033[91m✗ [{completed}/{total}] {rel_path} [{tier_name}] ({elapsed:.1f}s) - FAILED\033[0m"
        
        print(status)
    
    async def _process_single_file(
        self, 
        file_info: Dict[str, any],
        base_path: str
    ) -> Dict[str, any]:
        """Process a single file according to its compression tier.
        
        Args:
            file_info: File information including path, tier, and tokens
            base_path: Base path for relative path calculation
            
        Returns:
            Processing result dictionary
        """
        filepath = file_info['path']
        tier = file_info['tier']
        original_tokens = file_info['original_tokens']
        
        try:
            content = self._read_file_content(filepath)
            
            # Handle uncompressed files (tier 100)
            if tier == 100:
                return self._handle_uncompressed_file(
                    filepath, content, tier, original_tokens, base_path
                )
            
            # Skip compression if already at minimum size
            if original_tokens <= 100:
                return self._handle_minimal_file(
                    filepath, content, tier, original_tokens
                )
            
            # Perform compression
            return await self._compress_file(
                filepath, content, tier, original_tokens
            )
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return self._create_error_result(
                filepath, tier, original_tokens, str(e)
            )
    
    def _read_file_content(self, filepath: str) -> str:
        """Read file content.
        
        Args:
            filepath: Path to the file
            
        Returns:
            File content as string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _handle_uncompressed_file(
        self, 
        filepath: str, 
        content: str, 
        tier: int, 
        original_tokens: int,
        base_path: str
    ) -> Dict[str, any]:
        """Handle files that don't need compression.
        
        Args:
            filepath: Path to the file
            content: File content
            tier: Compression tier (100)
            original_tokens: Original token count
            base_path: Base path for relative path calculation
            
        Returns:
            Result dictionary
        """
        file_ext = Path(filepath).suffix.lower()
        
        # Add line numbers for uncompressed code files if requested
        if file_ext not in DOC_EXTENSIONS and self._should_add_line_numbers(filepath, base_path):
            content = self._add_line_numbers(content)
        
        return self._create_success_result(
            filepath, tier, content, original_tokens, original_tokens
        )
    
    def _handle_minimal_file(
        self, 
        filepath: str, 
        content: str, 
        tier: int, 
        original_tokens: int
    ) -> Dict[str, any]:
        """Handle files already at minimum size.
        
        Args:
            filepath: Path to the file
            content: File content
            tier: Compression tier
            original_tokens: Original token count
            
        Returns:
            Result dictionary
        """
        logger.info(
            f"{filepath}: Skipping compression - already at minimum size "
            f"({original_tokens} tokens)"
        )
        return self._create_success_result(
            filepath, tier, content, original_tokens, original_tokens
        )
    
    async def _compress_file(
        self, 
        filepath: str, 
        content: str, 
        tier: int, 
        original_tokens: int
    ) -> Dict[str, any]:
        """Compress file content using LLM.
        
        Args:
            filepath: Path to the file
            content: File content
            tier: Compression tier
            original_tokens: Original token count
            
        Returns:
            Result dictionary
        """
        prompt = self._get_compression_prompt(filepath, content, tier)
        
        async with self.semaphore:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                num_retries=5,
                request_timeout=300,
            )
            
            summary = response.choices[0].message.content
            compressed_tokens = self.tokenizer.count_tokens(summary)
            
            return self._create_success_result(
                filepath, tier, summary, original_tokens, compressed_tokens
            )
    
    def _get_compression_prompt(self, filepath: str, content: str, tier: int) -> str:
        """Get appropriate compression prompt for file.
        
        Args:
            filepath: Path to the file
            content: File content
            tier: Compression tier
            
        Returns:
            Formatted prompt string
        """
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext in DOC_EXTENSIONS:
            prompt_template = DOC_PROMPTS[tier]
        else:
            prompt_template = CODE_PROMPTS[tier]
        
        return prompt_template.replace("{code}", content)
    
    def _create_success_result(
        self, 
        filepath: str, 
        tier: int, 
        content: str, 
        original_tokens: int, 
        compressed_tokens: int
    ) -> Dict[str, any]:
        """Create success result dictionary.
        
        Args:
            filepath: Path to the file
            tier: Compression tier
            content: Processed content
            original_tokens: Original token count
            compressed_tokens: Compressed token count
            
        Returns:
            Result dictionary
        """
        return {
            'path': filepath,
            'tier': tier,
            'summary': content,
            'success': True,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens
        }
    
    def _create_error_result(
        self, 
        filepath: str, 
        tier: int, 
        original_tokens: int, 
        error: str
    ) -> Dict[str, any]:
        """Create error result dictionary.
        
        Args:
            filepath: Path to the file
            tier: Compression tier
            original_tokens: Original token count
            error: Error message
            
        Returns:
            Result dictionary
        """
        return {
            'path': filepath,
            'tier': tier,
            'error': error,
            'success': False,
            'original_tokens': original_tokens,
            'compressed_tokens': 0
        }
    
    
