"""File discovery module for finding files matching glob patterns."""

import logging
import mimetypes
import re
from pathlib import Path
from typing import List, Tuple, Set, Optional, NamedTuple
from fnmatch import fnmatch


class PatternParts(NamedTuple):
    """Parsed pattern with optional content search."""
    glob: str
    content: Optional[str] = None


class FileDiscovery:
    """Handles file discovery and filtering."""
    
    # Common binary file extensions to skip
    BINARY_EXTENSIONS = frozenset({
        '.pyc', '.pyo', '.so', '.dll', '.dylib', '.exe', '.bin',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.sqlite', '.db', '.pkl', '.pickle', '.npy', '.npz',
        '.woff', '.woff2', '.ttf', '.eot', '.otf',
    })
    
    def _parse_pattern(self, pattern: str) -> PatternParts:
        """Parse a pattern into glob and optional content search parts.
        
        Args:
            pattern: Pattern string, optionally with '::' content separator
            
        Returns:
            PatternParts with glob and optional content search
        """
        if '::' in pattern:
            parts = pattern.split('::', 1)
            if len(parts) == 2:
                return PatternParts(glob=parts[0], content=parts[1])
        return PatternParts(glob=pattern)
    
    def _parse_patterns(self, patterns: List[str]) -> Tuple[List[str], List[PatternParts]]:
        """Separate patterns into simple globs and content-search patterns.
        
        Args:
            patterns: List of pattern strings
            
        Returns:
            Tuple of (simple_globs, content_patterns)
        """
        simple_globs = []
        content_patterns = []
        
        for pattern in patterns:
            parsed = self._parse_pattern(pattern)
            if parsed.content:
                content_patterns.append(parsed)
            else:
                simple_globs.append(parsed.glob)
        
        return simple_globs, content_patterns

    def is_binary_file(self, filepath: Path) -> bool:
        """
        Determine if a file is binary based on extension and mimetype.
        
        Args:
            filepath: Path object pointing to the file to check
            
        Returns:
            True if file appears to be binary, False otherwise
        """
        # Check extension first (faster)
        if filepath.suffix.lower() in self.BINARY_EXTENSIONS:
            return True
        
        # Try mimetype detection
        mime_type, _ = mimetypes.guess_type(str(filepath))
        if mime_type:
            # Text files typically start with 'text/'
            return not mime_type.startswith('text/')
        
        # Default to treating as text if unsure
        return False
    
    def _load_gitignore_patterns(self, base_path: Path) -> List[str]:
        """
        Load patterns from .gitignore file.
        
        Args:
            base_path: Base directory to look for .gitignore
            
        Returns:
            List of gitignore patterns
        """
        gitignore_path = base_path / '.gitignore'
        
        if not gitignore_path.exists():
            return []
        
        try:
            with gitignore_path.open('r') as f:
                return [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]
        except (FileNotFoundError, PermissionError) as e:
            logging.debug(f"Could not read .gitignore: {e}")
            return []
    
    def _is_gitignored(self, filepath: Path, base_path: Path, gitignore_patterns: List[str]) -> bool:
        """
        Check if a file matches any gitignore pattern.
        
        Args:
            filepath: The file to check
            base_path: The base directory (where .gitignore is)
            gitignore_patterns: List of patterns from .gitignore
            
        Returns:
            True if file should be ignored
        """
        try:
            relative_path = filepath.relative_to(base_path)
            relative_str = str(relative_path)
            path_parts = relative_path.parts
            
            for pattern in gitignore_patterns:
                # Handle directory patterns (ending with /)
                if pattern.endswith('/'):
                    pattern_name = pattern[:-1]
                    # Check if any part of the path matches
                    if any(fnmatch(part, pattern_name) for part in path_parts):
                        return True
                # Handle patterns starting with / (root-only match)
                elif pattern.startswith('/'):
                    if fnmatch(relative_str, pattern[1:]):
                        return True
                else:
                    # Match anywhere in path
                    if fnmatch(relative_str, pattern):
                        return True
                    # Check if any parent directory matches
                    if any(fnmatch(part, pattern) for part in path_parts[:-1]):
                        return True
        except ValueError:
            # filepath is not relative to base_path
            pass
        
        return False
    
    def _file_contains_pattern(self, filepath: Path, pattern: str) -> bool:
        """
        Check if a file contains a specific pattern (regex).
        
        Args:
            filepath: Path to the file to search
            pattern: Regular expression pattern to search for
            
        Returns:
            True if pattern is found in file content
        """
        if self.is_binary_file(filepath):
            return False
        
        try:
            # Compile pattern with error handling
            try:
                compiled_pattern = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            except re.error as e:
                # If pattern is invalid regex, try literal search
                logging.debug(f"Invalid regex pattern '{pattern}': {e}. Using literal search.")
                pattern = re.escape(pattern)
                compiled_pattern = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            
            with filepath.open('r', encoding='utf-8') as f:
                # Read in chunks for large files
                chunk_size = 8192
                
                while chunk := f.read(chunk_size):
                    if compiled_pattern.search(chunk):
                        return True
                return False
        except (UnicodeDecodeError, PermissionError) as e:
            logging.debug(f"Could not read {filepath}: {e}")
            return False
    
    def _gather_files_by_glob(self, base: Path, patterns: List[str]) -> Set[Path]:
        """Gather all files matching the given glob patterns.
        
        Args:
            base: Base path to search from
            patterns: List of glob patterns
            
        Returns:
            Set of matching file paths
        """
        files = set()
        
        for pattern in patterns:
            try:
                if base.is_dir():
                    matches = base.glob(pattern)
                    files.update(p for p in matches if p.is_file())
                elif base.is_file() and base.match(pattern):
                    files.add(base)
            except Exception as e:
                logging.debug(f"Error with pattern '{pattern}': {e}")
        
        return files
    
    def _gather_files_by_content(self, base: Path, patterns: List[PatternParts]) -> Set[Path]:
        """Gather files matching glob patterns that contain specific content.
        
        Args:
            base: Base path to search from
            patterns: List of PatternParts with glob and content
            
        Returns:
            Set of matching file paths
        """
        files = set()
        
        for pattern_parts in patterns:
            glob_files = self._gather_files_by_glob(base, [pattern_parts.glob])
            
            for filepath in glob_files:
                if self._file_contains_pattern(filepath, pattern_parts.content):
                    files.add(filepath)
        
        return files
    
    def _filter_gitignored(self, files: Set[Path], base: Path, patterns: List[str]) -> Set[Path]:
        """Filter out gitignored files.
        
        Args:
            files: Set of file paths to filter
            base: Base directory
            patterns: Gitignore patterns
            
        Returns:
            Filtered set of files
        """
        if not patterns:
            return files
        
        filtered = {f for f in files if not self._is_gitignored(f, base, patterns)}
        
        ignored_count = len(files) - len(filtered)
        if ignored_count > 0:
            logging.info(f"Ignored {ignored_count} files based on .gitignore patterns")
        
        return filtered
    
    def _validate_files(self, files: Set[Path]) -> List[Path]:
        """Validate and filter files for readability and content.
        
        Args:
            files: Set of file paths to validate
            
        Returns:
            Sorted list of valid file paths
        """
        valid_files = []
        
        for path in files:
            # Skip binary files
            if self.is_binary_file(path):
                logging.debug(f"Skipping binary file: {path}")
                continue
            
            try:
                # Check if file is empty
                if path.stat().st_size == 0:
                    logging.debug(f"Skipping empty file: {path}")
                    continue
                
                # Verify readability
                with path.open('r', encoding='utf-8') as f:
                    # Just try to read first byte to verify
                    f.read(1)
                
                valid_files.append(path)
            except (PermissionError, UnicodeDecodeError) as e:
                logging.debug(f"Skipping unreadable file {path}: {e}")
        
        return sorted(valid_files)

    def find_files(self, base_path: str, include_patterns: List[str], exclude_patterns: Optional[List[str]] = None, respect_gitignore: bool = True) -> Tuple[List[Path], List[str]]:
        """
        Discover all files matching include patterns but not exclude patterns.
        Supports special "contains::pattern" syntax for content-based searching.
        
        Args:
            base_path: Base directory to search from
            include_patterns: List of glob patterns or "contains::pattern" to include
            exclude_patterns: List of glob patterns or "contains::pattern" to exclude
            respect_gitignore: Whether to respect .gitignore patterns (default: True)
            
        Returns:
            Tuple of (found_files, errors) where found_files is a list of Path objects
            and errors is a list of error messages encountered
        """
        errors: List[str] = []
        exclude_patterns = exclude_patterns or []
        
        base = Path(base_path).resolve()
        if not base.exists():
            errors.append(f"Path does not exist: {base_path}")
            return [], errors
        
        try:
            # Parse patterns
            include_globs, include_content = self._parse_patterns(include_patterns)
            exclude_globs, exclude_content = self._parse_patterns(exclude_patterns)
            
            # Gather files by pattern type
            included_files = self._gather_files_by_glob(base, include_globs)
            included_files.update(self._gather_files_by_content(base, include_content))
            
            # Handle exclusions
            excluded_files = self._gather_files_by_glob(base, exclude_globs)
            
            # For content-based exclusions, only exclude from included files
            if exclude_content:
                content_excluded = set()
                for pattern in exclude_content:
                    # Only check files that are already included
                    candidates = included_files & self._gather_files_by_glob(base, [pattern.glob])
                    for filepath in candidates:
                        if self._file_contains_pattern(filepath, pattern.content):
                            content_excluded.add(filepath)
                excluded_files.update(content_excluded)
            
            # Apply exclusions
            final_files = included_files - excluded_files
            
            # Filter gitignored files if requested
            if respect_gitignore and base.is_dir():
                gitignore_patterns = self._load_gitignore_patterns(base)
                final_files = self._filter_gitignored(final_files, base, gitignore_patterns)
            
            # Validate and sort files
            return self._validate_files(final_files), errors
            
        except Exception as e:
            errors.append(f"Unexpected error during file discovery: {str(e)}")
            return [], errors
