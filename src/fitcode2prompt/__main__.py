"""Entry point for Summarizely CLI."""

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional

# Suppress runtime warnings before any imports
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"
os.environ["LITELLM_LOG"] = "ERROR"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="litellm")
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")


class BlankLinesHelpFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter that adds blank lines between options."""

    def _split_lines(self, text: str, width: int) -> List[str]:
        lines = super()._split_lines(text, width)
        return lines + [''] if lines else lines


class CompressionLevel:
    """Compression level constants and mappings."""

    # Internal values represent what percentage remains after compression
    NONE = 100   # 0% compression = 100% remains
    TRIM = 95    # 5% compression = 95% remains
    LIGHT = 85   # 15% compression = 85% remains
    MEDIUM = 50  # 50% compression = 50% remains
    HEAVY = 10   # 90% compression = 10% remains
    MAX = 0      # 100% compression = 0% remains

    # Map user-friendly names and percentages to internal values
    # Numbers represent compression percentage (how much is removed)
    ALIASES = {
        # No compression (0% compression, 100% remains)
        'none': NONE, '0': NONE,
        # Trim compression (5% compression, 95% remains)
        'trim': TRIM, '5': TRIM,
        # Light compression (15% compression, 85% remains)
        'light': LIGHT, '15': LIGHT,
        # Medium compression (50% compression, 50% remains)
        'medium': MEDIUM, '50': MEDIUM,
        # Heavy compression (90% compression, 10% remains)
        'heavy': HEAVY, '90': HEAVY,
        # Maximum compression (100% compression, 0% remains)
        'max': MAX, 'full': MAX, '100': MAX
    }

    DESCRIPTIONS = {
        NONE: "No compression (0%), file unchanged",
        TRIM: "Trim compression (5%), remove imports and whitespace",
        LIGHT: "Light compression (15%), remove redundant comments",
        MEDIUM: "Medium compression (50%), replace simple functions with descriptions",
        HEAVY: "Heavy compression (90%), skeleton only with signatures",
        MAX: "Maximum compression (100%), one to three sentence summary"
    }


def parse_compression_level(value: str) -> int:
    """Parse compression level from string to internal value."""
    if value in CompressionLevel.ALIASES:
        return CompressionLevel.ALIASES[value]

    valid_options = ', '.join(sorted(CompressionLevel.ALIASES.keys()))
    raise argparse.ArgumentTypeError(
        f"Invalid compression level: '{value}'. Valid options: {valid_options}"
    )


def parse_patterns(pattern_string: str) -> List[str]:
    """Parse comma-separated patterns into a list."""
    if not pattern_string:
        return []
    return [p.strip() for p in pattern_string.split(',') if p.strip()]


def parse_budget(budget_string: Optional[str]) -> Optional[int]:
    """Parse budget string, extracting only numeric characters."""
    if not budget_string:
        return None

    digits = ''.join(c for c in budget_string if c.isdigit())
    if not digits:
        raise ValueError("Budget must contain at least one digit")

    return int(digits)


def build_compression_config(args: argparse.Namespace) -> Dict[str, int]:
    """Build compression configuration from command-line arguments."""
    config = {}

    # Map argument names to compression levels
    # Numbers represent compression percentage (how much is removed)
    compression_args = [
        (args.no_compression, CompressionLevel.NONE),        # --compression-0: 0% compression (no compression)
        (args.compression_5, CompressionLevel.TRIM),         # --compression-5: ~5% compression (trim)
        (args.compression_15, CompressionLevel.LIGHT),       # --compression-15: ~15% compression (light)
        (args.compression_50, CompressionLevel.MEDIUM),      # --compression-50: ~50% compression (medium)
        (args.compression_90, CompressionLevel.HEAVY),       # --compression-90: ~90% compression (heavy)
        (args.compression_100, CompressionLevel.MAX),        # --compression-100: ~100% compression (maximum)
    ]

    for arg_value, level in compression_args:
        for pattern in parse_patterns(arg_value):
            config[pattern] = level

    return config


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='summarize',
        description='Intelligently compress codebases into LLM-ready summaries',
        formatter_class=BlankLinesHelpFormatter
    )

    # Positional arguments
    parser.add_argument('path', help='Path to search (directory or file)')

    # Output options
    parser.add_argument(
        '-o', '--output-dir',
        default='./',
        help='Output directory for results'
    )

    # File selection
    parser.add_argument(
        '-i', '--include',
        default='**/*',
        help='Comma-separated glob patterns to include.\n'
             'Supports content search: "*.py::TODO" finds Python files containing TODO'
    )

    parser.add_argument(
        '-e', '--exclude',
        default='',
        help='Comma-separated glob patterns to exclude.\n'
             'Supports content search: "*.js::console.log" excludes JS files with console.log'
    )

    # Budget and planning
    parser.add_argument(
        '-b', '--budget',
        help='Maximum token budget (triggers AI planner for optimal compression)'
    )

    parser.add_argument(
        '--buffer-percent',
        type=int,
        default=10,
        help='Buffer percentage for planner budget (default: 10%%)'
    )

    # Model configuration
    parser.add_argument(
        '--planner',
        default='o3-mini',
        help='LLM model for compression planning (default: o3-mini)'
    )

    parser.add_argument(
        '--summarizer',
        default='gpt-4.1-nano',
        help='LLM model for summarization (default: gpt-4.1-nano)'
    )

    parser.add_argument(
        '-m', '--encoding-model',
        default='cl100k_base',
        dest='model',
        help='Tiktoken encoding for token counting (default: cl100k_base)'
    )

    # Compression settings
    compression_help = '\n'.join(
        f'"{alias}" - {CompressionLevel.DESCRIPTIONS[level]}'
        for alias, level in [
            ('0/none', CompressionLevel.NONE),
            ('5/trim', CompressionLevel.TRIM),
            ('15/light', CompressionLevel.LIGHT),
            ('50/medium', CompressionLevel.MEDIUM),
            ('90/heavy', CompressionLevel.HEAVY),
            ('100/max', CompressionLevel.MAX)
        ]
    )

    parser.add_argument(
        '--default-compression',
        type=parse_compression_level,
        default=CompressionLevel.TRIM,
        metavar='LEVEL',
        help=f'Default compression level:\n{compression_help}'
    )

    # Per-pattern compression levels
    compression_levels = [
        (0, 'no-compression', 'no compression (0%), preserve unchanged'),
        (5, 'compression-trim', 'trim compression (5%), remove imports/whitespace'),
        (15, 'compression-light', 'light compression (15%), remove redundant comments'),
        (50, 'compression-medium', 'medium compression (50%), simplify functions'),
        (90, 'compression-heavy', 'heavy compression (90%), skeleton only'),
        (100, 'compression-max', 'maximum compression (100%), 1-3 sentences'),
    ]

    for level, name, desc in compression_levels:
        dest = name.replace('-', '_')
        if level == 0:
            parser.add_argument(
                f'--compression-{level}', f'--{name}',
                default='',
                metavar='GLOBS',
                dest=dest,
                help=f'Comma-separated globs for files to {desc}'
            )
        else:
            parser.add_argument(
                f'--compression-{level}',
                default='',
                metavar='GLOBS',
                dest=f'compression_{level}',
                help=f'Comma-separated globs for files to {desc}'
            )

    # Additional options
    parser.add_argument(
        '--line-numbers',
        default='',
        metavar='GLOBS',
        help='Comma-separated globs for files to add line numbers (uncompressed only)'
    )

    # Flags
    parser.add_argument(
        '--strict-glob',
        action='store_true',
        help='Use standard glob patterns without recursive subfolder inclusion'
    )

    parser.add_argument(
        '--no-ignore',
        action='store_true',
        help='Do not respect .gitignore patterns'
    )

    parser.add_argument(
        '--no-clipboard',
        action='store_true',
        help='Do not copy output to clipboard'
    )

    parser.add_argument(
        '--count-only',
        action='store_true',
        help='Only count tokens without summarizing'
    )

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Parse patterns
    include_patterns = parse_patterns(args.include)
    exclude_patterns = parse_patterns(args.exclude)
    line_number_patterns = parse_patterns(args.line_numbers)

    # Parse budget
    try:
        budget = parse_budget(args.budget)
    except ValueError as e:
        parser.error(str(e))
        return 1

    # Build compression configuration
    compression_config = build_compression_config(args)

    # Import Summarizer only when needed (after argument parsing)
    from .summarizer import Summarizer

    # Create and configure summarizer
    summarizer = Summarizer(
        path=args.path,
        llm_model_planner=args.planner,
        llm_model_summarizer=args.summarizer,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        strict_glob=args.strict_glob,
        budget=budget,
        model=args.model,
        output_dir=args.output_dir,
        verbose=False,
        use_planner=(budget is not None),
        buffer_percent=args.buffer_percent,
        default_compression=args.default_compression,
        respect_gitignore=(not args.no_ignore),
        compression_config=compression_config,
        line_number_patterns=line_number_patterns,
        no_clipboard=args.no_clipboard
    )

    # Execute the appropriate action
    if args.count_only:
        return summarizer.count_tokens()
    else:
        return summarizer.run()


if __name__ == "__main__":
    sys.exit(main())