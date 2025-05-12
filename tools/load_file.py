"""
Helper function for loading files in Qwen CLI.
This function will be dynamically loaded by the CLI.
"""

import logging
from pathlib import Path

logger = logging.getLogger("qwen_cli.helpers.load_file")

def handle_load_file(filepath):
    """
    Whenever the user asks to read, load, analyze some code and provides a relative path, such as ./file.py or utils/utils.py or similar,
    write the command `[LOAD_FILE filepath]` (e.g., `[LOAD_FILE ./my_script.py]`).

    This command will load the file and return its content formatted for the model context.
    """
    logger.info(f"Loading file: {filepath}")
    
    try:
        file_path = Path(filepath)
        
        logger.debug(f"Initial filepath: {filepath}, is_absolute: {file_path.is_absolute()}")
        
        if not file_path.is_absolute():
            app_path = Path("/my-projects")
            possible_path = app_path / file_path
            logger.debug(f"Trying /project-relative path: {possible_path}")
            if possible_path.exists():
                file_path = possible_path
                logger.debug(f"  Found: Using /project-relative path")
            else:
                cwd_path = Path.cwd()
                possible_path = cwd_path / file_path
                logger.debug(f"Trying CWD-relative path: {possible_path}")
                if possible_path.exists():
                    file_path = possible_path
                    logger.debug(f"  Found: Using CWD-relative path")
                else:
                    logger.debug("  Not found in /project or CWD")
        
        logger.debug(f"Resolved file_path: {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path} (original: {filepath})")
            return None
            
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path} (original: {filepath})")
            return None
            
        content = file_path.read_text(encoding='utf-8', errors='replace')
        
        language = get_language_from_extension(file_path)
        
        formatted_content = f"[file: {file_path}]\\n```{language}\\n{content}\\n```"
        
        logger.info(f"Successfully loaded file: {file_path} ({len(content)} characters)")
        return formatted_content
        
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {e}")
        return None
    
def get_language_from_extension(file_path: Path) -> str:
    """Determine language for markdown code block based on file extension."""
    ext = file_path.suffix.lower()
    lang_map = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.html': 'html', '.css': 'css', '.java': 'java', '.c': 'c',
        '.cpp': 'cpp', '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
        '.php': 'php', '.sh': 'bash', '.md': 'markdown', '.json': 'json',
        '.yaml': 'yaml', '.yml': 'yaml', '.xml': 'xml', '.sql': 'sql',
        '.dockerfile': 'dockerfile', '.tf': 'terraform', '.hcl': 'terraform',
        '.jsx': 'jsx', '.tsx': 'tsx'
    }
    return lang_map.get(ext, 'text')