"""Dependency-free source guard for the native atomic-polarizability call path."""

import re


_PROCESS_CALL = re.compile(
    r"(?<![\w])(?:std\s*::\s*)?"
    r"(?:system|_?popen|v?fork|exec(?:l|le|lp|lpe|v|ve|vp|vpe)|posix_spawnp?|"
    r"CreateProcess[AW]?|ShellExecute[AW]?|WinExec)\s*\("
)
_EXTERNAL_TERM = re.compile(r"(?i)(?<![\w])(?:\.camcasp-reference|camcasp|orient|pfit|casimir)(?![\w])")
_INCLUDE_LINE = re.compile(r"^\s*#\s*include\b.*$", re.MULTILINE)


def _without_cpp_comments(text):
    """Remove C++ comments while preserving literals and line boundaries."""
    output = []
    index = 0
    mode = "code"

    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if mode == "code":
            if char == "/" and following == "/":
                mode = "line-comment"
                index += 2
                continue
            if char == "/" and following == "*":
                mode = "block-comment"
                index += 2
                continue
            output.append(char)
            if char == '"':
                mode = "string"
            elif char == "'":
                mode = "character"
            index += 1
            continue

        if mode == "line-comment":
            if char == "\n":
                output.append(char)
                mode = "code"
            index += 1
            continue

        if mode == "block-comment":
            if char == "*" and following == "/":
                mode = "code"
                index += 2
            else:
                if char == "\n":
                    output.append(char)
                index += 1
            continue

        output.append(char)
        if char == "\\" and following:
            output.append(following)
            index += 2
            continue
        if (mode == "string" and char == '"') or (mode == "character" and char == "'"):
            mode = "code"
        index += 1

    return "".join(output)


def source_violations(text):
    """Return forbidden native launch calls and external/reference terms in active C++ text."""
    active_text = _INCLUDE_LINE.sub("", _without_cpp_comments(text))
    violations = []

    for match in _PROCESS_CALL.finditer(active_text):
        call = re.sub(r"\s+", "", match.group())
        violations.append(f"forbidden process API: {call}")
    for match in _EXTERNAL_TERM.finditer(active_text):
        violations.append(f"forbidden external term: {match.group().lower()}")

    return violations
