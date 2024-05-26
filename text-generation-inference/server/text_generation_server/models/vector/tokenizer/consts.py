# handy mapping from language names to internal tags
LANG2TAG = {
    "text": "<en_XX>",
    "python": "<prog_PY>",
    "java": "<prog_JAVA>",
    "javascript": "<prog_JS>",
    "typescript": "<prog_TS>",
    "csharp": "<prog_CS>",
    "c": "<prog_C>",
    "cpp": "<prog_CPP>",
    "php": "<prog_PHP>",
    "go": "<prog_GO>",
    "ruby": "<prog_RB>",
    "shell": "<prog_SH>",
    "rust": "<prog_RS>",
    "kotlin": "<prog_KT>",
    "scala": "<prog_SC>",
}

FAIRSEQ_LANGUAGE_CODES = list(LANG2TAG.values())

"""
whitespace = [
    '\t', '\t\t', '\t' * 3, '\t' * 4, '\t' * 5,
    '\n', '\n\n',
    '\r\n', '\r\n\r\n',
    ' ' * 3, ' ' * 5, ' ' * 7, ' ' * 9, ' ' * 11, ' ' * 13, ' ' * 15,
    ' ' * 2, ' ' * 4, ' ' * 6, ' ' * 8, ' ' * 10, ' ' * 12, ' ' * 14, ' ' * 16
]
"""
USER_DEFINED_SYMBOLS = [
    "ĉ",
    "ĉĉ",
    "ĉĉĉ",
    "ĉĉĉĉ",
    "ĉĉĉĉĉ",
    "Ċ",
    "ĊĊ",
    "čĊ",
    "čĊčĊ",
    "ĠĠĠ",
    "ĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠ",
    "ĠĠĠĠ",
    "ĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠ",
    "ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ",
]


# USER_DEFINED_SYMBOLS = [
#     "<NEW_LINE>",
#     "<INDENT>",
#     "<DEDENT>",
#     "<DENOISE>",
#     "<GENERATE>",
#     "<COMPLETE>",
#     "<AST>",
#     "<DATAFLOW>",
# ]

LOOKAHEAD_SYMBOLS = [
    '<end_of_insertion_gXaC7CZD7B>',
    '<code_missing_here_qoSa3n5fmW>',
    '<end_of_right_context_7CyOC9jrBA>',
    '<start_right_context_hPx8WhSpdj>',
]
