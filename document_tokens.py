def document_tokens(document_text):
    """
    Parameters
    ----------
    document_text : TYPE
        DESCRIPTION.

    Returns document_tokens
    -------
    None.

    """

    document_textsplit = document_text.split(" ")
    document_tokens = []
    for token in document_textsplit:
        document_tokens.append({"token":token, "start_byte":-1, "end_byte":-1, "html_token":"<" in token})

