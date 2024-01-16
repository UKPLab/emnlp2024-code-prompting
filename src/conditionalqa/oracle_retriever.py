def get_summarized_doc(x, url2doc):
    """
    Oracle retriever for the conditionalQA dataset.
    Returns the contextualized rationales for the given example.
    Contextualized rationales are defined as the sections that contain the rationales.
    """
    doc = url2doc[x["url"]]
    list_sections = get_sections(doc["contents"])
    summarized_doc = create_contextualized_rationales(list_sections, x["evidences"])
    return summarized_doc


def get_sections(doc):
    """
    This function takes in a document as input and returns a list of sections.
    A section is defined as a list of tags that are enclosed by a header tag (h1, h2, h3, or h4).
    """
    list_sections = []
    section = []
    for tag in doc:
        if "<h1>" in tag or "<h2>" in tag or "<h3>" in tag or "<h4>" in tag:
            if len(section) > 0:
                list_sections.append(section)
            section = []

        section.append(tag)
    if len(section) > 0:
        list_sections.append(section)
    return list_sections


def create_contextualized_rationales(list_sections, list_rationales):
    """
    This function takes in two lists: list_sections and list_rationales.
    It returns a string that contains the contextualized rationales.
    The function first adds the first section of list_sections to the output list.
    Then, for each section in list_sections, it checks if any of the rationales in list_rationales
    are present in the section. If so, it adds the section to the output list and moves on to the next section.
    The output list is then flattened and joined by newline characters to create the final output string.
    """
    contextualized_rationales = []
    # always add the first section, which is usually an overview
    contextualized_rationales.append(list_sections[0])
    for section in list_sections[1:]:
        for rationale in list_rationales:
            if rationale in section:
                contextualized_rationales.append(section)
                break
    # flatten the list
    contextualized_rationales = [
        item for sublist in contextualized_rationales for item in sublist
    ]
    # join by \n
    contextualized_rationales = "\n".join(contextualized_rationales)
    return contextualized_rationales
