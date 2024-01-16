import random
from langchain.prompts.chat import ChatPromptTemplate


class Doc2Code:
    """
    A class that transforms a document into pseudo-code using a language model.

    Args:
        llm: A language model used to generate the pseudo-code.
        doc2code_examples: A list of examples containing documents and their corresponding pseudo-code.

    Attributes:
        llm: A language model used to generate the pseudo-code.
        doc2code_examples: A list of examples containing documents and their corresponding pseudo-code.
        doc2code_chain: A chain of components used to transform a document into pseudo-code.

    Methods:
        __call__(self, doc): Transforms the given document into pseudo-code using the doc2code_chain.
    """

    def __init__(self, llm, doc2code_examples, num_examples=None, seed=0):
        random.seed(seed)
        self.llm = llm
        self.doc2code_examples = doc2code_examples
        self.num_examples = num_examples if num_examples else len(doc2code_examples)
        self.doc2code_chain = self._create_chain()

    def __call__(self, doc):
        return self.doc2code_chain.invoke({"doc": doc}).content

    def _create_chain(self):
        task_description = 'You are a helpful assistant that transforms a document into pseudo-code. You have to create variables that model the the document. If the document has conditional statements like "if ..." you should model that in code too with a conditional clause. I\'ll give you some examples first.'
        list_icl_chat_examples = [("system", task_description)]
        demonstrations = random.sample(self.doc2code_examples, self.num_examples)
        for doc2code in demonstrations:
            list_icl_chat_examples.append(("human", doc2code["document"]))
            list_icl_chat_examples.append(("assistant", doc2code["code"]))
        list_icl_chat_examples.append(
            (
                "human",
                "{doc}. Remember to generate if conditions for every conditional statement in the document (for example, a sentence saying 'if').",
            )
        )
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        self.prompt = chat_prompt
        doc2code_chain = chat_prompt | self.llm
        return doc2code_chain


class Question2Code:
    """
    A class that transforms a question into pseudo-code using a given LLM and a set of question-to-code examples.

    Attributes:
    - llm (LanguageModel): The language model used to generate the pseudo-code.
    - question2code_examples (list): A list of dictionaries containing question-to-code examples, where each dictionary
      has two keys: "question" (string) and "code" (string).
    - question2code_chain (Chain): The chain of operations used to transform a question into pseudo-code.

    Methods:
    - __call__(self, question): Transforms the given question into pseudo-code using the question2code_chain.
    """

    def __init__(self, llm, question2code_examples, num_examples=None, seed=0):
        random.seed(seed)
        self.llm = llm
        self.question2code_examples = question2code_examples
        self.num_examples = (
            num_examples if num_examples else len(question2code_examples)
        )
        self.question2code_chain = self._create_chain()

    def __call__(self, question):
        return self.question2code_chain.invoke({"question": question}).content

    def _create_chain(self):
        task_description = "You are a helpful assistant that transforms a question into pseudo-code. You have to create variables that model the question. I'll give you some examples first."
        list_icl_chat_examples = [("system", task_description)]
        demonstrations = random.sample(self.question2code_examples, self.num_examples)
        for question2code in demonstrations:
            list_icl_chat_examples.append(
                ("human", "Question: " + question2code["question"])
            )
            list_icl_chat_examples.append(("assistant", question2code["code"]))
        list_icl_chat_examples.append(
            (
                "human",
                "Question: {question}",
            )
        )
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        question2code_chain = chat_prompt | self.llm
        return question2code_chain


class Code2NL:
    """
    A class that transforms code into natural language.

    Args:
        llm: The language model used for code-to-natural language transformation.
        code2nl_examples: A list of code-to-natural language examples.
        num_examples: The number of examples to use for creating the transformation chain.

    Attributes:
        llm: The language model used for code-to-natural language transformation.
        code2nl_examples: A list of code-to-natural language examples.
        num_examples: The number of examples to use for creating the transformation chain.
        code2nl_chain: The transformation chain for code-to-natural language conversion.

    Methods:
        __call__(self, doc): Transforms the given code into natural language.
        _create_chain(self): Creates the transformation chain for code-to-natural language conversion.
    """

    def __init__(
        self, llm, code2nl_examples, use_key_concepts=False, num_examples=None
    ):
        self.llm = llm
        self.code2nl_examples = code2nl_examples
        self.use_key_concepts = use_key_concepts
        self.num_examples = num_examples if num_examples else len(code2nl_examples)
        self.code2nl_chain = self._create_chain()

    def __call__(self, doc):
        """
        Transforms the given code into natural language.

        Args:
            doc: The code to be transformed.

        Returns:
            str: The natural language representation of the code.
        """
        return self.code2nl_chain.invoke({"doc": doc}).content

    def _create_chain(self):
        """
        Creates the transformation chain for code-to-natural language conversion.

        Returns:
            Chain: The transformation chain for code-to-natural language conversion.
        """
        task_description = "You are a helpful assistant that transforms a code into natural language. Your main task is to transform the `if` statements into natural language so that the reasoning behind the code is preserved in the natural language text. I'll give you some examples first."
        list_icl_chat_examples = [("system", task_description)]
        for code2nl in self.code2nl_examples[: self.num_examples]:
            list_icl_chat_examples.append(("human", code2nl["doc_code"]))
            if self.use_key_concepts:
                list_icl_chat_examples.append(("assistant", code2nl["nl_code_var"]))
            else:
                list_icl_chat_examples.append(("assistant", code2nl["nl_code"]))
        list_icl_chat_examples.append(
            (
                "human",
                "{doc}. Remember to transform every if condition into natural language.",
            )
        )
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        self.prompt = chat_prompt
        code2nl_chain = chat_prompt | self.llm
        return code2nl_chain
