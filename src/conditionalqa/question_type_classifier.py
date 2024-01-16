from langchain.prompts.chat import ChatPromptTemplate


class QuestionType:
    """
    A class representing the different types of questions that can be asked.

    Attributes:
    -----------
    YESNO : str
        Represents a Yes/No question type.
    SPAN : str
        Represents a Span question type.
    """

    YESNO = "Yes/No"
    SPAN = "Span"


class QuestionTypeClassifier:
    """
    A class for classifying the type of a given question.

    Attributes:
        llm (LanguageModel): The language model to use for classification.
        examples (list): A list of example questions and their corresponding types.
        task_description (str): A description of the task to be performed.
        qtype_chain (Chain): A chain of ICL and LLM models to classify the question type.
    """

    def __init__(self, llm):
        """
        Initializes a new instance of the QuestionTypeClassifier class.

        Args:
            llm (LanguageModel): The language model to use for classification.
        """
        self.llm = llm
        self.examples = [
            {
                "question": "Do I have a greater right to probate in respect of my late father's estate?",
                "type": "Yes/No",
            },
            {
                "question": "When can i start paying the inheritance tax?",
                "type": "Span",
            },
            {
                "question": "Is interest payable if I agree to pay it in installments?",
                "type": "Yes/No",
            },
            {
                "question": "Am I eligible for the Maternity allowance and if so how much will I get?",
                "type": "Span",
            },
        ]
        self.task_description = 'Your task is to classify whether a question should be answer with Yes/No or a full span. I\'ll give you some examples first. You should only answer "Yes/No" or "Span".'
        self.qtype_chain = self._create_chain()

    def __call__(self, question):
        """
        Classifies the type of the given question.

        Args:
            question (str): The question to classify.

        Returns:
            str: The type of the question.
        """
        return self.qtype_chain.invoke({"question": question}).content

    def _create_chain(self):
        """
        Creates a chain of ICL and LLM models to classify the question type.

        Returns:
            A chain of ICL and LLM models to classify the question type.
        """
        list_icl_chat_examples = [("system", self.task_description)]
        prompt_template = "Question: {question}\nQuestion Type:"
        for x in self.examples:
            question = x["question"]
            qtype = x["type"]
            list_icl_chat_examples.append(
                ("human", prompt_template.format(question=question))
            )
            list_icl_chat_examples.append(("assistant", qtype))

        list_icl_chat_examples.append(("human", prompt_template))
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        qtype_chain = chat_prompt | self.llm
        return qtype_chain
