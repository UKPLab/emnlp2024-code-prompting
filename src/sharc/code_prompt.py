import random

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from .utils import SharcLabel


class CodePrompt:
    """
    A class to handle text prompting for the SHARC dataset.
    Args:
        llm: the language model to use
        demonstrations: the demonstrations to use for text prompting
        num_demonstrations_per_class: the number of demonstrations to use per class
        seed: the seed to use for random sampling
    Methods:
        __call__: invoke the text prompting chain with the given question, scenario, doc, and history
        process_response: process the response from the text prompting chain
        __create_conv_history: create the conversation history for the text prompting chain
    """

    def __init__(
        self,
        llm,
        demonstrations,
        num_nl2code_demonstations,
        num_inference_demonstrations_per_class,
        seed,
    ):
        self.llm = llm
        self.num_nl2code_demonstations = num_nl2code_demonstations
        self.num_inference_demonstrations_per_class = (
            num_inference_demonstrations_per_class
        )
        random.seed(seed)

        # stratify demonstrations by label
        demonstrations_by_label = {}
        for demonstration in demonstrations:
            label = demonstration["label"]
            if label not in demonstrations_by_label:
                demonstrations_by_label[label] = []
            demonstrations_by_label[label].append(demonstration)

        # 1) Create NL2Code chain
        ## Create the NL2Code prompt template
        nl2code_prompt_template = self.__get_nl2code_prompt_template(
            random.sample(demonstrations, self.num_nl2code_demonstations)
        )
        ## Create the NL2Code chain
        nl2code_chain = nl2code_prompt_template | llm.bind(stop=["\nHuman:"]) | StrOutputParser()

        # 2) Create code inference chain
        ## Sample k demonstration per label
        code_prompting_demonstrations = []
        for label, demonstrations in demonstrations_by_label.items():
            code_prompting_demonstrations.extend(random.sample(demonstrations, 1))
        ## Create the code inference prompt template
        code_inference_prompt_template = self.__get_code_inference_chain(
            code_prompting_demonstrations
        )
        ## Create the code inference chain
        inference_chain = code_inference_prompt_template | llm.bind(stop=["\nHuman:"]) | StrOutputParser()

        # 3) Combine the two chains to create the code prompting chain
        code2inference = lambda code: {
            "code": code,
            "question_variable": self.get_question_variable(code),
        }
        self.code_prompting_chain =  nl2code_chain | code2inference | inference_chain
        

    def __call__(self, batch):
        """
        Invoke the text prompting chain with the given question, scenario, doc, and history.
        Args:
            batch: a list of dictionaries containing the question, scenario, doc, and history
        Returns:
            the response from the text prompting chain
        """
        return self.code_prompting_chain.batch(batch)

    def __get_nl2code_prompt_template(self, demonstrations):
        instr = "You have to create variables that model the text. Pay special attention to conditional statements and model them with `if blocks`. If the document uses a variable that is not initialized with the question, or the conversation history, you must define it in the section `# Other variables needed for the document:` and initialize it to None."
        prompt_format = (
            "Question: {scenario} {question}\nDocument: {doc}\nConversation history: {history}\nTransform this natural language text into code. "
            + instr
        )
        example_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "{input}\nTransform this natural language text into code. " + instr,
                ),
                ("ai", "{code}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=demonstrations,
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that transforms a text into pseudo-code. "
                    + instr,
                ),
                few_shot_prompt,
                ("human", prompt_format),
                ("ai", ""),
            ]
        )
        return prompt_template

    def __get_code_inference_chain(self, demonstrations):
        prompt_format = "{code}\n{question_variable} = "
        # This is a prompt template used to format each individual example.
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{code}\n{question_variable} = "),
                ("ai", "{label}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=demonstrations,
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a question-answering system that answers questions based on a document, and conversation history. The text is pseudo-code that models the document and conversation history. You must run the code and update the value of the variable that answers the question. The values can be True, False, or None.",
                ),
                few_shot_prompt,
                ("human", prompt_format),
                ("ai", ""),
            ]
        )
        return prompt_template

    def get_question_variable(self, code):
        for line in code.split("\n# Conversation history:")[0].split("\n"):
            if "None" in line:
                question_variable = line.split(" = ")[0]
                return question_variable
        return None

    def process_response(self, response):
        if "True" in response:
            return SharcLabel.YES
        elif "False" in response:
            return SharcLabel.NO
        else:
            return SharcLabel.NOT_ENOUGH_INFO
