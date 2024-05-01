import random

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser

from .utils import SharcLabel


class TextPrompt:
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

    def __init__(self, llm, demonstrations, num_demonstrations_per_class, seed):
        self.llm = llm
        self.demonstrations = demonstrations
        random.seed(seed)

        # stratify demonstrations by label
        demonstrations_by_label = {}
        for demonstration in demonstrations:
            label = demonstration["label"]
            if label not in demonstrations_by_label:
                demonstrations_by_label[label] = []
            demonstrations_by_label[label].append(demonstration)

        # sample k demonstration per label
        demonstrations_sample = []
        for label, demonstrations in demonstrations_by_label.items():
            demonstrations_sample.extend(
                random.sample(demonstrations, num_demonstrations_per_class)
            )

        prompt_template = self.__get_prompt_template(demonstrations_sample)
        self.text_prompting_chain = prompt_template | llm.bind(stop=["\nHuman:"]) | StrOutputParser()

    def __call__(self, batch):
        """
        Invoke the text prompting chain with the given question, scenario, doc.
        Args:
            batch: a list of dictionaries containing the question, scenario, doc
        Returns:
            - the response from the text prompting chain
            - the list of prompts
        """
        input_prompts = []
        for x in batch:
            prompt = self.text_prompting_chain.get_prompts()[0].format(**x)
            input_prompts.append(prompt)
        return self.text_prompting_chain.batch(batch), input_prompts

    def __get_prompt_template(self, demonstrations):
        instr = "You must answer `yes`, `no`, or `not enough information` to the question and nothing else."
        prompt_format = (
            "Question: {scenario} {question}\nDocument: {doc}\nConversation history: {history}\nWhat is the answer to the question: {question} "
            + instr
        )
        # This is a prompt template used to format each individual example.
        example_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "{input}\nWhat is the answer to the question: {question} " + instr,
                ),
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
                    "You are a question answering system that answers questions given a document and a conversation history. The conversation history gives information about the background of the person posing the question. "
                    + instr,
                ),
                few_shot_prompt,
                ("human", prompt_format),
                ("ai", "")
            ]
        )
        return prompt_template

    def process_response(self, response):
        response_tokens = response.split()
        if "Yes" == response_tokens[0]:
            return SharcLabel.YES
        elif "No" == response_tokens[0]:
            return SharcLabel.NO
        else:
            return SharcLabel.NOT_ENOUGH_INFO
