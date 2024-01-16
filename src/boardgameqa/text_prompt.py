import random

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser

from .evaluation import Answer


class TextPrompt:
    """
    A class for text prompts for BoardgameQA dataset.

    Args:
        icl_examples (list): List of demonstrations.
        llm (LanguageModel): The language model used for generating responses.
        num_examples (int): Number of demonstrations.
        seed (int): Seed value for randomization. Default is 42.

    Attributes:
        icl_examples (list): List of demonstrations.
        llm (LanguageModel): The language model used for generating responses.
        num_examples (int): Number of demonstrations.
        seed (int): Seed value for randomization.
        chain (PromptChain): The prompt chain for generating responses.
        prompt_template (ChatPromptTemplate): The template for formatting each individual example.

    Methods:
        __call__(input_example): Generates a response for the given input example.
        _create_chain(): Creates the prompt chain for generating responses.
        _create_icl_demos_partitions(): Creates partitions of ICLE examples based on their labels.
        process_response(response): Processes the response and extracts the answer.

    """

    def __init__(self, icl_examples, llm, num_examples=6, seed=42):
        self.icl_examples = icl_examples
        self.llm = llm
        random.seed(seed)
        self.num_examples = num_examples
        self.chain, self.prompt_template = self._create_chain()

    def __call__(self, input_example):
        prompt_input = self.prompt_template.format_messages(input=input_example)
        response = self.chain.invoke({"input": input_example})
        return response, prompt_input

    def _create_chain(self):
        examples = []
        icl_demos = self._create_icl_demos_partitions()
        for x in icl_demos:
            examples.append({"input": x["example"], "output": x["proof"]})

        # This is a prompt template used to format each individual example.
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # print(few_shot_prompt.format())
        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a question-answering system that solves the problem of reasoning with contradictory information guided by preferences over sources of information.",
                ),
                few_shot_prompt,
                ("human", "{input}"),
            ]
        )
        chain = final_prompt | self.llm | StrOutputParser()
        return chain, final_prompt

    def _create_icl_demos_partitions(self):
        proved = []
        disproved = []
        unknown = []
        for x in self.icl_examples:
            if x["label"] == "proved":
                proved.append(x)
            elif x["label"] == "disproved":
                disproved.append(x)
            else:
                unknown.append(x)
        num_demos_per_class = self.num_examples / 3
        remainder_demos = self.num_examples % 3
        demos = (
            random.sample(proved, int(num_demos_per_class))
            + random.sample(disproved, int(num_demos_per_class))
            + random.sample(unknown, int(num_demos_per_class))
            + random.sample(self.icl_examples, remainder_demos)
        )
        random.shuffle(demos)
        return demos

    def process_response(self, response):
        answer = Answer.UNKNOWN
        for line in response.split("\n"):
            if "the answer is" in line:
                answer = line.split("the answer is")[1].strip()
                # clean answer
                answer = (
                    answer.replace("(", "").replace(")", "").replace('"', "").strip()
                )
                if "yes" in answer.lower():
                    answer = Answer.YES
                elif "no" in answer.lower():
                    answer = Answer.NO
                else:
                    answer = Answer.UNKNOWN
                return answer
        return answer
