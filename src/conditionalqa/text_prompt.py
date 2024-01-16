import random
import string

from langchain.prompts.chat import ChatPromptTemplate

from .question_type_classifier import QuestionType, QuestionTypeClassifier
from .oracle_retriever import get_summarized_doc


class TextPrompt:
    """
    A class that represents an end-to-end question answering system.

    Args:
    -----
    llm : LanguageModel
        A language model that will be used to answer the questions.
    train : list
        A list of training examples (for ICL).
    url2doc : dict
        A dictionary that maps URLs to documents.
    num_examples : int
        The number of examples to use for ICL.

    Attributes:
    -----------
    llm : LanguageModel
        A language model that will be used to answer the questions.
    train : list
        A list of training examples.
    url2doc : dict
        A dictionary that maps URLs to documents.
    num_examples : int
        The number of examples to use for training.
    task_reminder : str
        A string that reminds the user of the task requirements.
    end2endqa_chain : Chain
        A chain of components that will be used to answer the questions.

    Methods:
    --------
    __call__(self, question, doc): Answers a question given a document.
    _create_chain(self): Creates a chain of components that will be used to answer the questions.

    """

    def __init__(
        self,
        llm,
        train,
        url2doc,
        use_rationales=False,
        use_conditions=False,
        use_semistructure_docs=None,
        num_span_examples=5,
        num_yn_examples=6,
        seed=42,
    ):
        random.seed(seed)
        self.llm = llm
        self.use_rationales = use_rationales
        self.use_conditions = use_conditions
        self.use_semistructure_docs = use_semistructure_docs
        self.qtype_model = QuestionTypeClassifier(llm)

        self.train = train
        self.url2doc = url2doc
        self.num_span_examples = num_span_examples
        self.num_yn_examples = num_yn_examples

        self.yn_task_reminder = 'Answers can be "yes" or "no". You have to write "yes" or "no" and nothing else. Do not write "it depends or anything similar". You HAVE to write only "yes" or "no", even if you are uncertain.'
        self.span_task_reminder = "Answers must be a short span of the document. You have to extract the span from the document. Do not write anything else."

        if self.use_conditions:
            condition_task_desc = 'Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true after the answer. We call these sentence(s) "conditions". You must have used this sentence(s) in your reasoning steps.'
            self.span_task_reminder += "\n" + condition_task_desc
            self.yn_task_reminder += "\n" + condition_task_desc

        self.end2end_ynqa_chain, self.yn_prompt_tmplt = self._create_chain(
            QuestionType.YESNO,
        )
        self.end2end_spanqa_chain, self.span_prompt_tmplt = self._create_chain(
            QuestionType.SPAN
        )

    def __call__(self, question, short_question, doc):
        """
        Answers a question given a document.

        Parameters:
        -----------
        question : str
            The question to answer.
        doc : str
            The document to search for the answer.

        Returns:
        --------
        str
            The answer to the question.
        """
        qtype_raw = self.qtype_model(short_question)
        if "span" in qtype_raw.lower():
            qtype = QuestionType.SPAN
        else:
            qtype = QuestionType.YESNO
        metadata = {"question_type": qtype_raw}

        prompt = ""
        if qtype == QuestionType.YESNO:
            response = self.end2end_ynqa_chain.invoke(
                {
                    "question": question,
                    "document": doc,
                    "task_reminder": self.yn_task_reminder,
                }
            ).content
            prompt = self.yn_prompt_tmplt.format(
                question=question, document=doc, task_reminder=self.yn_task_reminder
            )
        else:
            response = self.end2end_spanqa_chain.invoke(
                {
                    "question": question,
                    "document": doc,
                    "task_reminder": self.span_task_reminder,
                }
            ).content
            prompt = self.span_prompt_tmplt.format(
                question=question, document=doc, task_reminder=self.span_task_reminder
            )

        metadata["full_response"] = response
        metadata["prompt"] = prompt
        answer = self.format_prediction(response, qtype)

        return answer, metadata

    def _create_chain(self, answer_type):
        """
        Creates a chain for the end-to-end QA system.

        Args:
            answer_type (str): The type of answer to generate.

        Returns:
            end2endqa_chain: The chain for the end-to-end QA system.
        """
        if answer_type == QuestionType.YESNO:
            yn_examples = self.num_yn_examples // 2
            yes_examples = self._get_yes_examples(self.train, yn_examples)
            no_examples = self._get_no_examples(self.train, yn_examples)
            examples = yes_examples + no_examples
            random.shuffle(examples)
            task_reminder = self.yn_task_reminder
        else:
            span_examples = self._get_span_examples(self.train, self.num_span_examples)
            examples = span_examples
            task_reminder = self.span_task_reminder
        task_description = f"You are a helpful assistant that answers questions given a document. {task_reminder} I will give you some examples first."

        list_icl_chat_examples = [("system", task_description)]

        prompt_template = "Question: {question}\nDocument: {document}\n{task_reminder}"
        output_template = ""
        if self.use_rationales:
            prompt_template += " Let's think step by step:\n\n"
            output_template += "{rationales}\nAnswer: {answer}\n\n"
        else:
            prompt_template += " Answer:"
            output_template += "{answer}\n\n"

        if self.use_conditions:
            # remove the last \n\n
            output_template = output_template[:-2]
            output_template += " . Conditions: {conditions}\n\n"

        for x in examples:
            question = x["scenario"] + ". " + x["question"]
            if self.use_semistructure_docs is None:
                summarized_doc = get_summarized_doc(x, self.url2doc)
            elif self.use_semistructure_docs == "semi_structured_doc":
                summarized_doc = x["semi_structured_doc"]
            elif self.use_semistructure_docs == "extended_doc":
                summarized_doc = x["extended_doc"]
            elif self.use_semistructure_docs == "nl_code":
                summarized_doc = x["nl_code"]
            elif self.use_semistructure_docs == "nl_code_var":
                summarized_doc = x["nl_code_var"]
            answer = x["answers"][0][0]
            list_icl_chat_examples.append(
                (
                    "human",
                    prompt_template.format(
                        question=question,
                        document=summarized_doc,
                        task_reminder=task_reminder,
                    ),
                )
            )
            rationales = ""
            conditions = ""
            if self.use_rationales:
                rationales = "\n".join(x["evidences"])
            if self.use_conditions:
                conditions = "\n".join(x["answers"][0][1])
            prompt_output = self.format_output_template(
                output_template, answer, rationales, conditions
            )
            list_icl_chat_examples.append(
                (
                    "assistant",
                    prompt_output,
                )
            )

        list_icl_chat_examples.append(("human", prompt_template))
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        end2endqa_chain = chat_prompt | self.llm
        return end2endqa_chain, chat_prompt

    def format_output_template(self, output_template, answer, rationales, conditions):
        formatter = string.Formatter()
        ans_vars = [
            field_name
            for _, field_name, _, _ in formatter.parse(output_template)
            if field_name
        ]
        dict_ans_vars = dict()
        if "rationales" in ans_vars:
            dict_ans_vars["rationales"] = rationales
        if "answer" in ans_vars:
            dict_ans_vars["answer"] = answer
        if "conditions" in ans_vars:
            dict_ans_vars["conditions"] = conditions

        output = output_template.format(**dict_ans_vars)
        if len(conditions) == 0:
            output = output.replace(". Conditions:", "\n\n")
        return output

    def format_prediction(self, prediction, qtype):
        answer = prediction
        conditions = []

        if self.use_rationales:
            try:
                answer = prediction.split("Answer: ")[1].strip()
            except IndexError:
                answer = prediction

        if self.use_conditions:
            try:
                conditions_text = answer.split("Conditions: ")[1].strip()
                conditions = conditions_text.split("\n")
                # remove the conditions from the answer
                answer = answer.split(". Conditions:")[0].strip()
            except IndexError:
                conditions = []

        if qtype == QuestionType.YESNO and answer.lower() not in ["yes", "no"]:
            # the model is not confident enough to answer yes/no
            # maybe both answers are correct
            answer = [["yes", []], ["no", []]]
        else:
            answer = [[answer, conditions]]
        return answer

    def _get_answer_type(self, answer):
        if answer in ["yes", "no"]:
            return QuestionType.YESNO
        else:
            return QuestionType.SPAN

    def _get_yes_examples(self, train, num_examples):
        yes_examples = [x for x in train if x["answers"][0][0] == "yes"]
        return random.sample(yes_examples, num_examples)

    def _get_no_examples(self, train, num_examples):
        no_examples = [x for x in train if x["answers"][0][0] == "no"]
        return random.sample(no_examples, num_examples)

    def _get_span_examples(self, train, num_examples):
        span_examples = [x for x in train if x["answers"][0][0] not in ["yes", "no"]]
        return random.sample(span_examples, num_examples)
