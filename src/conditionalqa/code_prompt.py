import random
import string
import time

from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable.history import RunnableWithMessageHistory

from .question_type_classifier import QuestionType, QuestionTypeClassifier
from .translation import Doc2Code, Question2Code
from ..utils import get_current_time


class CodePrompt:
    def __init__(
        self,
        llm,
        question2code_examples,
        doc2code_examples,
        qacode2yn_answer_examples,
        qacode2span_answer_examples,
        use_rationales=None,
        use_conditions=False,
        use_memory=False,
        num_translation_examples=7,
        num_interpreter_examples=6,
        seed=0,
    ):
        self.q2code = Question2Code(
            llm, question2code_examples, num_translation_examples, seed
        )
        self.doc2code = Doc2Code(llm, doc2code_examples, num_translation_examples, seed)
        self.qtype_model = QuestionTypeClassifier(llm)
        self.qacode2answer = Code2Answer(
            llm,
            qacode2yn_answer_examples,
            qacode2span_answer_examples,
            use_rationales=use_rationales,
            use_conditions=use_conditions,
            use_memory=use_memory,
            num_examples=num_interpreter_examples,
            seed=seed,
        )

    def __call__(
        self,
        long_question,
        short_question,
        doc,
        q_code=None,
        doc_code=None,
        qtype_raw=None,
        verbose=False,
    ):
        # 1) Get question code
        if q_code is None:
            if verbose:
                print("Running translation for the question...")
            q_code = self.q2code(long_question)

        # 2) Get document code
        if doc_code is None:
            if verbose:
                print("Running translation for the document...")
            doc_code = self.doc2code(doc)

        # 3) Get question type
        if qtype_raw is None:
            if verbose:
                print("Running QA type predictor...")
            qtype_raw = self.qtype_model(short_question)
            if verbose:
                print("Question type:", qtype_raw)

        if "span" in qtype_raw.lower():
            qtype = QuestionType.SPAN
        else:
            qtype = QuestionType.YESNO

        # 4) Get answer
        answer, metadata, conditions = self.qacode2answer(
            q_code, doc_code, long_question, qtype
        )

        if qtype == QuestionType.YESNO and answer.lower() not in ["yes", "no"]:
            # the model is not confident enough to answer yes/no
            # maybe both answers are correct
            answer = [["yes", conditions], ["no", conditions]]
        else:
            answer = [[answer, conditions]]

        metadata["q_code"] = q_code
        metadata["doc_code"] = doc_code
        metadata["timestamp"] = get_current_time()
        return answer, metadata


class Code2Answer:
    def __init__(
        self,
        llm,
        qacode2yn_answer_examples,
        qacode2span_answer_examples,
        use_rationales,
        use_conditions,
        use_memory=False,
        num_examples=6,
        seed=0,
    ):
        random.seed(seed)
        self.llm = llm
        self.use_rationales = use_rationales
        self.use_conditions = use_conditions
        self.use_memory = use_memory
        self.num_examples = num_examples

        self.span_task_description = "You are a helpful assistant. Your task is to process a pseudo-code that describes a question and a document. You need to reason using that document and the comments to return the answers. Answers must be a short span of the document. You have to extract the span from the code comments. Do not write anything else. I will give you some examples first."
        self.qacode2span_answer_examples = qacode2span_answer_examples
        self.span_task_reminder = "# Answers must be a short span of the document. You have to extract the span from the code comments. Do not write anything else."

        self.yn_task_description = 'You are a helpful assistant. Your task is to process a pseudo-code that describes a question and a document. You need to reason using that document and the comments to return the answers. Answers can be "yes" or "no". You have to write "yes" or "no" and nothing else. I will give you some examples first.'
        self.qacode2yn_answer_examples = qacode2yn_answer_examples
        self.yn_task_reminder = '# Answers can be "yes" or "no". You have to write "yes" or "no" and nothing else. Do not write "it depends or anything similar". You HAVE to write only "yes" or "no", even if you are uncertain.'

        if self.use_conditions:
            condition_task_desc = "Some answers may required the assumption of some sentences from the text to be true. If you think that is the case, you must write the full sentence(s) that you think are required to be true in the code comments. You must have used this sentence(s) in your reasoning steps."
            self.span_task_reminder += "\n#" + condition_task_desc
            self.yn_task_reminder += "\n#" + condition_task_desc

        self.code2span_answer_chain = self.__create_span_chain()
        self.code2yn_answer_chain = self.__create_yesno_chain()

    def __call__(self, q_code, doc_code, question, qtype):
        if self.use_memory:
            chat_memory_id = str(time.time())
            chain_config = {"configurable": {"session_id": chat_memory_id}}
        else:
            chain_config = {}
        prompt = ""
        if qtype == QuestionType.SPAN:
            llm_response = self.code2span_answer_chain.invoke(
                {
                    "q_code": q_code,
                    "doc_code": doc_code,
                    "question": question,
                    "task_reminder": self.span_task_reminder,
                }
            ).content
            prompt = self.span_prompt.format(
                q_code=q_code,
                doc_code=doc_code,
                question=question,
                task_reminder=self.span_task_reminder,
            )
        else:
            llm_response = self.code2yn_answer_chain.invoke(
                {
                    "q_code": q_code,
                    "doc_code": doc_code,
                    "question": question,
                    "task_reminder": self.yn_task_reminder,
                },
                config=chain_config,
            ).content
            prompt = self.yesno_prompt.format(
                q_code=q_code,
                doc_code=doc_code,
                question=question,
                task_reminder=self.span_task_reminder,
            )
        answer, conditions = self._process_llm_response(llm_response)
        metadata = {"qtype": qtype, "full_response": llm_response, "prompt": prompt}

        # check answer requirements
        if self.use_memory and qtype == QuestionType.YESNO:
            if answer.lower() not in ["yes", "no"]:
                # ask for a new answer
                new_response = self.code2yn_answer_chain.invoke(
                    {
                        "q_code": q_code,
                        "doc_code": doc_code,
                        "question": question,
                        "task_reminder": f"The answer have to be 'yes' or 'no' and nothing else. Your answer {llm_response} is not useful for me. You can write conditions after the answer. This is the output format you must follow in this case: 'yes\t if: `condition_variable` # `text_conditions`' or 'no\t if: `condition_variable` # `text_conditions`'.",
                    },
                    config={"configurable": {"session_id": chat_memory_id}},
                ).content
                answer, conditions = self._process_llm_response(new_response)
                metadata["full_response2"] = new_response

        return answer, metadata, conditions

    def _process_llm_response(self, llm_response):
        conditions = []
        if self.use_rationales is not None:
            try:
                answer = llm_response.split("# Answer: ")[1].strip()
            except IndexError:
                return llm_response, conditions

        if self.use_conditions:
            # format: answer\t if: condition1\ncondition2\n...
            ans_cond_tup = answer.split("\t if: ")
            if len(ans_cond_tup) > 1:
                conditions = ans_cond_tup[1].split("\n")
            answer = ans_cond_tup[0]
        return answer, conditions

    def __create_yesno_chain(self):
        list_icl_chat_examples = [("system", self.yn_task_description)]

        # prepare the templates
        code_template, answer_template = self._get_templates()
        demosntrations = self.__sample_yesno_demonstrations()
        # get all ICL examples
        for code2answer in demosntrations:
            rationales = self._get_rationales(code2answer)
            conditions = self._get_conditions(code2answer)

            # format the templates
            code, answer = self._format_templates(
                code_template,
                answer_template,
                code2answer["q_code"],
                code2answer["doc_code"],
                code2answer["question"],
                self.span_task_reminder,
                rationales,
                code2answer["answer"],
                conditions,
            )

            # add to the list of examples
            list_icl_chat_examples.append(("human", code))
            list_icl_chat_examples.append(("assistant", answer))

        # add memory
        if self.use_memory:
            list_icl_chat_examples.append(
                MessagesPlaceholder(variable_name="chat_history"),
            )
        # add user prompt
        list_icl_chat_examples.append(("human", code_template))
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        self.yesno_prompt = chat_prompt
        code2answer_chain = chat_prompt | self.llm

        if self.use_memory:
            chain_with_history = RunnableWithMessageHistory(
                code2answer_chain,
                RedisChatMessageHistory,
                input_messages_key="question",
                history_messages_key="chat_history",
            )
            return chain_with_history
        else:
            return code2answer_chain

    def __sample_yesno_demonstrations(self):
        yes_examples = [
            x for x in self.qacode2yn_answer_examples if x["answer"] == "yes"
        ]
        no_examples = [x for x in self.qacode2yn_answer_examples if x["answer"] == "no"]
        yes_demonstrations = random.sample(yes_examples, self.num_examples // 2)
        no_demonstrations = random.sample(no_examples, self.num_examples // 2)
        demonstrations = yes_demonstrations + no_demonstrations
        random.shuffle(demonstrations)
        return demonstrations

    def __create_span_chain(self):
        list_icl_chat_examples = [("system", self.span_task_description)]

        # prepare the templates
        code_template, answer_template = self._get_templates()

        demonstrations = random.sample(
            self.qacode2span_answer_examples, self.num_examples
        )
        for code2answer in demonstrations:
            rationales = self._get_rationales(code2answer)
            conditions = self._get_conditions(code2answer)

            # format the templates
            code, answer = self._format_templates(
                code_template,
                answer_template,
                code2answer["q_code"],
                code2answer["doc_code"],
                code2answer["question"],
                self.span_task_reminder,
                rationales,
                code2answer["answer"],
                conditions,
            )

            list_icl_chat_examples.append(("human", code))
            list_icl_chat_examples.append(("assistant", answer))

        list_icl_chat_examples.append(("human", code_template))
        chat_prompt = ChatPromptTemplate.from_messages(list_icl_chat_examples)
        self.span_prompt = chat_prompt
        code2answer_chain = chat_prompt | self.llm
        return code2answer_chain

    def _get_templates(self):
        if self.use_rationales is not None:
            code_template = "{q_code}\n{doc_code}\n# Question: {question}\n{task_reminder}\n# Let's think step by step:"
            answer_template = "{rationales}\n# Answer: {answer}"
        else:
            code_template = "{q_code}\n{doc_code}\n# Question: {question}\n{task_reminder}\n# Answer:"
            answer_template = "{answer}"

        if self.use_conditions:
            answer_template += "\t if: {conditions}"

        return code_template, answer_template

    def _get_rationales(self, code2answer):
        rationales = ""
        if self.use_rationales is not None:
            if self.use_rationales == RationaleType.CODE:
                rationales = "#" + code2answer["code_rationales"]
            else:
                rationales = "#" + code2answer["evidences"].replace("\n", "\n# ")
        return rationales

    def _get_conditions(self, code2answer):
        conditions = ""
        if self.use_conditions:
            conditions = "\n".join(code2answer["conditions"])
        return conditions

    def _format_templates(
        self,
        code_template,
        answer_template,
        q_code,
        doc_code,
        question,
        task_reminder,
        rationales,
        answer,
        conditions,
    ):
        dict_code_vars = dict()
        dict_code_vars["q_code"] = q_code
        dict_code_vars["doc_code"] = doc_code
        dict_code_vars["question"] = question
        dict_code_vars["task_reminder"] = task_reminder

        formatter = string.Formatter()
        ans_vars = [
            field_name
            for _, field_name, _, _ in formatter.parse(answer_template)
            if field_name
        ]
        dict_ans_vars = dict()
        if "rationales" in ans_vars:
            dict_ans_vars["rationales"] = rationales
        if "answer" in ans_vars:
            dict_ans_vars["answer"] = answer
        if "conditions" in ans_vars:
            dict_ans_vars["conditions"] = conditions

        code = code_template.format(**dict_code_vars)
        answer = answer_template.format(**dict_ans_vars)
        if len(conditions) == 0:
            answer = answer.replace("\t if: ", "")
        return code, answer


class RationaleType:
    """
    Represents the types of rationales that can be used.
    """

    CODE = "code"
    TEXT = "text"
