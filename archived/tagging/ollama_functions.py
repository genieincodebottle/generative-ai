import json
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.runnables.base import RunnableMap
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.tools import BaseTool

DEFAULT_SYSTEM_TEMPLATE = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""  # noqa: E501

DEFAULT_RESPONSE_FUNCTION = {
    "name": "__conversational_response",
    "description": (
        "Respond conversationally if no other tools should be called for a given query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Conversational response to the user.",
            },
        },
        "required": ["response"],
    },
}

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_DictOrPydantic = Union[Dict, _BM]


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and (
        issubclass(obj, BaseModel) or BaseModel in obj.__bases__
    )


def convert_to_ollama_tool(tool: Any) -> Dict:
    """Convert a tool to an Ollama tool."""
    if _is_pydantic_class(tool):
        schema = tool.construct().schema()
        definition = {"name": schema["title"], "properties": schema["properties"]}
        if "required" in schema:
            definition["required"] = schema["required"]

        return definition
    raise ValueError(
        f"Cannot convert {tool} to an Ollama tool. {tool} needs to be a Pydantic model."
    )


class _AllReturnType(TypedDict):
    raw: BaseMessage
    parsed: Optional[_DictOrPydantic]
    parsing_error: Optional[BaseException]


def parse_response(message: BaseMessage) -> str:
    """Extract `function_call` from `AIMessage`."""
    if isinstance(message, AIMessage):
        kwargs = message.additional_kwargs
        if "function_call" in kwargs:
            if "arguments" in kwargs["function_call"]:
                return kwargs["function_call"]["arguments"]
            raise ValueError(
                f"`arguments` missing from `function_call` within AIMessage: {message}"
            )
        raise ValueError(
            "`function_call` missing from `additional_kwargs` "
            f"within AIMessage: {message}"
        )
    raise ValueError(f"`message` is not an instance of `AIMessage`: {message}")


class OllamaFunctions(ChatOllama):
    """Function chat model that uses Ollama API."""

    tool_system_prompt_template: str = DEFAULT_SYSTEM_TEMPLATE

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self.bind(functions=tools, **kwargs)

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]:
        ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from langchain_experimental.llms import OllamaFunctions
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = OllamaFunctions(model="phi3", format="json", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Pydantic schema (include_raw=True):
            .. code-block:: python

                from langchain_experimental.llms import OllamaFunctions
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = OllamaFunctions(model="phi3", format="json", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: dict schema (method="include_raw=False):
            .. code-block:: python

                from langchain_experimental.llms import OllamaFunctions, convert_to_ollama_tool
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_ollama_tool(AnswerWithJustification)
                llm = OllamaFunctions(model="phi3", format="json", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }


        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        llm = self.bind_tools(tools=[schema], format="json")
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticOutputParser(
                pydantic_object=schema
            )
        else:
            output_parser = JsonOutputParser()

        parser_chain = RunnableLambda(parse_response) | output_parser
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser_chain, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | parser_chain

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        functions = kwargs.get("functions", [])
        if "functions" in kwargs:
            del kwargs["functions"]
        if "function_call" in kwargs:
            functions = [
                fn for fn in functions if fn["name"] == kwargs["function_call"]["name"]
            ]
            if not functions:
                raise ValueError(
                    "If `function_call` is specified, you must also pass a "
                    "matching function in `functions`."
                )
            del kwargs["function_call"]
        elif not functions:
            functions.append(DEFAULT_RESPONSE_FUNCTION)
        if _is_pydantic_class(functions[0]):
            functions = [convert_to_ollama_tool(fn) for fn in functions]
        system_message_prompt_template = SystemMessagePromptTemplate.from_template(
            self.tool_system_prompt_template
        )
        system_message = system_message_prompt_template.format(
            tools=json.dumps(functions, indent=2)
        )
        response_message = super()._generate(
            [system_message] + messages, stop=stop, run_manager=run_manager, **kwargs
        )
        chat_generation_content = response_message.generations[0].text
        if not isinstance(chat_generation_content, str):
            raise ValueError("OllamaFunctions does not support non-string output.")
        try:
            parsed_chat_result = json.loads(chat_generation_content)
        except json.JSONDecodeError:
            raise ValueError(
                f"""'{self.model}' did not respond with valid JSON. 
                Please try again. 
                Response: {chat_generation_content}"""
            )
        called_tool_name = parsed_chat_result["tool"]
        called_tool_arguments = parsed_chat_result["tool_input"]
        called_tool = next(
            (fn for fn in functions if fn["name"] == called_tool_name), None
        )
        if called_tool is None:
            raise ValueError(
                f"Failed to parse a function call from {self.model} output: "
                f"{chat_generation_content}"
            )
        if called_tool["name"] == DEFAULT_RESPONSE_FUNCTION["name"]:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=called_tool_arguments["response"],
                        )
                    )
                ]
            )

        response_message_with_functions = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": called_tool_name,
                    "arguments": json.dumps(called_tool_arguments)
                    if called_tool_arguments
                    else "",
                },
            },
        )

        return ChatResult(
            generations=[ChatGeneration(message=response_message_with_functions)]
        )

    @property
    def _llm_type(self) -> str:
        return "ollama_functions"