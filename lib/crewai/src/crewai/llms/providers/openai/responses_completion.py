from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx
from openai import APIConnectionError, AsyncOpenAI, NotFoundError, OpenAI
from pydantic import BaseModel

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.llms.hooks.base import BaseInterceptor
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool


class OpenAIResponsesCompletion(BaseLLM):
    """OpenAI Responses API implementation.

    This class provides direct integration with OpenAI's Responses API,
    offering a simplified interface for text generation with support for
    streaming, function calling, and structured outputs.

    The Responses API uses a simpler input format and provides stateful
    interactions with built-in tools support.

    Example:
        >>> from crewai.llms.providers.openai.responses_completion import (
        ...     OpenAIResponsesCompletion
        ... )
        >>> llm = OpenAIResponsesCompletion(model="gpt-4o")
        >>> response = llm.call("What is 2 + 2?")
        >>> print(response)
        4

    Attributes:
        model: The OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
        stream: Whether to stream responses
        temperature: Sampling temperature (0-2)
        max_output_tokens: Maximum tokens in the response
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, Any] | None = None,
        client_params: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        seed: int | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        reasoning_effort: str | None = None,
        include_reasoning: bool = False,
        provider: str | None = None,
        interceptor: BaseInterceptor[httpx.Request, httpx.Response] | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI Responses API client.

        Args:
            model: The model to use (default: "gpt-4o")
            api_key: OpenAI API key
            base_url: Base URL for API calls
            organization: Organization ID
            project: Project ID
            timeout: Request timeout
            max_retries: Maximum number of retries
            default_headers: Default headers for requests
            default_query: Default query parameters
            client_params: Additional client parameters
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_output_tokens: Maximum output tokens
            seed: Random seed for reproducibility
            stream: Whether to stream responses
            response_format: Response format specification
            reasoning_effort: Reasoning effort for o-series models
            include_reasoning: Whether to include reasoning content in response
            provider: Provider name
            interceptor: HTTP interceptor for hooks
            instructions: System instructions for the model
            previous_response_id: ID of previous response for multi-turn
            **kwargs: Additional parameters
        """
        if provider is None:
            provider = kwargs.pop("provider", "openai_responses")

        self.interceptor = interceptor

        # Client configuration attributes
        self.organization = organization
        self.project = project
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.client_params = client_params
        self.timeout = timeout
        self.base_url = base_url
        self.api_base = kwargs.pop("api_base", None)

        super().__init__(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            provider=provider,
            **kwargs,
        )

        client_config = self._get_client_params()
        if self.interceptor:
            transport = HTTPTransport(interceptor=self.interceptor)
            http_client = httpx.Client(transport=transport)
            client_config["http_client"] = http_client

        self.client = OpenAI(**client_config)

        async_client_config = self._get_client_params()
        if self.interceptor:
            async_transport = AsyncHTTPTransport(interceptor=self.interceptor)
            async_http_client = httpx.AsyncClient(transport=async_transport)
            async_client_config["http_client"] = async_http_client

        self.async_client = AsyncOpenAI(**async_client_config)

        # Completion parameters
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.stream = stream
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.include_reasoning = include_reasoning
        self.instructions = instructions
        self.previous_response_id = previous_response_id
        self.is_o_series_model = any(
            prefix in model.lower() for prefix in ["o1", "o3", "o4"]
        )

        # Store last reasoning content from responses
        self._last_reasoning_content: list[str] = []
        self._last_raw_response: Any = None

    @property
    def last_reasoning_content(self) -> list[str]:
        """Get the reasoning content from the last API call.

        Returns:
            List of reasoning summary strings from the last response.
        """
        return self._last_reasoning_content

    @property
    def last_raw_response(self) -> Any:
        """Get the raw response object from the last API call.

        Returns:
            The raw response object from the last API call.
        """
        return self._last_raw_response

    def _extract_reasoning_content(self, response: Any) -> list[str]:
        """Extract reasoning content from a Responses API response.

        Args:
            response: The API response object.

        Returns:
            List of reasoning summary strings.
        """
        reasoning_content = []
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type") and item.type == "reasoning":
                    # Handle summary format (list of summary items with text)
                    if hasattr(item, "summary") and item.summary:
                        for summary_item in item.summary:
                            if hasattr(summary_item, "text") and summary_item.text:
                                reasoning_content.append(summary_item.text)
                    # Handle content format (list of content items)
                    if hasattr(item, "content") and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, "text") and content_item.text:
                                reasoning_content.append(content_item.text)
                    # Handle direct text attribute
                    if hasattr(item, "text") and item.text:
                        reasoning_content.append(item.text)
        return reasoning_content

    def _get_client_params(self) -> dict[str, Any]:
        """Get OpenAI client parameters."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("OPENAI_API_KEY is required")

        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url
            or self.api_base
            or os.getenv("OPENAI_BASE_URL")
            or None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        client_params = {k: v for k, v in base_params.items() if v is not None}

        if self.client_params:
            client_params.update(self.client_params)

        return client_params

    def _format_input(self, messages: str | list[LLMMessage]) -> str | list[dict[str, Any]]:
        """Format messages to input for Responses API.

        The Responses API accepts either a simple string or a list of content blocks.

        Args:
            messages: Input messages (string or list of message dicts)

        Returns:
            Formatted input for the Responses API
        """
        if isinstance(messages, str):
            return messages

        # For message list, combine into appropriate format
        # The Responses API can accept input as a list of content items
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System messages become instructions in Responses API
                # We'll handle this separately via instructions parameter
                continue
            elif role == "user":
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": content,
                })
            elif role == "assistant":
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": content,
                })

        # If only one user message with simple content, return as string
        if len(input_items) == 1 and input_items[0]["role"] == "user":
            return input_items[0]["content"]

        return input_items

    def _extract_system_instructions(self, messages: str | list[LLMMessage]) -> str | None:
        """Extract system instructions from messages.

        The Responses API uses a separate 'instructions' parameter for system prompts.

        Args:
            messages: Input messages

        Returns:
            System instructions string or None
        """
        if isinstance(messages, str):
            return self.instructions

        system_parts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)

        if system_parts:
            combined = "\n\n".join(system_parts)
            if self.instructions:
                return f"{self.instructions}\n\n{combined}"
            return combined

        return self.instructions

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Call OpenAI Responses API.

        Args:
            messages: Input messages for the response
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output

        Returns:
            Response text or tool call result
        """
        try:
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            formatted_messages = self._format_messages(messages)

            if not self._invoke_before_llm_call_hooks(formatted_messages, from_agent):
                raise ValueError("LLM call blocked by before_llm_call hook")

            params = self._prepare_response_params(
                messages=messages, tools=tools, response_model=response_model
            )

            if self.stream:
                return self._handle_streaming_response(
                    params=params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return self._handle_response(
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
                messages=formatted_messages,
            )

        except Exception as e:
            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def acall(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        """Async call to OpenAI Responses API.

        Args:
            messages: Input messages for the response
            tools: List of tool/function definitions
            callbacks: Callback functions (not used in native implementation)
            available_functions: Available functions for tool calling
            from_task: Task that initiated the call
            from_agent: Agent that initiated the call
            response_model: Response model for structured output

        Returns:
            Response text or tool call result
        """
        try:
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )

            formatted_messages = self._format_messages(messages)

            params = self._prepare_response_params(
                messages=messages, tools=tools, response_model=response_model
            )

            if self.stream:
                return await self._ahandle_streaming_response(
                    params=params,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )

            return await self._ahandle_response(
                params=params,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
                messages=formatted_messages,
            )

        except Exception as e:
            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _prepare_response_params(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Prepare parameters for OpenAI Responses API.

        Args:
            messages: Input messages
            tools: Optional tools for function calling
            response_model: Optional response model for structured output

        Returns:
            Parameters dictionary for the API call
        """
        input_data = self._format_input(messages)
        instructions = self._extract_system_instructions(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }

        if instructions:
            params["instructions"] = instructions

        if self.stream:
            params["stream"] = True

        # Add optional parameters
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens
        if self.seed is not None:
            params["seed"] = self.seed
        if self.previous_response_id is not None:
            params["previous_response_id"] = self.previous_response_id

        # Handle reasoning parameters
        if self.reasoning_effort:
            reasoning_config: dict[str, Any] = {"effort": self.reasoning_effort}
            # Add summary to get readable reasoning content
            if self.include_reasoning:
                reasoning_config["summary"] = "auto"
            params["reasoning"] = reasoning_config
        elif self.include_reasoning:
            # If include_reasoning is True but no effort specified, just request summary
            params["reasoning"] = {"summary": "auto"}

        # Handle response format for structured output
        if response_model:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "json_schema": generate_model_description(response_model),
                }
            }
        elif self.response_format is not None:
            if isinstance(self.response_format, type) and issubclass(
                self.response_format, BaseModel
            ):
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "json_schema": generate_model_description(self.response_format),
                    }
                }
            elif isinstance(self.response_format, dict):
                params["text"] = {"format": self.response_format}

        # Handle tools
        if tools:
            params["tools"] = self._convert_tools_for_responses(tools)

        return params

    def _convert_tools_for_responses(
        self, tools: list[dict[str, BaseTool]]
    ) -> list[dict[str, Any]]:
        """Convert CrewAI tool format to OpenAI Responses API format.

        Args:
            tools: List of CrewAI tools

        Returns:
            List of tools in Responses API format
        """
        from crewai.llms.providers.utils.common import safe_tool_conversion

        responses_tools = []

        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")

            responses_tool = {
                "type": "function",
                "name": name,
                "description": description,
            }

            if parameters:
                if isinstance(parameters, dict):
                    responses_tool["parameters"] = parameters
                else:
                    responses_tool["parameters"] = dict(parameters)

            responses_tools.append(responses_tool)

        return responses_tools

    def _handle_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
        messages: list[LLMMessage] | None = None,
    ) -> str | Any:
        """Handle non-streaming response from Responses API.

        Args:
            params: API call parameters
            available_functions: Available functions for tool calling
            from_task: Task context
            from_agent: Agent context
            response_model: Response model for structured output
            messages: Original messages for event emission

        Returns:
            Response text or structured output
        """
        try:
            response = self.client.responses.create(**params)

            # Store raw response and extract reasoning content
            self._last_raw_response = response
            self._last_reasoning_content = self._extract_reasoning_content(response)

            # Extract token usage
            usage = self._extract_responses_token_usage(response)
            self._track_token_usage_internal(usage)

            # Check for tool calls
            if hasattr(response, "output") and response.output:
                for output_item in response.output:
                    if hasattr(output_item, "type") and output_item.type == "function_call":
                        if available_functions:
                            function_name = output_item.name
                            try:
                                function_args = json.loads(output_item.arguments)
                            except json.JSONDecodeError as e:
                                logging.error(f"Failed to parse function arguments: {e}")
                                function_args = {}

                            result = self._handle_tool_execution(
                                function_name=function_name,
                                function_args=function_args,
                                available_functions=available_functions,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                            if result is not None:
                                return result

            # Get the output text
            content = getattr(response, "output_text", "") or ""
            content = self._apply_stop_words(content)

            # Handle structured output validation
            if response_model:
                try:
                    structured_result = self._validate_structured_output(
                        content, response_model
                    )
                    if isinstance(structured_result, BaseModel):
                        structured_json = structured_result.model_dump_json()
                        self._emit_call_completed_event(
                            response=structured_json,
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=messages,
                        )
                        return structured_json
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
            )

            if usage.get("total_tokens", 0) > 0:
                logging.info(f"OpenAI Responses API usage: {usage}")

            content = self._invoke_after_llm_call_hooks(
                messages or [], content, from_agent
            )

            return content

        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                logging.error(f"Context window exceeded: {e}")
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _handle_streaming_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming response from Responses API.

        Args:
            params: API call parameters
            available_functions: Available functions for tool calling
            from_task: Task context
            from_agent: Agent context
            response_model: Response model for structured output

        Returns:
            Complete response text
        """
        full_response = ""
        function_calls: dict[str, dict[str, str]] = {}
        usage_data = {"total_tokens": 0}

        try:
            with self.client.responses.create(**params) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)

                    # Handle text delta events
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            full_response += delta
                            self._emit_stream_chunk_event(
                                chunk=delta,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                    # Handle function call events
                    elif event_type == "response.function_call_arguments.delta":
                        call_id = getattr(event, "call_id", "default")
                        if call_id not in function_calls:
                            function_calls[call_id] = {
                                "name": "",
                                "arguments": "",
                            }
                        delta = getattr(event, "delta", "")
                        if delta:
                            function_calls[call_id]["arguments"] += delta

                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item and getattr(item, "type", None) == "function_call":
                            call_id = getattr(item, "call_id", "default")
                            if call_id not in function_calls:
                                function_calls[call_id] = {
                                    "name": getattr(item, "name", ""),
                                    "arguments": "",
                                }
                            else:
                                function_calls[call_id]["name"] = getattr(item, "name", "")

                    # Handle completion event with usage
                    elif event_type == "response.completed":
                        response_obj = getattr(event, "response", None)
                        if response_obj:
                            usage_data = self._extract_responses_token_usage(response_obj)

            self._track_token_usage_internal(usage_data)

            # Handle function calls if present
            if function_calls and available_functions:
                for call_data in function_calls.values():
                    function_name = call_data["name"]
                    arguments = call_data["arguments"]

                    if not function_name or not arguments:
                        continue

                    if function_name not in available_functions:
                        logging.warning(
                            f"Function '{function_name}' not found in available functions"
                        )
                        continue

                    try:
                        function_args = json.loads(arguments)
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse streamed function arguments: {e}")
                        continue

                    result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if result is not None:
                        return result

            full_response = self._apply_stop_words(full_response)

            # Handle structured output
            if response_model:
                try:
                    parsed_object = response_model.model_validate_json(full_response)
                    structured_json = parsed_object.model_dump_json()
                    self._emit_call_completed_event(
                        response=structured_json,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                    )
                    return structured_json
                except Exception as e:
                    logging.warning(f"Failed to parse structured output: {e}")

            self._emit_call_completed_event(
                response=full_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
            )

            return full_response

        except Exception as e:
            if is_context_length_exceeded(e):
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API streaming failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def _ahandle_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
        messages: list[LLMMessage] | None = None,
    ) -> str | Any:
        """Handle non-streaming async response from Responses API.

        Args:
            params: API call parameters
            available_functions: Available functions for tool calling
            from_task: Task context
            from_agent: Agent context
            response_model: Response model for structured output
            messages: Original messages for event emission

        Returns:
            Response text or structured output
        """
        try:
            response = await self.async_client.responses.create(**params)

            # Store raw response and extract reasoning content
            self._last_raw_response = response
            self._last_reasoning_content = self._extract_reasoning_content(response)

            usage = self._extract_responses_token_usage(response)
            self._track_token_usage_internal(usage)

            # Check for tool calls
            if hasattr(response, "output") and response.output:
                for output_item in response.output:
                    if hasattr(output_item, "type") and output_item.type == "function_call":
                        if available_functions:
                            function_name = output_item.name
                            try:
                                function_args = json.loads(output_item.arguments)
                            except json.JSONDecodeError as e:
                                logging.error(f"Failed to parse function arguments: {e}")
                                function_args = {}

                            result = self._handle_tool_execution(
                                function_name=function_name,
                                function_args=function_args,
                                available_functions=available_functions,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                            if result is not None:
                                return result

            content = getattr(response, "output_text", "") or ""
            content = self._apply_stop_words(content)

            if response_model:
                try:
                    structured_result = self._validate_structured_output(
                        content, response_model
                    )
                    if isinstance(structured_result, BaseModel):
                        structured_json = structured_result.model_dump_json()
                        self._emit_call_completed_event(
                            response=structured_json,
                            call_type=LLMCallType.LLM_CALL,
                            from_task=from_task,
                            from_agent=from_agent,
                            messages=messages,
                        )
                        return structured_json
                except ValueError as e:
                    logging.warning(f"Structured output validation failed: {e}")

            self._emit_call_completed_event(
                response=content,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
                messages=messages,
            )

            return content

        except NotFoundError as e:
            error_msg = f"Model {self.model} not found: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ValueError(error_msg) from e
        except APIConnectionError as e:
            error_msg = f"Failed to connect to OpenAI API: {e}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise ConnectionError(error_msg) from e
        except Exception as e:
            if is_context_length_exceeded(e):
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API call failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    async def _ahandle_streaming_response(
        self,
        params: dict[str, Any],
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        """Handle streaming async response from Responses API.

        Args:
            params: API call parameters
            available_functions: Available functions for tool calling
            from_task: Task context
            from_agent: Agent context
            response_model: Response model for structured output

        Returns:
            Complete response text
        """
        full_response = ""
        function_calls: dict[str, dict[str, str]] = {}
        usage_data = {"total_tokens": 0}

        try:
            async with await self.async_client.responses.create(**params) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)

                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            full_response += delta
                            self._emit_stream_chunk_event(
                                chunk=delta,
                                from_task=from_task,
                                from_agent=from_agent,
                            )

                    elif event_type == "response.function_call_arguments.delta":
                        call_id = getattr(event, "call_id", "default")
                        if call_id not in function_calls:
                            function_calls[call_id] = {
                                "name": "",
                                "arguments": "",
                            }
                        delta = getattr(event, "delta", "")
                        if delta:
                            function_calls[call_id]["arguments"] += delta

                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item and getattr(item, "type", None) == "function_call":
                            call_id = getattr(item, "call_id", "default")
                            if call_id not in function_calls:
                                function_calls[call_id] = {
                                    "name": getattr(item, "name", ""),
                                    "arguments": "",
                                }
                            else:
                                function_calls[call_id]["name"] = getattr(item, "name", "")

                    elif event_type == "response.completed":
                        response_obj = getattr(event, "response", None)
                        if response_obj:
                            usage_data = self._extract_responses_token_usage(response_obj)

            self._track_token_usage_internal(usage_data)

            if function_calls and available_functions:
                for call_data in function_calls.values():
                    function_name = call_data["name"]
                    arguments = call_data["arguments"]

                    if not function_name or not arguments:
                        continue

                    if function_name not in available_functions:
                        continue

                    try:
                        function_args = json.loads(arguments)
                    except json.JSONDecodeError:
                        continue

                    result = self._handle_tool_execution(
                        function_name=function_name,
                        function_args=function_args,
                        available_functions=available_functions,
                        from_task=from_task,
                        from_agent=from_agent,
                    )

                    if result is not None:
                        return result

            full_response = self._apply_stop_words(full_response)

            if response_model:
                try:
                    parsed_object = response_model.model_validate_json(full_response)
                    structured_json = parsed_object.model_dump_json()
                    self._emit_call_completed_event(
                        response=structured_json,
                        call_type=LLMCallType.LLM_CALL,
                        from_task=from_task,
                        from_agent=from_agent,
                    )
                    return structured_json
                except Exception as e:
                    logging.warning(f"Failed to parse structured output: {e}")

            self._emit_call_completed_event(
                response=full_response,
                call_type=LLMCallType.LLM_CALL,
                from_task=from_task,
                from_agent=from_agent,
            )

            return full_response

        except Exception as e:
            if is_context_length_exceeded(e):
                raise LLMContextLengthExceededError(str(e)) from e

            error_msg = f"OpenAI Responses API streaming failed: {e!s}"
            logging.error(error_msg)
            self._emit_call_failed_event(
                error=error_msg, from_task=from_task, from_agent=from_agent
            )
            raise

    def _extract_responses_token_usage(self, response: Any) -> dict[str, Any]:
        """Extract token usage from Responses API response.

        Args:
            response: The API response object

        Returns:
            Dictionary with token usage information
        """
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0)
                or (getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)),
            }
        return {"total_tokens": 0}

    def supports_function_calling(self) -> bool:
        """Check if the model supports function calling."""
        # O-series models have limited function calling support
        return not self.is_o_series_model

    def supports_stop_words(self) -> bool:
        """Check if the model supports stop words.

        Note: Responses API doesn't have a direct stop parameter,
        but stop words can be applied post-processing.
        """
        return True

    def get_context_window_size(self) -> int:
        """Get the context window size for the model."""
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        # Context window sizes for OpenAI models
        # Ordered from most specific to least specific for prefix matching
        context_windows = [
            ("gpt-4o-mini", 200000),
            ("gpt-4o", 128000),
            ("gpt-4-turbo", 128000),
            ("gpt-4.1-mini", 1047576),
            ("gpt-4.1-nano", 1047576),
            ("gpt-4.1", 1047576),
            ("gpt-4", 8192),
            ("o1-preview", 128000),
            ("o1-mini", 128000),
            ("o3-mini", 200000),
            ("o4-mini", 200000),
        ]

        for model_prefix, size in context_windows:
            if self.model.startswith(model_prefix):
                return int(size * CONTEXT_WINDOW_USAGE_RATIO)

        return int(8192 * CONTEXT_WINDOW_USAGE_RATIO)

