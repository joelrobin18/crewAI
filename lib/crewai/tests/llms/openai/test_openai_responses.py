import os
import sys
import types
from unittest.mock import patch, MagicMock
import pytest

from crewai.llm import LLM
from crewai.crew import Crew
from crewai.agent import Agent
from crewai.task import Task


def test_openai_responses_completion_is_used_when_openai_responses_provider():
    """
    Test that OpenAIResponsesCompletion is used when LLM uses provider 'openai_responses'
    """
    llm = LLM(model="gpt-4o", provider="openai_responses")

    assert llm.__class__.__name__ == "OpenAIResponsesCompletion"
    assert llm.provider == "openai_responses"
    assert llm.model == "gpt-4o"


def test_openai_responses_completion_is_used_when_model_has_prefix():
    """
    Test that OpenAIResponsesCompletion is used when model has openai_responses/ prefix
    """
    llm = LLM(model="openai_responses/gpt-4o")

    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.provider == "openai_responses"
    assert llm.model == "gpt-4o"


def test_openai_responses_completion_module_is_imported():
    """
    Test that the responses_completion module is properly imported when using openai_responses provider
    """
    module_name = "crewai.llms.providers.openai.responses_completion"

    if module_name in sys.modules:
        del sys.modules[module_name]

    LLM(model="gpt-4o", provider="openai_responses")

    assert module_name in sys.modules
    completion_mod = sys.modules[module_name]
    assert isinstance(completion_mod, types.ModuleType)
    assert hasattr(completion_mod, 'OpenAIResponsesCompletion')


def test_openai_responses_raises_error_when_initialization_fails():
    """
    Test that LLM raises ImportError when native OpenAI Responses completion fails to initialize.
    """
    with patch('crewai.llm.LLM._get_native_provider') as mock_get_provider:

        class FailingCompletion:
            def __init__(self, *args, **kwargs):
                raise Exception("Native OpenAI Responses SDK failed")

        mock_get_provider.return_value = FailingCompletion

        with pytest.raises(ImportError) as excinfo:
            LLM(model="gpt-4o", provider="openai_responses")

        assert "Error importing native provider" in str(excinfo.value)
        assert "Native OpenAI Responses SDK failed" in str(excinfo.value)


def test_openai_responses_completion_initialization_parameters():
    """
    Test that OpenAIResponsesCompletion is initialized with correct parameters
    """
    llm = LLM(
        model="gpt-4o",
        provider="openai_responses",
        temperature=0.7,
        max_output_tokens=1000,
        api_key="test-key"
    )

    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 1000


def test_openai_responses_specific_parameters():
    """
    Test OpenAI Responses-specific parameters like instructions and reasoning_effort
    """
    llm = LLM(
        model="gpt-4o",
        provider="openai_responses",
        instructions="You are a helpful assistant.",
        reasoning_effort="high",
        stream=True,
        max_retries=5,
        timeout=60
    )

    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.instructions == "You are a helpful assistant."
    assert llm.reasoning_effort == "high"
    assert llm.stream == True


def test_openai_responses_completion_call():
    """
    Test that OpenAIResponsesCompletion call method works
    """
    llm = LLM(model="openai_responses/gpt-4o")

    with patch.object(llm, 'call', return_value="Hello! I'm ready to help.") as mock_call:
        result = llm.call("Hello, how are you?")

        assert result == "Hello! I'm ready to help."
        mock_call.assert_called_once_with("Hello, how are you?")


def test_openai_responses_completion_called_during_crew_execution():
    """
    Test that OpenAIResponsesCompletion.call is actually invoked when running a crew
    """
    openai_llm = LLM(model="gpt-4o", provider="openai_responses")

    with patch.object(openai_llm, 'call', return_value="Tokyo has 14 million people.") as mock_call:
        agent = Agent(
            role="Research Assistant",
            goal="Find population info",
            backstory="You research populations.",
            llm=openai_llm,
        )

        task = Task(
            description="Find Tokyo population",
            expected_output="Population number",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert mock_call.called
        assert "14 million" in str(result)


def test_openai_responses_completion_call_arguments():
    """
    Test that OpenAIResponsesCompletion.call is invoked with correct arguments
    """
    openai_llm = LLM(model="gpt-4o", provider="openai_responses")

    with patch.object(openai_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed successfully."

        agent = Agent(
            role="Test Agent",
            goal="Complete a simple task",
            backstory="You are a test agent.",
            llm=openai_llm
        )

        task = Task(
            description="Say hello world",
            expected_output="Hello world",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        assert mock_call.called

        call_args = mock_call.call_args
        assert call_args is not None

        messages = call_args[0][0]
        assert isinstance(messages, (str, list))

        if isinstance(messages, str):
            assert "hello world" in messages.lower()
        elif isinstance(messages, list):
            message_content = str(messages).lower()
            assert "hello world" in message_content


def test_multiple_openai_responses_calls_in_crew():
    """
    Test that OpenAIResponsesCompletion.call is invoked multiple times for multiple tasks
    """
    openai_llm = LLM(model="gpt-4o", provider="openai_responses")

    with patch.object(openai_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent = Agent(
            role="Multi-task Agent",
            goal="Complete multiple tasks",
            backstory="You can handle multiple tasks.",
            llm=openai_llm
        )

        task1 = Task(
            description="First task",
            expected_output="First result",
            agent=agent,
        )

        task2 = Task(
            description="Second task",
            expected_output="Second result",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task1, task2]
        )
        crew.kickoff()

        assert mock_call.call_count >= 2

        for call in mock_call.call_args_list:
            assert len(call[0]) > 0
            messages = call[0][0]
            assert messages is not None


def test_openai_responses_completion_with_tools():
    """
    Test that OpenAIResponsesCompletion.call is invoked with tools when agent has tools
    """
    from crewai.tools import tool

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing"""
        return f"Tool result for: {query}"

    openai_llm = LLM(model="gpt-4o", provider="openai_responses")

    with patch.object(openai_llm, 'call') as mock_call:
        mock_call.return_value = "Task completed with tools."

        agent = Agent(
            role="Tool User",
            goal="Use tools to complete tasks",
            backstory="You can use tools.",
            llm=openai_llm,
            tools=[sample_tool]
        )

        task = Task(
            description="Use the sample tool",
            expected_output="Tool usage result",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()

        assert mock_call.called

        call_args = mock_call.call_args
        call_kwargs = call_args[1] if len(call_args) > 1 else {}

        if 'tools' in call_kwargs:
            assert call_kwargs['tools'] is not None
            assert len(call_kwargs['tools']) > 0


def test_openai_responses_supports_function_calling():
    """
    Test that OpenAI Responses API supports function calling
    """
    llm = LLM(model="gpt-4o", provider="openai_responses")
    assert llm.supports_function_calling() == True


def test_openai_responses_supports_stop_words():
    """
    Test that OpenAI Responses API supports stop words
    """
    llm = LLM(model="gpt-4o", provider="openai_responses")
    assert llm.supports_stop_words() == True


def test_openai_responses_context_window_size_gpt4o():
    """
    Test that gpt-4o returns correct context window size
    """
    from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO
    llm = LLM(model="gpt-4o", provider="openai_responses")
    context_size = llm.get_context_window_size()

    expected_size = int(128000 * CONTEXT_WINDOW_USAGE_RATIO)
    assert context_size == expected_size


def test_openai_responses_context_window_size_gpt4o_mini():
    """
    Test that gpt-4o-mini returns correct context window size
    """
    from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")
    context_size = llm.get_context_window_size()

    expected_size = int(200000 * CONTEXT_WINDOW_USAGE_RATIO)
    assert context_size == expected_size


def test_openai_responses_context_window_size_o1():
    """
    Test that o1 models return correct context window size
    """
    from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO
    llm = LLM(model="o1-mini", provider="openai_responses")
    context_size = llm.get_context_window_size()

    expected_size = int(128000 * CONTEXT_WINDOW_USAGE_RATIO)
    assert context_size == expected_size


def test_openai_responses_streaming_parameter():
    """
    Test that streaming parameter is properly handled
    """
    llm_no_stream = LLM(model="gpt-4o", provider="openai_responses", stream=False)
    assert llm_no_stream.stream == False

    llm_stream = LLM(model="gpt-4o", provider="openai_responses", stream=True)
    assert llm_stream.stream == True


def test_openai_responses_client_params_with_base_url():
    """
    Test that _get_client_params correctly handles base_url
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        base_url="https://custom.openai.com/v1",
    )
    client_params = llm._get_client_params()
    assert client_params["base_url"] == "https://custom.openai.com/v1"


def test_openai_responses_client_params_with_env_var():
    """
    Test that _get_client_params uses OPENAI_BASE_URL environment variable as fallback
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.openai.com/v1"}):
        llm = OpenAIResponsesCompletion(model="gpt-4o")
        client_params = llm._get_client_params()
        assert client_params["base_url"] == "https://env.openai.com/v1"


def test_openai_responses_client_params_no_base_url(monkeypatch):
    """
    Test that _get_client_params works correctly when no base_url is specified
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    llm = OpenAIResponsesCompletion(model="gpt-4o")
    client_params = llm._get_client_params()
    assert "base_url" not in client_params or client_params.get("base_url") is None


def test_openai_responses_with_custom_base_url_and_api_key():
    """
    Test that custom base_url and api_key are properly configured (e.g., Databricks scenario)
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    custom_base_url = "https://my-workspace.cloud.databricks.com/serving-endpoints"
    custom_api_key = "dapi-test-token-12345"

    llm = OpenAIResponsesCompletion(
        model="databricks-gpt-5",
        api_key=custom_api_key,
        base_url=custom_base_url,
    )

    client_params = llm._get_client_params()
    assert client_params["base_url"] == custom_base_url
    assert client_params["api_key"] == custom_api_key
    assert llm.model == "databricks-gpt-5"


def test_openai_responses_with_custom_base_url_and_api_key_via_llm_factory():
    """
    Test that custom base_url and api_key work via LLM factory
    """
    custom_base_url = "https://my-workspace.cloud.databricks.com/serving-endpoints"
    custom_api_key = "dapi-test-token-12345"

    llm = LLM(
        model="custom-model",
        provider="openai_responses",
        api_key=custom_api_key,
        base_url=custom_base_url,
    )

    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
    assert isinstance(llm, OpenAIResponsesCompletion)

    client_params = llm._get_client_params()
    assert client_params["base_url"] == custom_base_url
    assert client_params["api_key"] == custom_api_key


def test_openai_responses_mocked_api_call_with_custom_endpoint():
    """
    Test mocked API call with custom base_url and api_key
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="custom-model",
        api_key="test-api-key",
        base_url="https://custom-endpoint.com/v1",
    )

    with patch.object(llm.client.responses, 'create') as mock_create:
        mock_output = MagicMock()
        mock_output.type = "message"
        mock_output.content = [MagicMock(type="output_text", text="Custom endpoint response")]

        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_create.return_value = mock_response

        result = llm.call("Test message")

        assert mock_create.called
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['model'] == 'custom-model'


def test_openai_responses_message_formatting():
    """
    Test that messages are properly formatted for Responses API
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]

    formatted = llm._format_messages(test_messages)

    assert isinstance(formatted, list)
    assert len(formatted) == 4


def test_openai_responses_tool_conversion():
    """
    Test that tools are properly converted for Responses API
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")

    crewai_tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }]

    assert crewai_tools[0]["function"]["name"] == "test_tool"
    assert crewai_tools[0]["function"]["description"] == "A test tool"


def test_openai_responses_environment_variable_api_key():
    """
    Test that OpenAI API key is properly loaded from environment
    """
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
        llm = LLM(model="gpt-4o", provider="openai_responses")

        assert llm.client is not None
        assert hasattr(llm.client, 'responses')


def test_openai_responses_token_usage_extraction():
    """
    Test that token usage is properly extracted from Responses API responses
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")

    mock_response = MagicMock()
    mock_response.usage = MagicMock(
        input_tokens=50,
        output_tokens=25,
        total_tokens=75
    )

    usage = llm._extract_responses_token_usage(mock_response)

    assert usage["prompt_tokens"] == 50
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 75


def test_openai_responses_direct_instance_creation():
    """
    Test that OpenAIResponsesCompletion can be instantiated directly
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        temperature=0.5,
        max_output_tokens=500,
        stream=True
    )

    assert llm.model == "gpt-4o"
    assert llm.temperature == 0.5
    assert llm.max_output_tokens == 500
    assert llm.stream == True
    assert llm.provider == "openai_responses"


def test_openai_responses_with_response_format_pydantic():
    """
    Test that response_format with a Pydantic BaseModel works correctly
    """
    from pydantic import BaseModel, Field

    class TestResponse(BaseModel):
        answer: str = Field(description="The answer")
        confidence: float = Field(description="Confidence score")

    llm = LLM(model="gpt-4o", provider="openai_responses", response_format=TestResponse)

    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.response_format == TestResponse


def test_openai_responses_with_response_format_dict():
    """
    Test that response_format with a dict works correctly
    """
    llm = LLM(
        model="gpt-4o",
        provider="openai_responses",
        response_format={"type": "json_object"}
    )

    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
    assert isinstance(llm, OpenAIResponsesCompletion)
    assert llm.response_format == {"type": "json_object"}


def test_openai_responses_model_detection():
    """
    Test that various OpenAI model formats are properly detected with openai_responses provider
    """
    test_cases = [
        ("openai_responses/gpt-4o", "gpt-4o"),
        ("openai_responses/gpt-4o-mini", "gpt-4o-mini"),
        ("openai_responses/o1-mini", "o1-mini"),
        ("openai_responses/o1-preview", "o1-preview"),
    ]

    for model_name, expected_model in test_cases:
        llm = LLM(model=model_name)
        from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion
        assert isinstance(llm, OpenAIResponsesCompletion), f"Failed for model: {model_name}"
        assert llm.model == expected_model, f"Model mismatch for: {model_name}"


def test_openai_responses_extra_arguments_are_passed():
    """
    Test that extra arguments are passed to OpenAIResponsesCompletion
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        temperature=0.7,
        max_output_tokens=1000,
        top_p=0.5,
        max_retries=3
    )

    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 1000
    assert llm.top_p == 0.5
    assert llm.client.max_retries == 3


def test_openai_responses_client_setup_with_extra_arguments():
    """
    Test that OpenAIResponsesCompletion is initialized with correct client parameters
    """
    llm = LLM(
        model="gpt-4o",
        provider="openai_responses",
        temperature=0.7,
        max_output_tokens=1000,
        top_p=0.5,
        max_retries=3,
        timeout=30
    )

    assert llm.temperature == 0.7
    assert llm.max_output_tokens == 1000
    assert llm.top_p == 0.5
    assert llm.client.max_retries == 3
    assert llm.client.timeout == 30


def test_openai_responses_mocked_api_call():
    """
    Test that OpenAIResponsesCompletion properly makes API calls
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")

    with patch.object(llm.client.responses, 'create') as mock_create:
        mock_output = MagicMock()
        mock_output.type = "message"
        mock_output.content = [MagicMock(type="output_text", text="Test response")]

        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_create.return_value = mock_response

        result = llm.call("Hello")

        assert mock_create.called
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['model'] == 'gpt-4o'


def test_openai_responses_acall_exists():
    """
    Test that OpenAIResponsesCompletion has async call method
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")
    assert hasattr(llm, 'acall')
    assert callable(llm.acall)


def test_openai_responses_instructions_parameter():
    """
    Test that instructions parameter is properly handled
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    instructions = "You are a helpful coding assistant."
    llm = OpenAIResponsesCompletion(model="gpt-4o", instructions=instructions)

    assert llm.instructions == instructions


def test_openai_responses_reasoning_effort_parameter():
    """
    Test that reasoning_effort parameter is properly handled
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o", reasoning_effort="high")
    assert llm.reasoning_effort == "high"

    llm2 = OpenAIResponsesCompletion(model="gpt-4o", reasoning_effort="low")
    assert llm2.reasoning_effort == "low"


def test_openai_responses_seed_parameter():
    """
    Test that seed parameter is properly handled
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o", seed=42)
    assert llm.seed == 42


def test_openai_responses_provider_property():
    """
    Test that provider property returns 'openai_responses'
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")
    assert llm.provider == "openai_responses"


def test_openai_responses_model_property():
    """
    Test that model property returns correct model name
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o-mini")
    assert llm.model == "gpt-4o-mini"


def test_openai_responses_previous_response_id_parameter():
    """
    Test that previous_response_id parameter is properly handled
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        previous_response_id="resp_12345"
    )
    assert llm.previous_response_id == "resp_12345"


def test_openai_responses_distinguishes_from_regular_openai():
    """
    Test that openai_responses provider is distinct from regular openai provider
    """
    from crewai.llms.providers.openai.completion import OpenAICompletion
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    regular_llm = LLM(model="gpt-4o")
    responses_llm = LLM(model="gpt-4o", provider="openai_responses")

    assert isinstance(regular_llm, OpenAICompletion)
    assert isinstance(responses_llm, OpenAIResponsesCompletion)
    assert regular_llm.__class__ != responses_llm.__class__


def test_openai_responses_with_multi_agent_crew():
    """
    Test that OpenAIResponsesCompletion works with multiple agents in a crew
    """
    llm = LLM(model="gpt-4o", provider="openai_responses")

    with patch.object(llm, 'call') as mock_call:
        mock_call.return_value = "Task completed."

        agent1 = Agent(
            role="Researcher",
            goal="Research information",
            backstory="You research things.",
            llm=llm,
        )

        agent2 = Agent(
            role="Writer",
            goal="Write content",
            backstory="You write things.",
            llm=llm,
        )

        task1 = Task(
            description="Research the topic",
            expected_output="Research results",
            agent=agent1,
        )

        task2 = Task(
            description="Write about the topic",
            expected_output="Written content",
            agent=agent2,
        )

        crew = Crew(
            agents=[agent1, agent2],
            tasks=[task1, task2]
        )
        result = crew.kickoff()

        assert mock_call.called
        assert mock_call.call_count >= 2


def test_openai_responses_include_reasoning_parameter():
    """
    Test that include_reasoning parameter is properly handled
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        reasoning_effort="high",
        include_reasoning=True
    )

    assert llm.include_reasoning == True
    assert llm.reasoning_effort == "high"


def test_openai_responses_include_reasoning_without_effort():
    """
    Test that include_reasoning works without reasoning_effort
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        include_reasoning=True
    )

    with patch.object(llm.client.responses, 'create') as mock_create:
        mock_output = MagicMock()
        mock_output.type = "message"
        mock_output.content = [MagicMock(type="output_text", text="Test response")]

        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_create.return_value = mock_response

        llm.call("Test message")

        assert mock_create.called
        call_kwargs = mock_create.call_args[1]
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"]["summary"] == "auto"


def test_openai_responses_include_reasoning_in_params():
    """
    Test that reasoning.summary is added to API call when include_reasoning is True
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        reasoning_effort="high",
        include_reasoning=True
    )

    with patch.object(llm.client.responses, 'create') as mock_create:
        mock_output = MagicMock()
        mock_output.type = "message"
        mock_output.content = [MagicMock(type="output_text", text="Test response")]

        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_create.return_value = mock_response

        llm.call("Test message")

        assert mock_create.called
        call_kwargs = mock_create.call_args[1]
        assert "reasoning" in call_kwargs
        assert call_kwargs["reasoning"]["effort"] == "high"
        assert call_kwargs["reasoning"]["summary"] == "auto"


def test_openai_responses_last_reasoning_content_property():
    """
    Test that last_reasoning_content property is accessible
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")

    assert hasattr(llm, 'last_reasoning_content')
    assert llm.last_reasoning_content == []


def test_openai_responses_extract_reasoning_content():
    """
    Test that reasoning content is extracted from response summary
    """
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="gpt-4o",
        reasoning_effort="high",
        include_reasoning=True
    )

    mock_summary_item = MagicMock()
    mock_summary_item.text = "The model analyzed the problem step by step"

    mock_reasoning = MagicMock()
    mock_reasoning.type = "reasoning"
    mock_reasoning.summary = [mock_summary_item]
    mock_reasoning.content = None
    mock_reasoning.text = None

    mock_message = MagicMock()
    mock_message.type = "message"
    mock_message.content = [MagicMock(type="output_text", text="Test response")]

    mock_response = MagicMock()
    mock_response.output = [mock_reasoning, mock_message]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)

    with patch.object(llm.client.responses, 'create', return_value=mock_response):
        llm.call("Test message")

        assert len(llm.last_reasoning_content) == 1
        assert llm.last_reasoning_content[0] == "The model analyzed the problem step by step"
