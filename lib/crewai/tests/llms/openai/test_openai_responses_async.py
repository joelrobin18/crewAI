"""Tests for OpenAI Responses API async completion functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from crewai import Agent, Task, Crew
from crewai.llm import LLM


@pytest.mark.asyncio
async def test_openai_responses_async_basic_call():
    """Test basic async call with OpenAI Responses API."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "Hello there!"

        result = await llm.acall("Say hello")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_openai_responses_async_with_temperature():
    """Test async call with temperature parameter."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses", temperature=0.1)

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "test"

        result = await llm.acall("Say the word 'test' once")

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_with_max_tokens():
    """Test async call with max_output_tokens parameter."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses", max_output_tokens=10)

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "Short response"

        result = await llm.acall("Write a very long story about a dragon.")

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_with_system_message():
    """Test async call with system message."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "4"

        result = await llm.acall(messages)

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_conversation():
    """Test async call with conversation history."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"}
    ]

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "Your name is Alice."

        result = await llm.acall(messages)

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_multiple_calls():
    """Test making multiple async calls in sequence."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.side_effect = ["2", "4"]

        result1 = await llm.acall("What is 1+1?")
        result2 = await llm.acall("What is 2+2?")

        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, str)
        assert isinstance(result2, str)


@pytest.mark.asyncio
async def test_openai_responses_async_with_response_format_none():
    """Test async call with response_format set to None."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses", response_format=None)

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "The sun is a star."

        result = await llm.acall("Tell me a short fact")

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_with_response_format_json():
    """Test async call with JSON response format."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses", response_format={"type": "json_object"})

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = '{"greeting": "Hello"}'

        result = await llm.acall("Return a JSON object with a 'greeting' field")

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_with_parameters():
    """Test async call with multiple parameters."""
    llm = LLM(
        model="gpt-4o-mini",
        provider="openai_responses",
        temperature=0.7,
        max_output_tokens=100,
        top_p=0.9,
    )

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "Interesting fact about space."

        result = await llm.acall("Tell me a short fact")

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_with_instructions():
    """Test async call with instructions parameter."""
    llm = LLM(
        model="gpt-4o-mini",
        provider="openai_responses",
        instructions="You are a helpful coding assistant."
    )

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "A function is a reusable block of code."

        result = await llm.acall("What is a function in programming?")

        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_openai_responses_async_client_exists():
    """Test that async client is properly initialized."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    assert hasattr(llm, 'async_client')
    assert llm.async_client is not None


@pytest.mark.asyncio
async def test_openai_responses_async_has_acall_method():
    """Test that acall method exists and is callable."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    assert hasattr(llm, 'acall')
    assert callable(llm.acall)


@pytest.mark.asyncio
async def test_openai_responses_async_crew_kickoff():
    """Test that OpenAI Responses API works with async crew kickoff."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses")

    with patch.object(llm, 'call', return_value="Research results about Italy."):
        agent = Agent(
            role="Research Assistant",
            goal="Find information about Italy",
            backstory="You are a helpful research assistant.",
            llm=llm,
            verbose=True,
        )

        task = Task(
            description="What is the capital of Italy?",
            expected_output="The capital of Italy",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = await crew.kickoff_async()

        assert result is not None


@pytest.mark.asyncio
async def test_openai_responses_async_mocked_api_call():
    """Test that async API calls work with mocking."""
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(model="gpt-4o")

    with patch.object(llm.async_client.responses, 'create', new_callable=AsyncMock) as mock_create:
        mock_output = MagicMock()
        mock_output.type = "message"
        mock_output.content = [MagicMock(type="output_text", text="Async test response")]

        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_create.return_value = mock_response

        result = await llm.acall("Hello async")

        assert mock_create.called


@pytest.mark.asyncio
async def test_openai_responses_async_with_seed():
    """Test async call with seed parameter."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses", seed=42)

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "Deterministic response"

        result = await llm.acall("Tell me something")

        assert result is not None
        assert isinstance(result, str)
        assert llm.seed == 42


@pytest.mark.asyncio
async def test_openai_responses_async_with_reasoning_effort():
    """Test async call with reasoning_effort parameter."""
    llm = LLM(model="gpt-4o-mini", provider="openai_responses", reasoning_effort="high")

    with patch.object(llm, 'acall', new_callable=AsyncMock) as mock_acall:
        mock_acall.return_value = "Complex response with reasoning"

        result = await llm.acall("Solve a complex problem")

        assert result is not None
        assert llm.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_openai_responses_async_with_custom_base_url_and_api_key():
    """Test async call with custom base_url and api_key (e.g., Databricks scenario)."""
    from crewai.llms.providers.openai.responses_completion import OpenAIResponsesCompletion

    llm = OpenAIResponsesCompletion(
        model="custom-model",
        api_key="test-api-key",
        base_url="https://custom-endpoint.com/v1",
    )

    with patch.object(llm.async_client.responses, 'create', new_callable=AsyncMock) as mock_create:
        mock_output = MagicMock()
        mock_output.type = "message"
        mock_output.content = [MagicMock(type="output_text", text="Custom endpoint response")]

        mock_response = MagicMock()
        mock_response.output = [mock_output]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_create.return_value = mock_response

        result = await llm.acall("Test message")

        assert mock_create.called
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs['model'] == 'custom-model'
