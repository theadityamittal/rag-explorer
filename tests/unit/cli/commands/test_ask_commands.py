"""
Unit tests for CLI ask command functionality.

Tests the ask command behavior, interactive session management,
parameter handling, and integration with the UnifiedAskSession.
Since ask commands are implemented in main.py, these tests focus
on the ask command's behavior and options.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from tests.base import BaseCLITest

from support_deflect_bot.cli.main import cli, ask


class TestAskCommandInterface(BaseCLITest):
    """Test ask command interface and parameter handling."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.fixture
    def mock_services(self):
        """Mock all required services for ask command."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    # Setup mocks
                    mock_rag = Mock()
                    mock_query = Mock()
                    mock_session = Mock()
                    
                    mock_get_rag.return_value = mock_rag
                    mock_get_query.return_value = mock_query
                    mock_session_class.return_value = mock_session
                    
                    yield {
                        'rag': mock_rag,
                        'query': mock_query,
                        'session': mock_session,
                        'session_class': mock_session_class
                    }
                    
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_basic_invocation(self, cli_runner, mock_services):
        """Test basic ask command invocation."""
        result = cli_runner.invoke(cli, ["ask"])
        
        # Command should execute successfully
        assert result.exit_code == 0
        
        # UnifiedAskSession should be created and started
        mock_services['session_class'].assert_called_once()
        mock_services['session'].start.assert_called_once()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_with_domain_filter(self, cli_runner, mock_services):
        """Test ask command with domain filtering."""
        result = cli_runner.invoke(cli, ["ask", "--domains", "example.com,test.org"])
        
        assert result.exit_code == 0
        
        # Check session was created with correct domain filter
        call_args = mock_services['session_class'].call_args
        assert call_args[1]['domain_filter'] == ["example.com", "test.org"]
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_with_confidence_override(self, cli_runner, mock_services):
        """Test ask command with confidence threshold override."""
        result = cli_runner.invoke(cli, ["ask", "--confidence", "0.8"])
        
        assert result.exit_code == 0
        
        # Check session was created with confidence override
        call_args = mock_services['session_class'].call_args
        assert call_args[1]['confidence_override'] == 0.8
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_with_max_chunks(self, cli_runner, mock_services):
        """Test ask command with max chunks override."""
        result = cli_runner.invoke(cli, ["ask", "--max-chunks", "15"])
        
        assert result.exit_code == 0
        
        # Check session was created with max chunks override
        call_args = mock_services['session_class'].call_args
        assert call_args[1]['max_chunks_override'] == 15
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_verbose_mode(self, cli_runner, mock_services):
        """Test ask command in verbose mode."""
        result = cli_runner.invoke(cli, ["--verbose", "ask"])
        
        assert result.exit_code == 0
        
        # Check session was created with verbose flag
        call_args = mock_services['session_class'].call_args
        assert call_args[1]['verbose'] is True
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_quiet_mode(self, cli_runner, mock_services):
        """Test ask command in quiet mode."""
        result = cli_runner.invoke(cli, ["--quiet", "ask"])
        
        assert result.exit_code == 0
        
        # Check session was created with quiet flag
        call_args = mock_services['session_class'].call_args
        assert call_args[1]['quiet'] is True
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_combined_options(self, cli_runner, mock_services):
        """Test ask command with multiple combined options."""
        result = cli_runner.invoke(cli, [
            "--verbose", 
            "ask", 
            "--domains", "docs.example.com",
            "--confidence", "0.75",
            "--max-chunks", "20"
        ])
        
        assert result.exit_code == 0
        
        # Verify all options were passed correctly
        call_args = mock_services['session_class'].call_args
        assert call_args[1]['domain_filter'] == ["docs.example.com"]
        assert call_args[1]['confidence_override'] == 0.75
        assert call_args[1]['max_chunks_override'] == 20
        assert call_args[1]['verbose'] is True


class TestAskCommandSessionManagement(BaseCLITest):
    """Test ask command session lifecycle management."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_session_initialization(self, cli_runner):
        """Test proper session initialization."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    mock_rag = Mock()
                    mock_query = Mock()
                    mock_session = Mock()
                    
                    mock_get_rag.return_value = mock_rag
                    mock_get_query.return_value = mock_query
                    mock_session_class.return_value = mock_session
                    
                    result = cli_runner.invoke(cli, ["ask"])
                    
                    assert result.exit_code == 0
                    
                    # Verify services were retrieved
                    mock_get_rag.assert_called_once()
                    mock_get_query.assert_called_once()
                    
                    # Verify session was initialized with correct engines
                    call_args = mock_session_class.call_args
                    assert call_args[1]['rag_engine'] is mock_rag
                    assert call_args[1]['query_service'] is mock_query
                    
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_keyboard_interrupt_handling(self, cli_runner):
        """Test ask command handles KeyboardInterrupt gracefully."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    mock_session = Mock()
                    mock_session.start.side_effect = KeyboardInterrupt()
                    mock_session_class.return_value = mock_session
                    
                    result = cli_runner.invoke(cli, ["ask"])
                    
                    # Should handle KeyboardInterrupt gracefully
                    assert result.exit_code == 0
                    assert "Session interrupted" in result.output
                    assert "Goodbye" in result.output
                    
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_session_exception_handling(self, cli_runner):
        """Test ask command handles session exceptions."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    mock_session = Mock()
                    mock_session.start.side_effect = Exception("Session failed")
                    mock_session_class.return_value = mock_session
                    
                    result = cli_runner.invoke(cli, ["ask"])
                    
                    # Should handle exceptions and exit with error
                    assert result.exit_code != 0
                    assert "Session failed" in result.output
                    
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_verbose_exception_handling(self, cli_runner):
        """Test ask command shows stack trace in verbose mode on error."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    mock_session = Mock()
                    mock_session.start.side_effect = Exception("Detailed session error")
                    mock_session_class.return_value = mock_session
                    
                    result = cli_runner.invoke(cli, ["--verbose", "ask"])
                    
                    # Should show stack trace in verbose mode
                    assert result.exit_code != 0
                    assert "Session failed" in result.output
                    # Stack trace content may vary but should contain more details


class TestAskCommandParameterValidation(BaseCLITest):
    """Test ask command parameter validation and edge cases."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_domain_parsing(self, cli_runner):
        """Test domain list parsing from comma-separated string."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    mock_session_class.return_value = Mock()
                    
                    # Test various domain formats
                    test_cases = [
                        ("example.com", ["example.com"]),
                        ("example.com,test.org", ["example.com", "test.org"]),
                        ("example.com, test.org, docs.site.com", ["example.com", "test.org", "docs.site.com"]),
                        ("example.com,test.org,", ["example.com", "test.org", ""])  # Trailing comma
                    ]
                    
                    for domain_input, expected_list in test_cases:
                        result = cli_runner.invoke(cli, ["ask", "--domains", domain_input])
                        
                        assert result.exit_code == 0
                        call_args = mock_session_class.call_args
                        actual_domains = call_args[1]['domain_filter']
                        
                        # Strip whitespace from expected and actual
                        expected_stripped = [d.strip() for d in expected_list if d.strip()]
                        assert actual_domains == expected_stripped
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_confidence_validation(self, cli_runner):
        """Test confidence parameter validation."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    mock_session_class.return_value = Mock()
                    
                    # Test valid confidence values
                    valid_confidence_values = [0.0, 0.5, 0.75, 1.0]
                    
                    for confidence in valid_confidence_values:
                        result = cli_runner.invoke(cli, ["ask", "--confidence", str(confidence)])
                        
                        assert result.exit_code == 0
                        call_args = mock_session_class.call_args
                        assert call_args[1]['confidence_override'] == confidence
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_max_chunks_validation(self, cli_runner):
        """Test max chunks parameter validation."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    mock_session_class.return_value = Mock()
                    
                    # Test valid max chunks values
                    valid_chunk_values = [1, 5, 10, 20, 50]
                    
                    for chunks in valid_chunk_values:
                        result = cli_runner.invoke(cli, ["ask", "--max-chunks", str(chunks)])
                        
                        assert result.exit_code == 0
                        call_args = mock_session_class.call_args
                        assert call_args[1]['max_chunks_override'] == chunks
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_no_domains_none_value(self, cli_runner):
        """Test ask command without domains results in None filter."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    mock_session_class.return_value = Mock()
                    
                    result = cli_runner.invoke(cli, ["ask"])
                    
                    assert result.exit_code == 0
                    call_args = mock_session_class.call_args
                    assert call_args[1]['domain_filter'] is None


class TestAskCommandIntegration(BaseCLITest):
    """Test ask command integration with engine services."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_engine_service_integration(self, cli_runner):
        """Test ask command properly integrates with engine services."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    # Create realistic engine mocks
                    mock_rag = Mock()
                    mock_rag.answer_question.return_value = {
                        "answer": "Test answer",
                        "confidence": 0.8,
                        "citations": []
                    }
                    
                    mock_query = Mock()
                    mock_query.preprocess_query.return_value = {
                        "content": "processed query",
                        "keywords": ["test"]
                    }
                    
                    mock_session = Mock()
                    
                    mock_get_rag.return_value = mock_rag
                    mock_get_query.return_value = mock_query
                    mock_session_class.return_value = mock_session
                    
                    result = cli_runner.invoke(cli, ["ask"])
                    
                    assert result.exit_code == 0
                    
                    # Verify session was created with real engine instances
                    call_args = mock_session_class.call_args
                    assert call_args[1]['rag_engine'] is mock_rag
                    assert call_args[1]['query_service'] is mock_query
                    
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_console_integration(self, cli_runner):
        """Test ask command properly integrates with Rich console."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    with patch('support_deflect_bot.cli.main.console') as mock_console:
                        
                        mock_session_class.return_value = Mock()
                        
                        result = cli_runner.invoke(cli, ["ask"])
                        
                        assert result.exit_code == 0
                        
                        # Verify session was created with console instance
                        call_args = mock_session_class.call_args
                        assert call_args[1]['console'] is mock_console
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_settings_integration(self, cli_runner):
        """Test ask command properly uses settings for defaults."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    with patch('support_deflect_bot.cli.main.MAX_CHUNKS', 25) as mock_max_chunks:
                        
                        mock_session_class.return_value = Mock()
                        
                        result = cli_runner.invoke(cli, ["ask"])
                        
                        assert result.exit_code == 0
                        
                        # Verify session uses default MAX_CHUNKS setting
                        call_args = mock_session_class.call_args
                        assert call_args[1]['max_chunks_override'] == 25


class TestAskCommandUsabilityFeatures(BaseCLITest):
    """Test ask command usability and user experience features."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_help_text(self, cli_runner):
        """Test ask command help text is informative."""
        result = cli_runner.invoke(cli, ["ask", "--help"])
        
        assert result.exit_code == 0
        assert "interactive Q&A session" in result.output.lower()
        assert "unified RAG engine" in result.output.lower()
        assert "--domains" in result.output
        assert "--confidence" in result.output
        assert "--max-chunks" in result.output
        
    @pytest.mark.unit
    @pytest.mark.cli  
    def test_ask_command_option_descriptions(self, cli_runner):
        """Test ask command option descriptions are clear."""
        result = cli_runner.invoke(cli, ["ask", "--help"])
        
        assert result.exit_code == 0
        
        # Check for descriptive option help text
        help_text = result.output.lower()
        assert "domain" in help_text and "filter" in help_text
        assert "confidence" in help_text and "threshold" in help_text
        assert "chunks" in help_text
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_command_default_behavior(self, cli_runner):
        """Test ask command default behavior without options."""
        with patch('support_deflect_bot.cli.main.get_rag_engine'):
            with patch('support_deflect_bot.cli.main.get_query_service'):
                with patch('support_deflect_bot.cli.main.UnifiedAskSession') as mock_session_class:
                    
                    mock_session_class.return_value = Mock()
                    
                    result = cli_runner.invoke(cli, ["ask"])
                    
                    assert result.exit_code == 0
                    
                    # Check default parameter values
                    call_args = mock_session_class.call_args
                    assert call_args[1]['domain_filter'] is None
                    assert call_args[1]['confidence_override'] is None
                    assert call_args[1]['verbose'] is False
                    assert call_args[1]['quiet'] is False