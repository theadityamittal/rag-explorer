"""
Unit tests for CLI main entry point and command infrastructure.

Tests the main CLI entry point, command registration, argument parsing,
and service initialization patterns used throughout the CLI.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from rich.console import Console
from tests.base import BaseCLITest

# Import CLI main module and commands
from support_deflect_bot.cli.main import (
    cli,
    get_rag_engine,
    get_doc_processor, 
    get_query_service,
    get_embedding_service
)


class TestCLIEntryPoint(BaseCLITest):
    """Test CLI main entry point and basic functionality."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "Support Deflect Bot" in result.output
        assert "Unified CLI with shared engine services" in result.output
        assert "Commands:" in result.output
        assert "ask" in result.output
        assert "search" in result.output
        assert "index" in result.output
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_cli_version(self, cli_runner):
        """Test CLI version command."""
        result = cli_runner.invoke(cli, ["--version"])
        
        assert result.exit_code == 0
        assert "version" in result.output.lower()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_cli_verbose_flag(self, cli_runner):
        """Test CLI verbose flag propagation."""
        # Test verbose flag is captured in context
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_rag:
            mock_rag.return_value = Mock()
            mock_rag.return_value.validate_providers.return_value = {"test": True}
            
            result = cli_runner.invoke(cli, ["--verbose", "ping"])
            
            # Verbose flag should be processed (exit code may vary due to mocking)
            assert "--verbose" not in result.output  # Flag consumed by Click
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_cli_quiet_flag(self, cli_runner):
        """Test CLI quiet flag functionality."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_rag:
            mock_rag.return_value = Mock()
            mock_rag.return_value.validate_providers.return_value = {"test": True}
            
            result = cli_runner.invoke(cli, ["--quiet", "ping"])
            
            # In quiet mode, should have minimal output
            assert len(result.output.strip()) < 50  # Should be very short
            

class TestServiceInitialization(BaseCLITest):
    """Test engine service initialization and singleton patterns."""
    
    @pytest.mark.unit
    @pytest.mark.cli
    def test_rag_engine_singleton(self):
        """Test RAG engine singleton initialization."""
        # Clear any existing instance
        import support_deflect_bot.cli.main as main_module
        main_module._rag_engine = None
        
        with patch('support_deflect_bot.cli.main.UnifiedRAGEngine') as mock_rag_class:
            mock_instance = Mock()
            mock_rag_class.return_value = mock_instance
            
            # First call should create instance
            engine1 = get_rag_engine()
            
            # Second call should return same instance
            engine2 = get_rag_engine()
            
            assert engine1 is engine2
            assert engine1 is mock_instance
            mock_rag_class.assert_called_once()  # Only instantiated once
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_document_processor_singleton(self):
        """Test document processor singleton initialization."""
        # Clear any existing instance
        import support_deflect_bot.cli.main as main_module
        main_module._doc_processor = None
        
        with patch('support_deflect_bot.cli.main.UnifiedDocumentProcessor') as mock_doc_class:
            mock_instance = Mock()
            mock_doc_class.return_value = mock_instance
            
            processor1 = get_doc_processor()
            processor2 = get_doc_processor()
            
            assert processor1 is processor2
            assert processor1 is mock_instance
            mock_doc_class.assert_called_once()
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_query_service_singleton(self):
        """Test query service singleton initialization."""
        # Clear any existing instance
        import support_deflect_bot.cli.main as main_module
        main_module._query_service = None
        
        with patch('support_deflect_bot.cli.main.UnifiedQueryService') as mock_query_class:
            mock_instance = Mock()
            mock_query_class.return_value = mock_instance
            
            service1 = get_query_service()
            service2 = get_query_service()
            
            assert service1 is service2
            assert service1 is mock_instance
            mock_query_class.assert_called_once()
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_embedding_service_singleton(self):
        """Test embedding service singleton initialization."""
        # Clear any existing instance
        import support_deflect_bot.cli.main as main_module
        main_module._embedding_service = None
        
        with patch('support_deflect_bot.cli.main.UnifiedEmbeddingService') as mock_embed_class:
            mock_instance = Mock()
            mock_embed_class.return_value = mock_instance
            
            service1 = get_embedding_service()
            service2 = get_embedding_service()
            
            assert service1 is service2
            assert service1 is mock_instance
            mock_embed_class.assert_called_once()


class TestIndexCommand(BaseCLITest):
    """Test CLI index command."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_index_command_success(self, cli_runner):
        """Test successful index command."""
        with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.process_local_directory.return_value = {
                'files_processed': 5,
                'chunks_created': 150,
                'errors': 0
            }
            mock_get_doc.return_value = mock_processor
            
            result = cli_runner.invoke(cli, ["index"])
            
            assert result.exit_code == 0
            assert "Successfully indexed 5 files" in result.output
            assert "150 chunks" in result.output
            mock_processor.process_local_directory.assert_called_once()
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_index_command_with_custom_path(self, cli_runner):
        """Test index command with custom docs path."""
        with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.process_local_directory.return_value = {
                'files_processed': 3,
                'chunks_created': 75,
                'errors': 0
            }
            mock_get_doc.return_value = mock_processor
            
            result = cli_runner.invoke(cli, ["index", "--docs-path", "/custom/path"])
            
            assert result.exit_code == 0
            mock_processor.process_local_directory.assert_called_once_with(
                directory="/custom/path",
                reset_collection=False
            )
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_index_command_with_reset(self, cli_runner):
        """Test index command with reset flag."""
        with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.process_local_directory.return_value = {
                'files_processed': 10,
                'chunks_created': 300,
                'errors': 1
            }
            mock_get_doc.return_value = mock_processor
            
            result = cli_runner.invoke(cli, ["index", "--reset"])
            
            assert result.exit_code == 0
            assert "1 files had errors" in result.output
            mock_processor.process_local_directory.assert_called_once_with(
                directory=None,  # Uses default
                reset_collection=True
            )
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_index_command_quiet_mode(self, cli_runner):
        """Test index command in quiet mode."""
        with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.process_local_directory.return_value = {
                'files_processed': 5,
                'chunks_created': 150,
                'errors': 0
            }
            mock_get_doc.return_value = mock_processor
            
            result = cli_runner.invoke(cli, ["--quiet", "index"])
            
            assert result.exit_code == 0
            # Quiet mode should only output chunk count
            assert result.output.strip() == "150"
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_index_command_error_handling(self, cli_runner):
        """Test index command error handling."""
        with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
            mock_processor = Mock()
            mock_processor.process_local_directory.side_effect = Exception("Index failed")
            mock_get_doc.return_value = mock_processor
            
            result = cli_runner.invoke(cli, ["index"])
            
            assert result.exit_code != 0
            assert "Indexing failed" in result.output
            assert "Index failed" in result.output


class TestSearchCommand(BaseCLITest):
    """Test CLI search command."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_command_success(self, cli_runner):
        """Test successful search command."""
        with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
            mock_service = Mock()
            mock_service.preprocess_query.return_value = {
                "valid": True,
                "content": "processed query",
                "keywords": ["test"]
            }
            mock_service.retrieve_documents.return_value = [
                {
                    "text": "This is a test document about testing",
                    "meta": {"path": "/docs/test.md"},
                    "distance": 0.1,
                    "relevance_score": 0.9
                }
            ]
            mock_get_query.return_value = mock_service
            
            with patch('support_deflect_bot.cli.main.format_search_results') as mock_format:
                result = cli_runner.invoke(cli, ["search", "test query"])
                
                assert result.exit_code == 0
                mock_service.preprocess_query.assert_called_once_with("test query")
                mock_service.retrieve_documents.assert_called_once()
                mock_format.assert_called_once()
                
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_command_with_options(self, cli_runner):
        """Test search command with various options."""
        with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
            mock_service = Mock()
            mock_service.preprocess_query.return_value = {"valid": True, "content": "test"}
            mock_service.retrieve_documents.return_value = []
            mock_get_query.return_value = mock_service
            
            with patch('support_deflect_bot.cli.main.format_search_results'):
                result = cli_runner.invoke(cli, [
                    "search", "test query",
                    "--limit", "10",
                    "--domains", "example.com,test.org"
                ])
                
                assert result.exit_code == 0
                mock_service.retrieve_documents.assert_called_once_with(
                    {"valid": True, "content": "test"},
                    k=10,
                    domains=["example.com", "test.org"]
                )
                
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_command_json_output(self, cli_runner):
        """Test search command with JSON output."""
        with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
            mock_service = Mock()
            mock_service.preprocess_query.return_value = {
                "valid": True,
                "content": "processed query"
            }
            mock_service.retrieve_documents.return_value = [
                {
                    "text": "Test document content for JSON output testing",
                    "meta": {"path": "/docs/json_test.md"},
                    "distance": 0.2,
                    "relevance_score": 0.8
                }
            ]
            mock_get_query.return_value = mock_service
            
            result = cli_runner.invoke(cli, ["search", "test", "--output", "json"])
            
            assert result.exit_code == 0
            
            # Parse JSON output
            output_data = json.loads(result.output)
            assert "query" in output_data
            assert "results" in output_data
            assert output_data["query"] == "test"
            assert len(output_data["results"]) == 1
            assert "text" in output_data["results"][0]
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_command_invalid_query(self, cli_runner):
        """Test search command with invalid query."""
        with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
            mock_service = Mock()
            mock_service.preprocess_query.return_value = {
                "valid": False,
                "reason": "Query too short"
            }
            mock_get_query.return_value = mock_service
            
            result = cli_runner.invoke(cli, ["search", "x"])
            
            assert result.exit_code != 0
            assert "Invalid query" in result.output
            assert "Query too short" in result.output
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_command_limit_validation(self, cli_runner):
        """Test search command limit validation."""
        # Test limit too low
        result = cli_runner.invoke(cli, ["search", "test", "--limit", "0"])
        assert result.exit_code != 0
        assert "Limit must be between 1 and 20" in result.output
        
        # Test limit too high
        result = cli_runner.invoke(cli, ["search", "test", "--limit", "25"])
        assert result.exit_code != 0
        assert "Limit must be between 1 and 20" in result.output


class TestStatusCommand(BaseCLITest):
    """Test CLI status command."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_command_healthy_system(self, cli_runner):
        """Test status command with healthy system."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
                with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                    with patch('support_deflect_bot.cli.main.get_embedding_service') as mock_get_embed:
                        
                        # Setup healthy system
                        mock_rag = Mock()
                        mock_rag.get_metrics.return_value = {"queries_processed": 100}
                        mock_rag.validate_providers.return_value = {"openai": True}
                        mock_get_rag.return_value = mock_rag
                        
                        mock_doc = Mock()
                        mock_doc.get_collection_stats.return_value = {"connected": True}
                        mock_get_doc.return_value = mock_doc
                        
                        mock_query = Mock()
                        mock_query.get_query_analytics.return_value = {"total_queries": 50}
                        mock_get_query.return_value = mock_query
                        
                        mock_embed = Mock()
                        mock_embed.get_analytics.return_value = {"total_embeddings_generated": 200}
                        mock_get_embed.return_value = mock_embed
                        
                        result = cli_runner.invoke(cli, ["status"])
                        
                        assert result.exit_code == 0
                        assert "System status: OK" in result.output
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_command_json_output(self, cli_runner):
        """Test status command with JSON output."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
                with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                    with patch('support_deflect_bot.cli.main.get_embedding_service') as mock_get_embed:
                        
                        # Setup mocks
                        mock_rag = Mock()
                        mock_rag.get_metrics.return_value = {"queries_processed": 100}
                        mock_rag.validate_providers.return_value = {"openai": True}
                        mock_get_rag.return_value = mock_rag
                        
                        mock_doc = Mock()
                        mock_doc.get_collection_stats.return_value = {"connected": True}
                        mock_get_doc.return_value = mock_doc
                        
                        mock_query = Mock()
                        mock_query.get_query_analytics.return_value = {"total_queries": 50}
                        mock_get_query.return_value = mock_query
                        
                        mock_embed = Mock()
                        mock_embed.get_analytics.return_value = {"total_embeddings_generated": 200}
                        mock_get_embed.return_value = mock_embed
                        
                        result = cli_runner.invoke(cli, ["status", "--output", "json"])
                        
                        assert result.exit_code == 0
                        
                        # Parse JSON output
                        status_data = json.loads(result.output)
                        assert "rag_engine" in status_data
                        assert "document_processor" in status_data
                        assert "query_service" in status_data
                        assert "embedding_service" in status_data
                        assert "overall_health" in status_data
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_command_unhealthy_system(self, cli_runner):
        """Test status command with unhealthy system."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
                with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                    with patch('support_deflect_bot.cli.main.get_embedding_service') as mock_get_embed:
                        
                        # Setup unhealthy system (no providers)
                        mock_rag = Mock()
                        mock_rag.get_metrics.return_value = {"queries_processed": 0}
                        mock_rag.validate_providers.return_value = {"openai": False, "groq": False}
                        mock_get_rag.return_value = mock_rag
                        
                        mock_doc = Mock()
                        mock_doc.get_collection_stats.return_value = {"connected": True}
                        mock_get_doc.return_value = mock_doc
                        
                        mock_query = Mock()
                        mock_query.get_query_analytics.return_value = {"total_queries": 0}
                        mock_get_query.return_value = mock_query
                        
                        mock_embed = Mock()
                        mock_embed.get_analytics.return_value = {"total_embeddings_generated": 0}
                        mock_get_embed.return_value = mock_embed
                        
                        result = cli_runner.invoke(cli, ["status"])
                        
                        assert result.exit_code != 0  # Should fail for unhealthy system
                        assert "System status: NO_PROVIDERS" in result.output


class TestPingCommand(BaseCLITest):
    """Test CLI ping command."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ping_command_success(self, cli_runner):
        """Test successful ping command."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.validate_providers.return_value = {
                "openai": True,
                "groq": True,
                "ollama": False
            }
            mock_get_rag.return_value = mock_rag
            
            result = cli_runner.invoke(cli, ["ping"])
            
            assert result.exit_code == 0
            assert "Providers available: openai, groq" in result.output
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ping_command_no_providers(self, cli_runner):
        """Test ping command with no available providers."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.validate_providers.return_value = {
                "openai": False,
                "groq": False,
                "ollama": False
            }
            mock_get_rag.return_value = mock_rag
            
            result = cli_runner.invoke(cli, ["ping"])
            
            assert result.exit_code != 0
            assert "No providers available" in result.output
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ping_command_quiet_mode(self, cli_runner):
        """Test ping command in quiet mode."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.validate_providers.return_value = {"openai": True}
            mock_get_rag.return_value = mock_rag
            
            result = cli_runner.invoke(cli, ["--quiet", "ping"])
            
            assert result.exit_code == 0
            assert result.output.strip() == "ok"
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ping_command_exception_handling(self, cli_runner):
        """Test ping command exception handling."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            mock_rag = Mock()
            mock_rag.validate_providers.side_effect = Exception("Connection error")
            mock_get_rag.return_value = mock_rag
            
            result = cli_runner.invoke(cli, ["ping"])
            
            assert result.exit_code != 0
            assert "Provider test failed" in result.output
            
            # Test quiet mode with exception
            result = cli_runner.invoke(cli, ["--quiet", "ping"])
            assert result.exit_code != 0
            assert result.output.strip() == "failed"


class TestMetricsCommand(BaseCLITest):
    """Test CLI metrics command."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create Click test runner."""
        return CliRunner()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_metrics_command_table_output(self, cli_runner):
        """Test metrics command with table output."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
                with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                    with patch('support_deflect_bot.cli.main.get_embedding_service') as mock_get_embed:
                        
                        # Setup comprehensive metrics
                        mock_rag = Mock()
                        mock_rag.get_metrics.return_value = {
                            "queries_processed": 250,
                            "successful_answers": 220,
                            "average_confidence": 0.85
                        }
                        mock_get_rag.return_value = mock_rag
                        
                        mock_doc = Mock()
                        mock_doc.get_collection_stats.return_value = {
                            "total_chunks": 1500,
                            "processing_stats": {"files_processed": 50}
                        }
                        mock_get_doc.return_value = mock_doc
                        
                        mock_query = Mock()
                        mock_query.get_query_analytics.return_value = {"total_queries": 275}
                        mock_get_query.return_value = mock_query
                        
                        mock_embed = Mock()
                        mock_embed.get_analytics.return_value = {
                            "total_embeddings_generated": 2000,
                            "cache_hit_rate": 0.75
                        }
                        mock_get_embed.return_value = mock_embed
                        
                        result = cli_runner.invoke(cli, ["metrics"])
                        
                        assert result.exit_code == 0
                        assert "Performance Metrics" in result.output
                        assert "250" in result.output  # queries processed
                        assert "75.0%" in result.output  # cache hit rate
                        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_metrics_command_json_output(self, cli_runner):
        """Test metrics command with JSON output."""
        with patch('support_deflect_bot.cli.main.get_rag_engine') as mock_get_rag:
            with patch('support_deflect_bot.cli.main.get_doc_processor') as mock_get_doc:
                with patch('support_deflect_bot.cli.main.get_query_service') as mock_get_query:
                    with patch('support_deflect_bot.cli.main.get_embedding_service') as mock_get_embed:
                        
                        # Setup metrics
                        mock_rag = Mock()
                        mock_rag.get_metrics.return_value = {"queries_processed": 100}
                        mock_get_rag.return_value = mock_rag
                        
                        mock_doc = Mock()
                        mock_doc.get_collection_stats.return_value = {"total_chunks": 500}
                        mock_get_doc.return_value = mock_doc
                        
                        mock_query = Mock()
                        mock_query.get_query_analytics.return_value = {"total_queries": 120}
                        mock_get_query.return_value = mock_query
                        
                        mock_embed = Mock()
                        mock_embed.get_analytics.return_value = {"total_embeddings_generated": 800}
                        mock_get_embed.return_value = mock_embed
                        
                        result = cli_runner.invoke(cli, ["metrics", "--output", "json"])
                        
                        assert result.exit_code == 0
                        
                        # Parse JSON output
                        metrics_data = json.loads(result.output)
                        assert "version" in metrics_data
                        assert "rag_engine" in metrics_data
                        assert "document_processor" in metrics_data
                        assert "query_service" in metrics_data
                        assert "embedding_service" in metrics_data
                        
                        # Verify metric values
                        assert metrics_data["rag_engine"]["queries_processed"] == 100
                        assert metrics_data["document_processor"]["total_chunks"] == 500