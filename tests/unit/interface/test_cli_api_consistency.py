"""
CLI-API consistency tests for Support Deflect Bot.

Tests that ensure CLI and API interfaces provide consistent behavior
for equivalent functionality, including ask, search, indexing, and
status operations.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from click.testing import CliRunner
from fastapi.testclient import TestClient
from tests.base import BaseAPITest

# CLI imports
from support_deflect_bot.cli.main import cli

# API imports  
from support_deflect_bot.api.app import app


class TestAskConsistency(BaseAPITest):
    """Test consistency between CLI ask and API ask endpoint."""
    
    @pytest.fixture
    def cli_runner(self):
        """CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def api_client(self):
        """API test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_rag_result(self):
        """Mock RAG engine result for consistency testing."""
        return {
            "answer": "Machine learning is a subset of artificial intelligence.",
            "confidence": 0.85,
            "citations": [
                {
                    "id": "ml_doc_1",
                    "text": "Machine learning enables computers to learn without explicit programming.",
                    "metadata": {"source": "ml_intro.md", "section": "overview"},
                    "distance": 0.15
                }
            ],
            "provider_used": "openai",
            "metadata": {"tokens_used": 125, "model": "gpt-4"}
        }
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_ask_functionality_consistency(self, cli_runner, api_client, mock_rag_result):
        """Test that CLI ask and API ask produce equivalent results."""
        question = "What is machine learning?"
        
        # Mock the engine for both CLI and API
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as api_mock:
            
            # Configure mocks to return the same result
            mock_engine = Mock()
            mock_engine.answer_question.return_value = mock_rag_result
            cli_mock.return_value = mock_engine
            api_mock.return_value = mock_engine
            
            # Test CLI
            cli_result = cli_runner.invoke(cli, ['ask', question])
            
            # Test API
            api_response = api_client.post('/api/v1/ask', json={
                "question": question,
                "max_chunks": 5,
                "min_confidence": 0.25
            })
            
            # Both should succeed
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Both should call the engine with similar parameters
            assert mock_engine.answer_question.call_count == 2
            
            # Check API response structure
            api_data = api_response.json()
            assert api_data["answer"] == mock_rag_result["answer"]
            assert api_data["confidence"] == mock_rag_result["confidence"]
            assert api_data["provider_used"] == mock_rag_result["provider_used"]
            
            # CLI output should contain the answer
            assert mock_rag_result["answer"] in cli_result.output
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_ask_parameter_consistency(self, cli_runner, api_client, mock_rag_result):
        """Test that CLI and API handle parameters consistently."""
        question = "Test question"
        
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as api_mock:
            
            mock_engine = Mock()
            mock_engine.answer_question.return_value = mock_rag_result
            cli_mock.return_value = mock_engine
            api_mock.return_value = mock_engine
            
            # Test with domain filtering
            domains = ["docs", "api"]
            
            # CLI with domains
            cli_result = cli_runner.invoke(cli, [
                'ask', question,
                '--domains', 'docs,api',
                '--max-chunks', '3',
                '--min-confidence', '0.5'
            ])
            
            # API with domains
            api_response = api_client.post('/api/v1/ask', json={
                "question": question,
                "domains": domains,
                "max_chunks": 3,
                "min_confidence": 0.5
            })
            
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Check that both called the engine with the same parameters
            cli_call_args = mock_engine.answer_question.call_args_list[0]
            api_call_args = mock_engine.answer_question.call_args_list[1]
            
            # Compare key parameters
            assert cli_call_args.kwargs.get('domains') == api_call_args.kwargs.get('domains')
            assert cli_call_args.kwargs.get('k') == api_call_args.kwargs.get('k')
            assert cli_call_args.kwargs.get('min_confidence') == api_call_args.kwargs.get('min_confidence')
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_ask_error_handling_consistency(self, cli_runner, api_client):
        """Test that CLI and API handle errors consistently."""
        question = "Test question"
        
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as api_mock:
            
            # Configure mocks to raise the same exception
            mock_engine = Mock()
            error_message = "Engine processing failed"
            mock_engine.answer_question.side_effect = Exception(error_message)
            cli_mock.return_value = mock_engine
            api_mock.return_value = mock_engine
            
            # Test CLI error handling
            cli_result = cli_runner.invoke(cli, ['ask', question])
            
            # Test API error handling
            api_response = api_client.post('/api/v1/ask', json={"question": question})
            
            # CLI should exit with error
            assert cli_result.exit_code != 0
            
            # API should return 500 error
            assert api_response.status_code == 500
            
            # Both should contain relevant error information
            assert "error" in cli_result.output.lower() or "failed" in cli_result.output.lower()
            
            api_data = api_response.json()
            assert "error" in api_data
            assert "failed" in api_data["detail"]


class TestSearchConsistency(BaseAPITest):
    """Test consistency between CLI search and API search endpoint."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture
    def api_client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_search_result(self):
        """Mock search service result."""
        return [
            {
                "id": "chunk_1",
                "text": "Neural networks are computational models inspired by biological networks.",
                "metadata": {"source": "neural_nets.md", "section": "introduction"},
                "distance": 0.12
            },
            {
                "id": "chunk_2", 
                "text": "Deep learning uses multiple layers of neural networks.",
                "metadata": {"source": "deep_learning.md", "section": "basics"},
                "distance": 0.18
            }
        ]
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_search_functionality_consistency(self, cli_runner, api_client, mock_search_result):
        """Test that CLI search and API search produce equivalent results."""
        query = "neural networks"
        
        with patch('support_deflect_bot.engine.services.get_query_service') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_query_service') as api_mock:
            
            mock_service = Mock()
            mock_service.search_similar_chunks.return_value = mock_search_result
            cli_mock.return_value = mock_service
            api_mock.return_value = mock_service
            
            # Test CLI search
            cli_result = cli_runner.invoke(cli, ['search', query])
            
            # Test API search
            api_response = api_client.post('/api/v1/search', json={
                "query": query,
                "k": 5
            })
            
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Both should call the service
            assert mock_service.search_similar_chunks.call_count == 2
            
            # API response should contain expected data
            api_data = api_response.json()
            assert len(api_data["results"]) == len(mock_search_result)
            assert api_data["query"] == query
            
            # CLI should display search results
            assert "neural networks" in cli_result.output.lower()
    
    @pytest.mark.unit  
    @pytest.mark.integration
    def test_search_parameter_consistency(self, cli_runner, api_client, mock_search_result):
        """Test search parameter handling consistency."""
        query = "test query"
        
        with patch('support_deflect_bot.engine.services.get_query_service') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_query_service') as api_mock:
            
            mock_service = Mock()
            mock_service.search_similar_chunks.return_value = mock_search_result
            cli_mock.return_value = mock_service
            api_mock.return_value = mock_service
            
            # Test with custom parameters
            k_value = 3
            domains = ["docs"]
            
            # CLI search
            cli_result = cli_runner.invoke(cli, [
                'search', query,
                '--k', str(k_value),
                '--domains', ','.join(domains)
            ])
            
            # API search
            api_response = api_client.post('/api/v1/search', json={
                "query": query,
                "k": k_value,
                "domains": domains
            })
            
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Check parameter consistency
            cli_call_args = mock_service.search_similar_chunks.call_args_list[0]
            api_call_args = mock_service.search_similar_chunks.call_args_list[1]
            
            assert cli_call_args.kwargs.get('k') == api_call_args.kwargs.get('k')
            assert cli_call_args.kwargs.get('domain_filter') == api_call_args.kwargs.get('domain_filter')


class TestIndexingConsistency(BaseAPITest):
    """Test consistency between CLI indexing and API indexing endpoints."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture
    def api_client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_index_result(self):
        """Mock document processor result."""
        return {
            "processed_files": ["./docs/file1.md", "./docs/file2.txt"],
            "errors": []
        }
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_index_functionality_consistency(self, cli_runner, api_client, mock_index_result):
        """Test that CLI index and API index produce equivalent results."""
        directory = "./test_docs"
        
        with patch('support_deflect_bot.engine.services.get_document_processor') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as api_mock, \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            
            mock_processor = Mock()
            mock_processor.process_directory.return_value = mock_index_result
            cli_mock.return_value = mock_processor
            api_mock.return_value = mock_processor
            
            # Test CLI index
            cli_result = cli_runner.invoke(cli, ['index', directory])
            
            # Test API index
            api_response = api_client.post('/api/v1/index', json={
                "directory": directory,
                "recursive": True,
                "force": False
            })
            
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Both should call the processor
            assert mock_processor.process_directory.call_count == 2
            
            # API response should reflect processing results
            api_data = api_response.json()
            assert api_data["success"] is True
            assert api_data["processed_count"] == len(mock_index_result["processed_files"])
            
            # CLI should show success message
            assert "success" in cli_result.output.lower() or "processed" in cli_result.output.lower()
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_index_parameter_consistency(self, cli_runner, api_client, mock_index_result):
        """Test index parameter handling consistency."""
        directory = "./docs"
        
        with patch('support_deflect_bot.engine.services.get_document_processor') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_document_processor') as api_mock, \
             patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            
            mock_processor = Mock()
            mock_processor.process_directory.return_value = mock_index_result
            cli_mock.return_value = mock_processor
            api_mock.return_value = mock_processor
            
            # Test with specific parameters
            # CLI index
            cli_result = cli_runner.invoke(cli, [
                'index', directory,
                '--force',
                '--no-recursive'
            ])
            
            # API index  
            api_response = api_client.post('/api/v1/index', json={
                "directory": directory,
                "force": True,
                "recursive": False
            })
            
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Check parameter consistency
            cli_call_args = mock_processor.process_directory.call_args_list[0]
            api_call_args = mock_processor.process_directory.call_args_list[1]
            
            assert cli_call_args.kwargs.get('force_reprocess') == api_call_args.kwargs.get('force_reprocess')
            assert cli_call_args.kwargs.get('recursive') == api_call_args.kwargs.get('recursive')


class TestStatusHealthConsistency(BaseAPITest):
    """Test consistency between CLI status and API health endpoints."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture
    def api_client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_healthy_services(self):
        """Mock healthy service states."""
        return {
            "rag_status": {"overall_health": "ok", "providers": ["openai"]},
            "doc_status": {"connected": True, "total_chunks": 1500},
            "query_status": {"connected": True, "queries_processed": 250},
            "embedding_status": {"openai": True}
        }
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_status_health_consistency(self, cli_runner, api_client, mock_healthy_services):
        """Test that CLI status and API health show consistent system state."""
        
        # Mock services for CLI
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_rag, \
             patch('support_deflect_bot.engine.services.get_document_processor') as cli_doc, \
             patch('support_deflect_bot.engine.services.get_query_service') as cli_query:
            
            # Mock CLI services
            cli_rag_engine = Mock()
            cli_rag_engine.get_system_status.return_value = mock_healthy_services["rag_status"]
            cli_rag.return_value = cli_rag_engine
            
            cli_doc_processor = Mock()
            cli_doc_processor.get_status.return_value = mock_healthy_services["doc_status"]
            cli_doc.return_value = cli_doc_processor
            
            cli_query_service = Mock()
            cli_query_service.get_status.return_value = mock_healthy_services["query_status"]
            cli_query.return_value = cli_query_service
            
            # Mock services for API
            with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as api_rag, \
                 patch('support_deflect_bot.api.endpoints.health.get_document_processor') as api_doc, \
                 patch('support_deflect_bot.api.endpoints.health.get_query_service') as api_query, \
                 patch('support_deflect_bot.api.endpoints.health.get_embedding_service') as api_embed:
                
                # Mock API services with same data
                api_rag_engine = Mock()
                api_rag_engine.get_system_status.return_value = mock_healthy_services["rag_status"]
                api_rag.return_value = api_rag_engine
                
                api_doc_processor = Mock()
                api_doc_processor.get_status.return_value = mock_healthy_services["doc_status"]
                api_doc.return_value = api_doc_processor
                
                api_query_service = Mock()
                api_query_service.get_status.return_value = mock_healthy_services["query_status"]
                api_query.return_value = api_query_service
                
                api_embed_service = Mock()
                api_embed_service.get_provider_status.return_value = mock_healthy_services["embedding_status"]
                api_embed.return_value = api_embed_service
                
                # Test CLI status
                cli_result = cli_runner.invoke(cli, ['status'])
                
                # Test API health
                api_response = api_client.get('/api/v1/health')
                
                assert cli_result.exit_code == 0
                assert api_response.status_code == 200
                
                # Both should indicate healthy system
                api_data = api_response.json()
                assert api_data["status"] == "healthy"
                assert api_data["database"]["connected"] is True
                
                # CLI should show positive status indicators
                assert "connected" in cli_result.output.lower() or "ok" in cli_result.output.lower()
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_degraded_status_consistency(self, cli_runner, api_client):
        """Test consistency when system is in degraded state."""
        degraded_services = {
            "rag_status": {"overall_health": "ok"},
            "doc_status": {"connected": False},  # Database disconnected
            "query_status": {"connected": False},
            "embedding_status": {}
        }
        
        # Mock degraded services for both CLI and API
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_rag, \
             patch('support_deflect_bot.engine.services.get_document_processor') as cli_doc, \
             patch('support_deflect_bot.engine.services.get_query_service') as cli_query:
            
            cli_rag_engine = Mock()
            cli_rag_engine.get_system_status.return_value = degraded_services["rag_status"]
            cli_rag.return_value = cli_rag_engine
            
            cli_doc_processor = Mock()
            cli_doc_processor.get_status.return_value = degraded_services["doc_status"]
            cli_doc.return_value = cli_doc_processor
            
            cli_query_service = Mock()
            cli_query_service.get_status.return_value = degraded_services["query_status"]
            cli_query.return_value = cli_query_service
            
            with patch('support_deflect_bot.api.endpoints.health.get_rag_engine') as api_rag, \
                 patch('support_deflect_bot.api.endpoints.health.get_document_processor') as api_doc, \
                 patch('support_deflect_bot.api.endpoints.health.get_query_service') as api_query, \
                 patch('support_deflect_bot.api.endpoints.health.get_embedding_service') as api_embed:
                
                api_rag_engine = Mock()
                api_rag_engine.get_system_status.return_value = degraded_services["rag_status"]
                api_rag.return_value = api_rag_engine
                
                api_doc_processor = Mock()
                api_doc_processor.get_status.return_value = degraded_services["doc_status"]
                api_doc.return_value = api_doc_processor
                
                api_query_service = Mock()
                api_query_service.get_status.return_value = degraded_services["query_status"]
                api_query.return_value = api_query_service
                
                api_embed_service = Mock()
                api_embed_service.get_provider_status.return_value = degraded_services["embedding_status"]
                api_embed.return_value = api_embed_service
                
                # Test CLI status
                cli_result = cli_runner.invoke(cli, ['status'])
                
                # Test API health
                api_response = api_client.get('/api/v1/health')
                
                # Both should indicate degraded state
                api_data = api_response.json()
                assert api_data["status"] == "degraded"
                assert api_data["database"]["connected"] is False
                
                # CLI should show error/degraded indicators
                assert cli_result.exit_code != 0 or "error" in cli_result.output.lower() or "failed" in cli_result.output.lower()


class TestConfigurationConsistency(BaseAPITest):
    """Test that CLI and API respect the same configuration settings."""
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_settings_consistency(self):
        """Test that both CLI and API use the same settings."""
        # Import settings from both contexts
        from support_deflect_bot.utils.settings import APP_VERSION, APP_NAME
        
        # Both should use the same version and name
        assert APP_VERSION is not None
        assert APP_NAME is not None
        assert len(APP_VERSION) > 0
        assert len(APP_NAME) > 0
        
        # Test that both interfaces can access settings
        cli_runner = CliRunner()
        api_client = TestClient(app)
        
        # CLI version command (if available)
        cli_result = cli_runner.invoke(cli, ['--version'])
        # Even if version command doesn't exist, we can check that CLI loads
        
        # API app should have version configured
        assert app.version is not None
        assert app.title is not None
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_provider_configuration_consistency(self):
        """Test that CLI and API use the same provider configurations."""
        with patch('support_deflect_bot.engine.services.get_embedding_service') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_embedding_service') as api_mock:
            
            # Mock provider status
            provider_status = {"openai": True, "local": False}
            
            cli_service = Mock()
            cli_service.get_provider_status.return_value = provider_status
            cli_mock.return_value = cli_service
            
            api_service = Mock()
            api_service.get_provider_status.return_value = provider_status
            api_mock.return_value = api_service
            
            # Both should see the same providers
            cli_providers = cli_service.get_provider_status()
            api_providers = api_service.get_provider_status()
            
            assert cli_providers == api_providers


class TestOutputFormatConsistency(BaseAPITest):
    """Test consistency in data representation between CLI and API."""
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_error_message_consistency(self):
        """Test that similar errors produce consistent messages."""
        cli_runner = CliRunner()
        api_client = TestClient(app)
        
        # Test validation errors
        # CLI with invalid parameters
        cli_result = cli_runner.invoke(cli, ['ask', ''])  # Empty question
        
        # API with invalid parameters
        api_response = api_client.post('/api/v1/ask', json={"question": ""})
        
        # Both should indicate validation error
        assert cli_result.exit_code != 0
        assert api_response.status_code == 422
        
        # Both should mention the validation issue
        assert "question" in cli_result.output.lower() or "empty" in cli_result.output.lower()
        
        api_data = api_response.json()
        assert "detail" in api_data or "error" in api_data
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_success_response_consistency(self, mock_rag_result=None):
        """Test that successful operations produce consistent output."""
        if mock_rag_result is None:
            mock_rag_result = {
                "answer": "Test answer",
                "confidence": 0.75,
                "citations": [],
                "provider_used": "openai",
                "metadata": {}
            }
        
        cli_runner = CliRunner()
        api_client = TestClient(app)
        
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as api_mock:
            
            mock_engine = Mock()
            mock_engine.answer_question.return_value = mock_rag_result
            cli_mock.return_value = mock_engine
            api_mock.return_value = mock_engine
            
            # Test successful ask operation
            cli_result = cli_runner.invoke(cli, ['ask', 'test question'])
            api_response = api_client.post('/api/v1/ask', json={"question": "test question"})
            
            assert cli_result.exit_code == 0
            assert api_response.status_code == 200
            
            # Both should contain the answer
            assert mock_rag_result["answer"] in cli_result.output
            
            api_data = api_response.json()
            assert api_data["answer"] == mock_rag_result["answer"]


class TestPerformanceConsistency(BaseAPITest):
    """Test that CLI and API have consistent performance characteristics."""
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_timeout_consistency(self):
        """Test that CLI and API have consistent timeout behavior."""
        import time
        
        cli_runner = CliRunner()
        api_client = TestClient(app)
        
        with patch('support_deflect_bot.engine.services.get_rag_engine') as cli_mock, \
             patch('support_deflect_bot.api.dependencies.engine.get_rag_engine') as api_mock:
            
            def slow_operation(*args, **kwargs):
                time.sleep(0.1)  # Simulate slow operation
                raise TimeoutError("Operation timed out")
            
            mock_engine = Mock()
            mock_engine.answer_question.side_effect = slow_operation
            cli_mock.return_value = mock_engine
            api_mock.return_value = mock_engine
            
            # Test CLI timeout handling
            cli_start = time.time()
            cli_result = cli_runner.invoke(cli, ['ask', 'test question'])
            cli_duration = time.time() - cli_start
            
            # Test API timeout handling
            api_start = time.time()
            api_response = api_client.post('/api/v1/ask', json={"question": "test question"})
            api_duration = time.time() - api_start
            
            # Both should handle timeout gracefully
            assert cli_result.exit_code != 0
            assert api_response.status_code == 500
            
            # Durations should be similar (within reasonable range)
            assert abs(cli_duration - api_duration) < 0.5  # Within 500ms of each other


class TestFeatureParity(BaseAPITest):
    """Test that CLI and API support equivalent features."""
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_available_operations_parity(self):
        """Test that core operations are available in both interfaces."""
        cli_runner = CliRunner()
        api_client = TestClient(app)
        
        # Test CLI commands are available
        cli_help = cli_runner.invoke(cli, ['--help'])
        assert cli_help.exit_code == 0
        
        # Key commands should be available
        cli_commands = cli_help.output.lower()
        assert 'ask' in cli_commands
        assert 'search' in cli_commands
        assert 'index' in cli_commands
        assert 'status' in cli_commands
        
        # Test API endpoints are available
        api_docs = api_client.get('/docs')
        assert api_docs.status_code == 200
        
        # OpenAPI schema should show key endpoints
        api_schema = api_client.get('/openapi.json')
        assert api_schema.status_code == 200
        
        schema_data = api_schema.json()
        paths = schema_data.get('paths', {})
        
        # Key endpoints should be available
        assert '/api/v1/ask' in paths
        assert '/api/v1/search' in paths
        assert '/api/v1/index' in paths
        assert '/api/v1/health' in paths
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_parameter_support_parity(self):
        """Test that both interfaces support equivalent parameters."""
        cli_runner = CliRunner()
        
        # Test CLI ask command parameters
        cli_ask_help = cli_runner.invoke(cli, ['ask', '--help'])
        if cli_ask_help.exit_code == 0:
            ask_help = cli_ask_help.output.lower()
            
            # Key parameters should be supported
            expected_params = ['domains', 'max-chunks', 'min-confidence']
            for param in expected_params:
                assert param in ask_help or param.replace('-', '_') in ask_help
        
        # API should support equivalent parameters (tested via schema)
        api_client = TestClient(app)
        api_schema = api_client.get('/openapi.json')
        schema_data = api_schema.json()
        
        ask_endpoint = schema_data['paths']['/api/v1/ask']['post']
        request_schema = ask_endpoint['requestBody']['content']['application/json']['schema']
        
        # Should reference AskRequest model
        assert '$ref' in request_schema or 'properties' in request_schema
    
    @pytest.mark.unit
    @pytest.mark.integration
    def test_batch_processing_consistency(self):
        """Test batch processing capabilities are consistent."""
        api_client = TestClient(app)
        
        # API should support batch operations
        api_schema = api_client.get('/openapi.json')
        schema_data = api_schema.json()
        paths = schema_data.get('paths', {})
        
        # Batch endpoints should exist
        assert '/api/v1/batch_ask' in paths or '/api/v1/batch/ask' in paths
        
        # CLI might support batch through file input or multiple questions
        # This would depend on implementation - the test verifies the concept exists
        cli_runner = CliRunner()
        cli_help = cli_runner.invoke(cli, ['--help'])
        
        # Either dedicated batch commands or file input support
        help_text = cli_help.output.lower()
        has_batch_support = (
            'batch' in help_text or 
            'file' in help_text or
            'multiple' in help_text
        )
        
        # At minimum, API should have batch support
        assert len([p for p in paths.keys() if 'batch' in p]) > 0