"""
Unit tests for CLI output formatting utilities.

Tests output formatting functions including search results display,
answer formatting, metrics tables, and status summaries with various
styling and display options.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from tests.base import BaseCLITest

from support_deflect_bot.cli.output import (
    format_search_results,
    format_answer,
    format_metrics_table,
    format_status_summary
)


class TestSearchResultsFormatting(BaseCLITest):
    """Test search results formatting functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console for testing."""
        return Mock(spec=Console)
        
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results data."""
        return [
            {
                "text": "This is the first search result with detailed information about the topic being searched.",
                "meta": {"path": "/docs/first_result.md"},
                "relevance_score": 0.95,
                "distance": 0.05
            },
            {
                "text": "Second search result contains different but related information about the same topic.",
                "meta": {"path": "/docs/second_result.md"},  
                "relevance_score": 0.87,
                "distance": 0.13
            },
            {
                "text": "Third result with a very long path name to test truncation behavior in the output display system.",
                "meta": {"path": "/very/long/path/to/documentation/files/third_result_with_long_name.md"},
                "relevance_score": 0.72,
                "distance": 0.28
            }
        ]
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_normal_display(self, mock_console, sample_search_results):
        """Test search results display in normal mode."""
        format_search_results(mock_console, "test query", sample_search_results, quiet=False)
        
        # Verify console.print was called with a table
        assert mock_console.print.called
        
        # Get the table that was printed
        call_args = mock_console.print.call_args_list[0]
        printed_table = call_args[0][0]
        
        assert isinstance(printed_table, Table)
        assert "Search Results for: 'test query'" in str(printed_table)
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_quiet_mode(self, mock_console, sample_search_results):
        """Test search results display in quiet mode."""
        format_search_results(mock_console, "test query", sample_search_results, quiet=True)
        
        # In quiet mode, should only print the count
        mock_console.print.assert_called_once_with(3)  # 3 results
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_no_results(self, mock_console):
        """Test search results display with no results."""
        format_search_results(mock_console, "no matches", [], quiet=False)
        
        mock_console.print.assert_called_once_with("No results found.", style="yellow")
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_no_results_quiet(self, mock_console):
        """Test search results display with no results in quiet mode."""
        format_search_results(mock_console, "no matches", [], quiet=True)
        
        mock_console.print.assert_called_once_with(0)
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_path_truncation(self, mock_console):
        """Test long path truncation in search results."""
        long_path_result = [{
            "text": "Test with long path",
            "meta": {"path": "/very/long/path/that/exceeds/the/display/width/limit/file.md"},
            "relevance_score": 0.8,
            "distance": 0.2
        }]
        
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_search_results(mock_console, "long path test", long_path_result)
            
            # Check that add_row was called with truncated path
            mock_table.add_row.assert_called_once()
            call_args = mock_table.add_row.call_args[0]
            
            # The path should be truncated with "..." prefix
            displayed_path = call_args[1]  # Second argument is the path
            assert displayed_path.startswith("...")
            assert len(displayed_path) <= 37  # Max display width
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_preview_truncation(self, mock_console):
        """Test text preview truncation in search results."""
        long_text_result = [{
            "text": "A" * 200,  # 200 characters of text
            "meta": {"path": "/docs/long_text.md"},
            "relevance_score": 0.9
        }]
        
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_search_results(mock_console, "long text test", long_text_result)
            
            # Check that add_row was called with truncated preview
            mock_table.add_row.assert_called_once()
            call_args = mock_table.add_row.call_args[0]
            
            # The preview should be truncated with "..." suffix
            preview = call_args[3]  # Fourth argument is the preview
            assert preview.endswith("...")
            assert len(preview) <= 153  # 150 chars + "..."
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_search_results_relevance_score_formats(self, mock_console):
        """Test different relevance score format handling."""
        varied_results = [
            {
                "text": "Result with relevance score",
                "meta": {"path": "/docs/relevance.md"},
                "relevance_score": 0.85,
                "distance": 0.15
            },
            {
                "text": "Result with distance only",
                "meta": {"path": "/docs/distance.md"},
                "distance": 0.3
            },
            {
                "text": "Result with no scoring",
                "meta": {"path": "/docs/no_score.md"}
            }
        ]
        
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_search_results(mock_console, "varied scoring", varied_results)
            
            # Verify all three results were added
            assert mock_table.add_row.call_count == 3
            
            # Check relevance score calculations
            calls = mock_table.add_row.call_args_list
            
            # First call should have direct relevance score
            relevance1 = calls[0][0][2]
            assert "0.850" in relevance1
            
            # Second call should convert distance to similarity
            relevance2 = calls[1][0][2]
            assert relevance2 != "N/A"  # Should have calculated value
            
            # Third call should show N/A
            relevance3 = calls[2][0][2]
            assert relevance3 == "N/A"


class TestAnswerFormatting(BaseCLITest):
    """Test answer formatting functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console for testing."""
        return Mock(spec=Console)
        
    @pytest.fixture
    def sample_answer_response(self):
        """Create sample answer response data."""
        return {
            "answer": "This is a comprehensive answer to the user's question with detailed explanations.",
            "confidence": 0.85,
            "citations": [
                {
                    "path": "/docs/source1.md",
                    "preview": "Relevant excerpt from the first source document"
                },
                {
                    "path": "/docs/source2.md", 
                    "preview": "Important information from the second source"
                }
            ],
            "metadata": {
                "model": "gpt-4o-mini",
                "processing_time": 2.34,
                "tokens_used": 150
            }
        }
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_answer_formatting_basic(self, mock_console, sample_answer_response):
        """Test basic answer formatting."""
        with patch('support_deflect_bot.cli.output.Panel') as mock_panel_class:
            mock_panel = Mock()
            mock_panel_class.return_value = mock_panel
            
            format_answer(mock_console, sample_answer_response, verbose=False)
            
            # Verify Panel was created with correct parameters
            mock_panel_class.assert_called_once()
            call_args = mock_panel_class.call_args
            
            # Check answer content
            assert "comprehensive answer" in call_args[0][0]
            
            # Check title includes confidence
            assert "Confidence: 0.850" in call_args[1]["title"]
            
            # Check border style based on confidence (0.85 should be green)
            assert call_args[1]["border_style"] == "green"
            
            # Verify panel was printed
            mock_console.print.assert_called_with(mock_panel)
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_answer_formatting_confidence_colors(self, mock_console):
        """Test confidence-based border color selection."""
        test_cases = [
            (0.9, "green"),   # High confidence
            (0.5, "yellow"),  # Medium confidence  
            (0.2, "red")      # Low confidence
        ]
        
        for confidence, expected_color in test_cases:
            response = {
                "answer": "Test answer",
                "confidence": confidence,
                "citations": [],
                "metadata": {}
            }
            
            with patch('support_deflect_bot.cli.output.Panel') as mock_panel_class:
                format_answer(mock_console, response, verbose=False)
                
                call_args = mock_panel_class.call_args
                assert call_args[1]["border_style"] == expected_color
                
    @pytest.mark.unit
    @pytest.mark.cli
    def test_answer_formatting_verbose_citations(self, mock_console, sample_answer_response):
        """Test answer formatting with citations in verbose mode."""
        with patch('support_deflect_bot.cli.output.Panel'):
            with patch('support_deflect_bot.cli.output.Text') as mock_text_class:
                mock_text = Mock()
                mock_text_class.return_value = mock_text
                
                format_answer(mock_console, sample_answer_response, verbose=True)
                
                # Should print citations header
                mock_console.print.assert_any_call("\n📚 Sources:", style="cyan bold")
                
                # Should create Text objects for citations
                assert mock_text_class.call_count >= 1
                
                # Should append citation information
                assert mock_text.append.call_count >= 4  # Multiple append calls per citation
                
    @pytest.mark.unit
    @pytest.mark.cli
    def test_answer_formatting_verbose_metadata(self, mock_console, sample_answer_response):
        """Test answer formatting with metadata in verbose mode."""
        with patch('support_deflect_bot.cli.output.Panel'):
            format_answer(mock_console, sample_answer_response, verbose=True)
            
            # Should print metadata header
            mock_console.print.assert_any_call("\n📋 Metadata:", style="dim")
            
            # Should print each metadata key-value pair
            metadata_calls = [call for call in mock_console.print.call_args_list 
                            if len(call[0]) > 0 and ":" in str(call[0][0])]
            
            # Check that metadata items were printed
            metadata_content = " ".join([str(call[0][0]) for call in metadata_calls])
            assert "model:" in metadata_content or "processing_time:" in metadata_content
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_answer_formatting_non_verbose_no_citations(self, mock_console, sample_answer_response):
        """Test answer formatting without citations in non-verbose mode."""
        with patch('support_deflect_bot.cli.output.Panel'):
            format_answer(mock_console, sample_answer_response, verbose=False)
            
            # Should not print citations in non-verbose mode
            citations_calls = [call for call in mock_console.print.call_args_list 
                             if "📚 Sources:" in str(call)]
            assert len(citations_calls) == 0
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_answer_formatting_missing_fields(self, mock_console):
        """Test answer formatting with missing optional fields."""
        minimal_response = {
            "answer": "Minimal answer without optional fields"
            # Missing confidence, citations, metadata
        }
        
        with patch('support_deflect_bot.cli.output.Panel') as mock_panel_class:
            format_answer(mock_console, minimal_response, verbose=True)
            
            # Should still work with defaults
            mock_panel_class.assert_called_once()
            call_args = mock_panel_class.call_args
            
            # Should use default confidence of 0.0
            assert "Confidence: 0.000" in call_args[1]["title"]
            
            # Should use red border for zero confidence
            assert call_args[1]["border_style"] == "red"


class TestMetricsTableFormatting(BaseCLITest):
    """Test metrics table formatting functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console for testing."""
        return Mock(spec=Console)
        
    @pytest.fixture  
    def comprehensive_metrics_data(self):
        """Create comprehensive metrics data."""
        return {
            "rag_engine": {
                "queries_processed": 500,
                "successful_answers": 450,
                "average_confidence": 0.82,
                "refusals": 50
            },
            "document_processor": {
                "total_chunks": 2500,
                "connected": True,
                "processing_stats": {
                    "files_processed": 100
                }
            },
            "query_service": {
                "total_queries": 520,
                "success_rate": 0.96
            },
            "embedding_service": {
                "total_embeddings_generated": 3000,
                "cache_hit_rate": 0.75,
                "average_time_per_embedding": 0.0025
            }
        }
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_metrics_table_comprehensive(self, mock_console, comprehensive_metrics_data):
        """Test comprehensive metrics table formatting."""
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_metrics_table(mock_console, comprehensive_metrics_data)
            
            # Verify table was created with correct title
            mock_table_class.assert_called_once_with(title="Performance Metrics")
            
            # Verify columns were added
            assert mock_table.add_column.call_count == 3
            
            # Verify rows were added for each metric
            assert mock_table.add_row.call_count >= 10  # Should have many metric rows
            
            # Check some specific metric values
            row_calls = [call[0] for call in mock_table.add_row.call_args_list]
            
            # Find RAG engine metrics
            rag_rows = [row for row in row_calls if row[0] == "RAG Engine"]
            assert len(rag_rows) >= 1
            
            # Check if queries processed is included
            queries_rows = [row for row in row_calls if "500" in str(row)]
            assert len(queries_rows) >= 1
            
            # Verify table was printed
            mock_console.print.assert_called_once_with(mock_table)
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_metrics_table_partial_data(self, mock_console):
        """Test metrics table with partial data."""
        partial_data = {
            "rag_engine": {
                "queries_processed": 100,
                "successful_answers": 95
                # Missing average_confidence and refusals
            }
            # Missing other components
        }
        
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_metrics_table(mock_console, partial_data)
            
            # Should still create table
            mock_table_class.assert_called_once()
            
            # Should add available metrics with defaults for missing ones
            assert mock_table.add_row.call_count >= 2  # At least queries and answers
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_metrics_table_empty_data(self, mock_console):
        """Test metrics table with empty data."""
        empty_data = {}
        
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_metrics_table(mock_console, empty_data)
            
            # Should still create table structure
            mock_table_class.assert_called_once()
            
            # Should add columns but no data rows
            assert mock_table.add_column.call_count == 3
            assert mock_table.add_row.call_count == 0
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_metrics_table_formatting_calculations(self, mock_console):
        """Test metrics table value formatting and calculations."""
        data_with_calculations = {
            "query_service": {
                "total_queries": 1000,
                "success_rate": 0.95  # Should be formatted as percentage
            },
            "embedding_service": {
                "cache_hit_rate": 0.80,  # Should be formatted as percentage
                "average_time_per_embedding": 0.001234  # Should be formatted with precision
            }
        }
        
        with patch('support_deflect_bot.cli.output.Table') as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table
            
            format_metrics_table(mock_console, data_with_calculations)
            
            # Check formatting of percentage values
            row_calls = [call[0] for call in mock_table.add_row.call_args_list]
            
            # Find percentage formatted values
            percentage_rows = [row for row in row_calls if "%" in str(row)]
            assert len(percentage_rows) >= 2  # success_rate and cache_hit_rate
            
            # Find precision formatted values
            precision_rows = [row for row in row_calls if "0.001234" in str(row)]
            assert len(precision_rows) >= 1


class TestStatusSummaryFormatting(BaseCLITest):
    """Test status summary formatting functionality."""
    
    @pytest.fixture
    def mock_console(self):
        """Create mock console for testing."""
        return Mock(spec=Console)
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_summary_healthy_system(self, mock_console):
        """Test status summary for healthy system."""
        healthy_status = {
            "overall_health": "ok",
            "rag_engine": {"status": "healthy"},
            "document_processor": {"connected": True},
            "provider_validation": {"openai": True, "groq": True}
        }
        
        format_status_summary(mock_console, healthy_status, verbose=False)
        
        # Should print healthy status
        mock_console.print.assert_any_call("✅ System Status: [green]HEALTHY[/green]")
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_summary_database_error(self, mock_console):
        """Test status summary for database error."""
        database_error_status = {
            "overall_health": "database_error",
            "document_processor": {"connected": False}
        }
        
        format_status_summary(mock_console, database_error_status, verbose=False)
        
        # Should print database error status
        mock_console.print.assert_any_call("❌ System Status: [red]DATABASE ERROR[/red]")
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_summary_no_providers(self, mock_console):
        """Test status summary for no providers available."""
        no_providers_status = {
            "overall_health": "no_providers",
            "provider_validation": {"openai": False, "groq": False, "ollama": False}
        }
        
        format_status_summary(mock_console, no_providers_status, verbose=False)
        
        # Should indicate no providers
        calls = [str(call) for call in mock_console.print.call_args_list]
        status_messages = [call for call in calls if "NO PROVIDERS" in call or "no_providers" in call]
        assert len(status_messages) >= 1
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_summary_unknown_status(self, mock_console):
        """Test status summary for unknown health status."""
        unknown_status = {
            "overall_health": "unknown_error",
            "error": "Something unexpected happened"
        }
        
        format_status_summary(mock_console, unknown_status, verbose=False)
        
        # Should handle unknown status gracefully
        assert mock_console.print.called  # Should print something
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_summary_verbose_mode(self, mock_console):
        """Test status summary in verbose mode with additional details."""
        detailed_status = {
            "overall_health": "ok",
            "rag_engine": {
                "status": "healthy",
                "queries_processed": 100,
                "last_query_time": "2024-01-15T10:30:00"
            },
            "document_processor": {
                "connected": True,
                "total_chunks": 1500,
                "last_index_time": "2024-01-15T09:15:00"
            }
        }
        
        format_status_summary(mock_console, detailed_status, verbose=True)
        
        # In verbose mode, should provide more detail
        assert mock_console.print.call_count > 1  # Multiple print statements
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_status_summary_missing_health_field(self, mock_console):
        """Test status summary with missing overall_health field."""
        incomplete_status = {
            "rag_engine": {"status": "healthy"},
            "document_processor": {"connected": True}
            # Missing overall_health field
        }
        
        format_status_summary(mock_console, incomplete_status, verbose=False)
        
        # Should handle missing health field gracefully
        assert mock_console.print.called