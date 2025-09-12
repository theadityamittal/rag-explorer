"""
Unit tests for CLI interactive ask session functionality.

Tests the UnifiedAskSession class, session management, question processing,
statistics tracking, and user interaction patterns.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from rich.console import Console
from rich.prompt import Prompt
from tests.base import BaseCLITest

from support_deflect_bot.cli.ask_session import UnifiedAskSession


class TestUnifiedAskSessionInitialization(BaseCLITest):
    """Test ask session initialization and configuration."""
    
    @pytest.fixture
    def mock_rag_engine(self):
        """Create mock RAG engine."""
        mock = Mock()
        mock.answer_question.return_value = {
            "answer": "Test answer",
            "confidence": 0.85,
            "citations": ["test.md"],
            "refusal": False
        }
        return mock
        
    @pytest.fixture
    def mock_query_service(self):
        """Create mock query service."""
        return Mock()
        
    @pytest.fixture
    def mock_console(self):
        """Create mock console."""
        return Mock(spec=Console)
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_session_initialization_defaults(self, mock_rag_engine, mock_query_service, mock_console):
        """Test ask session initialization with default parameters."""
        session = UnifiedAskSession(
            rag_engine=mock_rag_engine,
            query_service=mock_query_service,
            console=mock_console
        )
        
        assert session.rag_engine is mock_rag_engine
        assert session.query_service is mock_query_service
        assert session.console is mock_console
        assert session.domain_filter is None
        assert session.confidence_override is None
        assert session.max_chunks_override is None
        assert session.verbose is False
        assert session.quiet is False
        
        # Check session stats initialization
        assert "questions_asked" in session.session_stats
        assert "successful_answers" in session.session_stats
        assert "refusals" in session.session_stats
        assert "total_time" in session.session_stats
        assert "start_time" in session.session_stats
        assert session.session_stats["questions_asked"] == 0
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_ask_session_initialization_with_options(self, mock_rag_engine, mock_query_service, mock_console):
        """Test ask session initialization with custom options."""
        session = UnifiedAskSession(
            rag_engine=mock_rag_engine,
            query_service=mock_query_service,
            console=mock_console,
            domain_filter=["example.com", "test.org"],
            confidence_override=0.7,
            max_chunks_override=10,
            verbose=True,
            quiet=False
        )
        
        assert session.domain_filter == ["example.com", "test.org"]
        assert session.confidence_override == 0.7
        assert session.max_chunks_override == 10
        assert session.verbose is True
        assert session.quiet is False


class TestAskSessionStartupAndShutdown(BaseCLITest):
    """Test ask session startup and shutdown behavior."""
    
    @pytest.fixture
    def session_with_mocks(self):
        """Create session with all mocks."""
        mock_rag = Mock()
        mock_query = Mock()
        mock_console = Mock(spec=Console)
        
        session = UnifiedAskSession(
            rag_engine=mock_rag,
            query_service=mock_query,
            console=mock_console
        )
        return session, mock_rag, mock_query, mock_console
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_startup_messages_normal_mode(self, session_with_mocks):
        """Test session startup messages in normal mode."""
        session, mock_rag, mock_query, mock_console = session_with_mocks
        
        with patch.object(Prompt, 'ask', side_effect=EOFError):  # Exit immediately
            session.start()
            
        # Check startup messages were printed
        mock_console.print.assert_any_call(
            "🤖 Support Deflect Bot - Interactive Q&A Session", 
            style="cyan bold"
        )
        mock_console.print.assert_any_call(
            "❓ Ask questions about the indexed documentation", 
            style="blue"
        )
        mock_console.print.assert_any_call(
            "💡 Type 'end', 'exit', or 'quit' to finish\n", 
            style="dim"
        )
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_startup_messages_quiet_mode(self, session_with_mocks):
        """Test session startup messages in quiet mode."""
        session, mock_rag, mock_query, mock_console = session_with_mocks
        session.quiet = True
        
        with patch.object(Prompt, 'ask', side_effect=EOFError):  # Exit immediately
            session.start()
            
        # In quiet mode, no startup messages should be printed
        mock_console.print.assert_not_called()
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_startup_with_filters_displayed(self, session_with_mocks):
        """Test session displays active filters during startup."""
        session, mock_rag, mock_query, mock_console = session_with_mocks
        session.domain_filter = ["example.com"]
        session.confidence_override = 0.8
        session.max_chunks_override = 5
        
        with patch.object(Prompt, 'ask', side_effect=EOFError):  # Exit immediately
            session.start()
            
        # Check filter display messages
        mock_console.print.assert_any_call(
            "🌐 Domain filter: example.com", 
            style="yellow"
        )
        mock_console.print.assert_any_call(
            "🎯 Confidence threshold: 0.800", 
            style="yellow"
        )
        mock_console.print.assert_any_call(
            "📊 Max chunks: 5", 
            style="yellow"
        )
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_summary_display(self, session_with_mocks):
        """Test session summary is displayed on exit."""
        session, mock_rag, mock_query, mock_console = session_with_mocks
        
        # Simulate some session activity
        session.session_stats["questions_asked"] = 5
        session.session_stats["successful_answers"] = 4
        session.session_stats["refusals"] = 1
        session.session_stats["total_time"] = 25.5
        
        with patch.object(session, '_show_session_summary') as mock_summary:
            with patch.object(Prompt, 'ask', side_effect=EOFError):  # Exit immediately
                session.start()
                
            mock_summary.assert_called_once()


class TestQuestionProcessing(BaseCLITest):
    """Test question processing and RAG integration."""
    
    @pytest.fixture
    def session_for_processing(self):
        """Create session configured for processing tests."""
        mock_rag = Mock()
        mock_query = Mock()
        mock_console = Mock(spec=Console)
        
        session = UnifiedAskSession(
            rag_engine=mock_rag,
            query_service=mock_query,
            console=mock_console,
            verbose=True
        )
        return session, mock_rag, mock_query, mock_console
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_successful_question_processing(self, session_for_processing):
        """Test successful question processing through RAG pipeline."""
        session, mock_rag, mock_query, mock_console = session_for_processing
        
        # Setup successful RAG response
        mock_rag.answer_question.return_value = {
            "answer": "The installation process requires running pip install -e .",
            "confidence": 0.9,
            "citations": ["installation.md"],
            "refusal": False
        }
        
        with patch('support_deflect_bot.cli.ask_session.format_answer') as mock_format:
            session._process_question("How do I install this?")
            
        # Verify RAG engine was called correctly
        mock_rag.answer_question.assert_called_once_with(
            question="How do I install this?",
            k=None,  # max_chunks_override not set
            domains=None,  # domain_filter not set
            min_confidence=None  # confidence_override not set
        )
        
        # Verify answer was formatted
        mock_format.assert_called_once()
        
        # Verify stats were updated
        assert session.session_stats["questions_asked"] == 1
        assert session.session_stats["successful_answers"] == 1
        assert session.session_stats["refusals"] == 0
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_question_processing_with_filters(self, session_for_processing):
        """Test question processing with domain filters and overrides."""
        session, mock_rag, mock_query, mock_console = session_for_processing
        
        # Set filters and overrides
        session.domain_filter = ["docs.example.com"]
        session.confidence_override = 0.8
        session.max_chunks_override = 15
        
        mock_rag.answer_question.return_value = {
            "answer": "Test answer with filters",
            "confidence": 0.85,
            "citations": ["filtered.md"],
            "refusal": False
        }
        
        with patch('support_deflect_bot.cli.ask_session.format_answer'):
            session._process_question("Test question with filters")
            
        # Verify RAG engine received the filters
        mock_rag.answer_question.assert_called_once_with(
            question="Test question with filters",
            k=15,  # max_chunks_override
            domains=["docs.example.com"],  # domain_filter
            min_confidence=0.8  # confidence_override
        )
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_question_processing_refusal_handling(self, session_for_processing):
        """Test handling of RAG refusals (low confidence)."""
        session, mock_rag, mock_query, mock_console = session_for_processing
        
        # Setup RAG refusal response
        mock_rag.answer_question.return_value = {
            "answer": "I don't have enough reliable information to answer this question.",
            "confidence": 0.3,
            "citations": [],
            "refusal": True
        }
        
        with patch('support_deflect_bot.cli.ask_session.format_answer') as mock_format:
            session._process_question("Obscure question")
            
        # Verify answer was still formatted (refusals are formatted too)
        mock_format.assert_called_once()
        
        # Verify refusal was tracked in stats
        assert session.session_stats["questions_asked"] == 1
        assert session.session_stats["successful_answers"] == 0
        assert session.session_stats["refusals"] == 1
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_question_processing_error_handling(self, session_for_processing):
        """Test question processing error handling."""
        session, mock_rag, mock_query, mock_console = session_for_processing
        
        # Setup RAG engine to throw exception
        mock_rag.answer_question.side_effect = Exception("RAG processing failed")
        
        session._process_question("Error-causing question")
        
        # Verify error message was displayed
        mock_console.print.assert_any_call(
            "❌ Error processing question: RAG processing failed", 
            style="red"
        )
        
        # Stats should still be updated for attempted question
        assert session.session_stats["questions_asked"] == 1
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_processing_time_tracking(self, session_for_processing):
        """Test that processing time is tracked accurately."""
        session, mock_rag, mock_query, mock_console = session_for_processing
        
        # Mock time.time to control timing
        with patch('support_deflect_bot.cli.ask_session.time.time') as mock_time:
            mock_time.side_effect = [100.0, 102.5]  # 2.5 second processing time
            
            mock_rag.answer_question.return_value = {
                "answer": "Test answer",
                "confidence": 0.8,
                "citations": [],
                "refusal": False
            }
            
            with patch('support_deflect_bot.cli.ask_session.format_answer'):
                session._process_question("Timed question")
                
        # Verify processing time was added to total
        assert session.session_stats["total_time"] == 2.5
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_verbose_mode_processing_messages(self, session_for_processing):
        """Test verbose mode shows processing messages."""
        session, mock_rag, mock_query, mock_console = session_for_processing
        session.verbose = True
        session.quiet = False
        
        mock_rag.answer_question.return_value = {
            "answer": "Verbose test answer",
            "confidence": 0.8,
            "citations": [],
            "refusal": False
        }
        
        with patch('support_deflect_bot.cli.ask_session.format_answer'):
            session._process_question("Verbose test question")
            
        # Verify processing message was displayed
        mock_console.print.assert_any_call(
            "\n⚙️ Processing: Verbose test question", 
            style="dim"
        )


class TestSessionLoop(BaseCLITest):
    """Test the main session loop and user interaction."""
    
    @pytest.fixture
    def loop_session(self):
        """Create session for loop testing."""
        mock_rag = Mock()
        mock_query = Mock()
        mock_console = Mock(spec=Console)
        
        mock_rag.answer_question.return_value = {
            "answer": "Loop test answer",
            "confidence": 0.8,
            "citations": [],
            "refusal": False
        }
        
        session = UnifiedAskSession(
            rag_engine=mock_rag,
            query_service=mock_query,
            console=mock_console,
            quiet=True  # Suppress startup messages
        )
        return session, mock_rag, mock_query, mock_console
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_loop_with_questions(self, loop_session):
        """Test session loop processes multiple questions."""
        session, mock_rag, mock_query, mock_console = loop_session
        
        # Simulate user input: ask two questions then quit
        questions = ["First question", "Second question", "quit"]
        
        with patch.object(Prompt, 'ask', side_effect=questions):
            with patch('support_deflect_bot.cli.ask_session.format_answer'):
                session.start()
                
        # Verify both questions were processed
        assert mock_rag.answer_question.call_count == 2
        assert session.session_stats["questions_asked"] == 2
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_loop_exit_commands(self, loop_session):
        """Test session loop exits on exit commands."""
        session, mock_rag, mock_query, mock_console = loop_session
        
        exit_commands = ["end", "exit", "quit", ""]
        
        for exit_cmd in exit_commands:
            # Reset session stats
            session.session_stats["questions_asked"] = 0
            
            with patch.object(Prompt, 'ask', return_value=exit_cmd):
                with patch('support_deflect_bot.cli.ask_session.format_answer'):
                    session.start()
                    
            # Verify no questions were processed for exit commands
            assert session.session_stats["questions_asked"] == 0
            
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_loop_keyboard_interrupt(self, loop_session):
        """Test session loop handles KeyboardInterrupt gracefully."""
        session, mock_rag, mock_query, mock_console = loop_session
        
        with patch.object(Prompt, 'ask', side_effect=KeyboardInterrupt):
            with patch('support_deflect_bot.cli.ask_session.format_answer'):
                session.start()  # Should not raise exception
                
        # Session should exit gracefully
        assert session.session_stats["questions_asked"] == 0
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_loop_eof_error(self, loop_session):
        """Test session loop handles EOFError gracefully."""
        session, mock_rag, mock_query, mock_console = loop_session
        
        with patch.object(Prompt, 'ask', side_effect=EOFError):
            with patch('support_deflect_bot.cli.ask_session.format_answer'):
                session.start()  # Should not raise exception
                
        # Session should exit gracefully
        assert session.session_stats["questions_asked"] == 0


class TestSessionStatistics(BaseCLITest):
    """Test session statistics tracking and reporting."""
    
    @pytest.fixture
    def stats_session(self):
        """Create session for statistics testing."""
        mock_rag = Mock()
        mock_query = Mock()
        mock_console = Mock(spec=Console)
        
        session = UnifiedAskSession(
            rag_engine=mock_rag,
            query_service=mock_query,
            console=mock_console,
            quiet=True
        )
        return session, mock_rag, mock_query, mock_console
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_statistics_tracking(self, stats_session):
        """Test comprehensive session statistics tracking."""
        session, mock_rag, mock_query, mock_console = stats_session
        
        # Simulate mixed success/refusal responses
        responses = [
            {"answer": "Success 1", "confidence": 0.9, "citations": [], "refusal": False},
            {"answer": "Low confidence", "confidence": 0.2, "citations": [], "refusal": True},
            {"answer": "Success 2", "confidence": 0.8, "citations": [], "refusal": False}
        ]
        
        mock_rag.answer_question.side_effect = responses
        
        with patch('support_deflect_bot.cli.ask_session.format_answer'):
            session._process_question("Question 1")
            session._process_question("Question 2")  
            session._process_question("Question 3")
            
        # Verify statistics
        assert session.session_stats["questions_asked"] == 3
        assert session.session_stats["successful_answers"] == 2
        assert session.session_stats["refusals"] == 1
        assert session.session_stats["total_time"] > 0
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_summary_display_content(self, stats_session):
        """Test session summary displays correct information."""
        session, mock_rag, mock_query, mock_console = stats_session
        
        # Setup session stats
        session.session_stats.update({
            "questions_asked": 10,
            "successful_answers": 8,
            "refusals": 2,
            "total_time": 45.7,
            "start_time": time.time() - 60  # 60 seconds ago
        })
        
        session._show_session_summary()
        
        # Verify summary information was displayed
        mock_console.print.assert_any_call(
            "\n📊 Session Summary", 
            style="cyan bold"
        )
        
        # Check for key statistics (exact format may vary)
        calls = [call[0][0] for call in mock_console.print.call_args_list]
        summary_text = " ".join(calls)
        
        assert "10" in summary_text  # questions asked
        assert "8" in summary_text   # successful answers  
        assert "2" in summary_text   # refusals
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_summary_quiet_mode(self, stats_session):
        """Test session summary in quiet mode."""
        session, mock_rag, mock_query, mock_console = stats_session
        session.quiet = True
        
        session.session_stats.update({
            "questions_asked": 5,
            "successful_answers": 4,
            "refusals": 1,
            "total_time": 20.0
        })
        
        session._show_session_summary()
        
        # In quiet mode, should show minimal summary or none
        # Verify fewer print calls than verbose mode
        assert mock_console.print.call_count <= 1
        
    @pytest.mark.unit
    @pytest.mark.cli
    def test_session_timing_accuracy(self, stats_session):
        """Test session timing calculations are accurate."""
        session, mock_rag, mock_query, mock_console = stats_session
        
        # Mock time progression
        with patch('support_deflect_bot.cli.ask_session.time.time') as mock_time:
            # Start time: 100, processing takes 3.0 seconds each
            mock_time.side_effect = [100.0, 103.0, 103.0, 106.0, 106.0, 109.0]
            
            mock_rag.answer_question.return_value = {
                "answer": "Timed answer",
                "confidence": 0.8,
                "citations": [],
                "refusal": False
            }
            
            with patch('support_deflect_bot.cli.ask_session.format_answer'):
                session._process_question("Q1")
                session._process_question("Q2")
                session._process_question("Q3")
                
        # Total time should be 9.0 seconds (3 questions × 3 seconds each)
        assert session.session_stats["total_time"] == 9.0
        assert session.session_stats["questions_asked"] == 3