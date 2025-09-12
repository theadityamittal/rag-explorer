"""Interactive ask session using unified engine services."""

import time
from typing import List, Optional, Dict
from rich.console import Console
from rich.prompt import Prompt

from ..engine import UnifiedRAGEngine, UnifiedQueryService
from .output import format_answer


class UnifiedAskSession:
    """Interactive Q&A session using unified engine services."""
    
    def __init__(
        self,
        rag_engine: UnifiedRAGEngine,
        query_service: UnifiedQueryService,
        console: Console,
        domain_filter: Optional[List[str]] = None,
        confidence_override: Optional[float] = None,
        max_chunks_override: Optional[int] = None,
        verbose: bool = False,
        quiet: bool = False
    ):
        """Initialize the ask session."""
        self.rag_engine = rag_engine
        self.query_service = query_service
        self.console = console
        self.domain_filter = domain_filter
        self.confidence_override = confidence_override
        self.max_chunks_override = max_chunks_override
        self.verbose = verbose
        self.quiet = quiet
        
        # Session statistics
        self.session_stats = {
            "questions_asked": 0,
            "successful_answers": 0,
            "refusals": 0,
            "total_time": 0.0,
            "start_time": time.time()
        }
    
    def start(self):
        """Start the interactive ask session."""
        if not self.quiet:
            self.console.print("🤖 Support Deflect Bot - Interactive Q&A Session", style="cyan bold")
            self.console.print("❓ Ask questions about the indexed documentation", style="blue")
            self.console.print("💡 Type 'end', 'exit', or 'quit' to finish\n", style="dim")
            
            if self.domain_filter:
                self.console.print(f"🌐 Domain filter: {', '.join(self.domain_filter)}", style="yellow")
            if self.confidence_override:
                self.console.print(f"🎯 Confidence threshold: {self.confidence_override:.3f}", style="yellow")
            if self.max_chunks_override:
                self.console.print(f"📊 Max chunks: {self.max_chunks_override}", style="yellow")
            
            if self.domain_filter or self.confidence_override or self.max_chunks_override:
                self.console.print()
        
        try:
            while True:
                # Get user input
                try:
                    question = Prompt.ask("[bold cyan]Question[/bold cyan]", console=self.console)
                except (EOFError, KeyboardInterrupt):
                    break
                
                # Check for exit commands
                if question.lower().strip() in ["end", "exit", "quit", ""]:
                    break
                
                # Process the question
                self._process_question(question)
        
        finally:
            self._show_session_summary()
    
    def _process_question(self, question: str):
        """Process a single question through the RAG pipeline."""
        start_time = time.time()
        self.session_stats["questions_asked"] += 1
        
        try:
            if not self.quiet:
                self.console.print(f"\n⚙️ Processing: {question}", style="dim")
            
            # Use RAG engine to answer the question
            response = self.rag_engine.answer_question(
                question=question,
                k=self.max_chunks_override,
                domains=self.domain_filter,
                min_confidence=self.confidence_override
            )
            
            processing_time = time.time() - start_time
            self.session_stats["total_time"] += processing_time
            
            # Check if it's a refusal
            answer = response.get("answer", "")
            if "don't have enough information" in answer.lower():
                self.session_stats["refusals"] += 1
            else:
                self.session_stats["successful_answers"] += 1
            
            # Format and display the answer
            self.console.print()
            format_answer(self.console, response, verbose=self.verbose)
            
            # Show processing time if verbose
            if self.verbose:
                confidence = response.get("confidence", 0.0)
                chunks_found = response.get("metadata", {}).get("chunks_found", 0)
                self.console.print(
                    f"\n⏱️ Processing time: {processing_time:.3f}s | "
                    f"Chunks: {chunks_found} | "
                    f"Confidence: {confidence:.3f}",
                    style="dim"
                )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.session_stats["total_time"] += processing_time
            
            self.console.print(f"\n❌ Error processing question: {e}", style="red")
            if self.verbose:
                import traceback
                self.console.print(traceback.format_exc(), style="dim red")
    
    def _show_session_summary(self):
        """Show session statistics."""
        if self.quiet:
            return
        
        total_session_time = time.time() - self.session_stats["start_time"]
        
        self.console.print("\n" + "="*50, style="dim")
        self.console.print("📋 Session Summary", style="cyan bold")
        
        # Basic stats
        questions = self.session_stats["questions_asked"]
        successful = self.session_stats["successful_answers"]
        refusals = self.session_stats["refusals"]
        
        if questions > 0:
            success_rate = (successful / questions) * 100
            self.console.print(f"Questions asked: {questions}", style="blue")
            self.console.print(f"Successful answers: {successful}", style="green")
            self.console.print(f"Refusals: {refusals}", style="yellow")
            self.console.print(f"Success rate: {success_rate:.1f}%", style="cyan")
            
            # Timing stats
            avg_processing_time = self.session_stats["total_time"] / questions
            self.console.print(f"Average processing time: {avg_processing_time:.3f}s", style="blue")
            self.console.print(f"Total session time: {total_session_time:.1f}s", style="dim")
        else:
            self.console.print("No questions were asked.", style="yellow")
        
        # Get final metrics from engines
        if self.verbose:
            self.console.print("\n🔧 Engine Metrics:", style="cyan")
            
            # RAG engine metrics
            rag_metrics = self.rag_engine.get_metrics()
            self.console.print(f"Total RAG queries: {rag_metrics.get('queries_processed', 0)}", style="dim")
            self.console.print(f"Overall confidence: {rag_metrics.get('average_confidence', 0):.3f}", style="dim")
            
            # Query service metrics
            query_metrics = self.query_service.get_query_analytics()
            self.console.print(f"Query service calls: {query_metrics.get('total_queries', 0)}", style="dim")
            self.console.print(f"Query success rate: {query_metrics.get('success_rate', 0)*100:.1f}%", style="dim")
        
        self.console.print("\n👋 Thanks for using Support Deflect Bot!", style="green")


class BatchAskProcessor:
    """Process multiple questions in batch mode."""
    
    def __init__(
        self,
        rag_engine: UnifiedRAGEngine,
        console: Console,
        domain_filter: Optional[List[str]] = None,
        confidence_override: Optional[float] = None,
        verbose: bool = False
    ):
        """Initialize batch processor."""
        self.rag_engine = rag_engine
        self.console = console
        self.domain_filter = domain_filter
        self.confidence_override = confidence_override
        self.verbose = verbose
    
    def process_questions(self, questions: List[str]) -> List[Dict]:
        """Process a list of questions and return results."""
        results = []
        
        if not questions:
            return results
        
        self.console.print(f"📊 Processing {len(questions)} questions in batch...", style="cyan")
        
        for i, question in enumerate(questions, 1):
            if self.verbose:
                self.console.print(f"[{i}/{len(questions)}] {question[:50]}...", style="dim")
            
            try:
                start_time = time.time()
                
                response = self.rag_engine.answer_question(
                    question=question,
                    domains=self.domain_filter,
                    min_confidence=self.confidence_override
                )
                
                processing_time = time.time() - start_time
                
                result = {
                    "question": question,
                    "response": response,
                    "processing_time": processing_time,
                    "success": True
                }
                
                results.append(result)
                
            except Exception as e:
                result = {
                    "question": question,
                    "error": str(e),
                    "processing_time": 0.0,
                    "success": False
                }
                results.append(result)
                
                if self.verbose:
                    self.console.print(f"❌ Error: {e}", style="red")
        
        return results
    
    def format_batch_results(self, results: List[Dict], output_format: str = "table"):
        """Format and display batch results."""
        if not results:
            self.console.print("No results to display.", style="yellow")
            return
        
        if output_format == "json":
            import json
            output_data = []
            for result in results:
                if result["success"]:
                    output_data.append({
                        "question": result["question"],
                        "answer": result["response"]["answer"],
                        "confidence": result["response"]["confidence"],
                        "processing_time": result["processing_time"]
                    })
                else:
                    output_data.append({
                        "question": result["question"],
                        "error": result["error"],
                        "processing_time": result["processing_time"]
                    })
            
            self.console.print(json.dumps(output_data, indent=2))
        
        else:
            # Table format
            from rich.table import Table
            
            table = Table(title="Batch Q&A Results")
            table.add_column("Question", style="cyan", width=40)
            table.add_column("Answer", style="white", width=50)
            table.add_column("Confidence", style="green", width=10)
            table.add_column("Time", style="blue", width=8)
            
            for result in results:
                question = result["question"][:37] + "..." if len(result["question"]) > 40 else result["question"]
                
                if result["success"]:
                    answer = result["response"]["answer"]
                    answer = answer[:47] + "..." if len(answer) > 50 else answer
                    confidence = f"{result['response']['confidence']:.3f}"
                else:
                    answer = f"❌ {result['error']}"[:50]
                    confidence = "N/A"
                
                time_str = f"{result['processing_time']:.3f}s"
                
                table.add_row(question, answer, confidence, time_str)
            
            self.console.print(table)
            
            # Summary
            successful = sum(1 for r in results if r["success"])
            total_time = sum(r["processing_time"] for r in results)
            avg_time = total_time / len(results) if results else 0
            
            self.console.print(f"\n📋 Batch Summary: {successful}/{len(results)} successful, {avg_time:.3f}s avg time", style="cyan")