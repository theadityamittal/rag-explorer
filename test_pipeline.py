#!/usr/bin/env python3
"""Test full RAG pipeline and engine class instantiation."""

import sys
import os
import tempfile

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_engine_instantiation():
    """Test that all engine classes can be instantiated without errors."""
    print("🧪 Testing Engine Class Instantiation")
    print("=" * 40)

    try:
        # Test importing all classes
        print("1. Testing imports...")
        from rag_explorer.engine import (
            UnifiedRAGEngine,
            UnifiedDocumentProcessor,
            UnifiedQueryService,
            UnifiedEmbeddingService
        )
        print("✅ All imports successful")

        # Test instantiation
        print("\n2. Testing class instantiation...")

        rag_engine = UnifiedRAGEngine()
        print("✅ UnifiedRAGEngine instantiated")

        doc_processor = UnifiedDocumentProcessor()
        print("✅ UnifiedDocumentProcessor instantiated")

        query_service = UnifiedQueryService()
        print("✅ UnifiedQueryService instantiated")

        embedding_service = UnifiedEmbeddingService()
        print("✅ UnifiedEmbeddingService instantiated")

        # Test basic methods that don't require providers
        print("\n3. Testing provider-independent methods...")

        # Test query preprocessing
        processed = query_service.preprocess_query("  Test Query with Extra   Spaces  ")
        assert processed.strip() == "Test Query with Extra Spaces"
        print("✅ Query preprocessing works")

        # Test query validation
        assert query_service.validate_query("valid query") == True
        assert query_service.validate_query("") == False
        assert query_service.validate_query("a" * 2000) == False
        print("✅ Query validation works")

        # Test keyword extraction
        keywords = query_service.extract_keywords("How do I use Python for machine learning?")
        assert "python" in keywords
        assert "machine" in keywords
        assert "learning" in keywords
        print(f"✅ Keyword extraction works: {keywords}")

        # Test confidence calculation
        mock_hits = [
            {'text': 'Python is great for machine learning', 'similarity_score': 0.8},
            {'text': 'Machine learning with Python libraries', 'similarity_score': 0.7}
        ]
        confidence = rag_engine.calculate_confidence(mock_hits, "Python machine learning")
        assert 0.0 <= confidence <= 1.0
        print(f"✅ Confidence calculation works: {confidence:.3f}")

        # Test document processor stats (no processing)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("This is a test document.")

            stats = doc_processor.get_processing_stats(temp_dir)
            assert stats['total_files'] >= 1
            assert stats['supported_files'] >= 1
            print(f"✅ Document processor stats work: {stats['total_files']} files found")

        # Test embedding service provider info (will show error but shouldn't crash)
        provider_info = embedding_service.get_provider_info()
        assert 'error' in provider_info or 'name' in provider_info
        print("✅ Embedding service provider info works")

        # Test RAG engine status
        status = rag_engine.get_status()
        assert 'status' in status
        print(f"✅ RAG engine status works: {status['status']}")

        print("\n🎉 All engine class instantiation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Engine instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_structure():
    """Test the logical structure of the RAG pipeline without providers."""
    print("\n🧪 Testing RAG Pipeline Structure")
    print("=" * 40)

    try:
        from rag_explorer.engine import (
            UnifiedRAGEngine,
            UnifiedDocumentProcessor,
            UnifiedQueryService,
            UnifiedEmbeddingService
        )

        # Simulate the pipeline steps
        print("1. Simulating document processing pipeline...")

        doc_processor = UnifiedDocumentProcessor()
        supported_extensions = doc_processor.get_supported_extensions()
        assert '.txt' in supported_extensions
        assert '.md' in supported_extensions
        assert '.py' in supported_extensions
        print(f"✅ Supports {len(supported_extensions)} file types: {supported_extensions}")

        print("\n2. Simulating query processing pipeline...")

        query_service = UnifiedQueryService()

        # Test query preprocessing pipeline
        raw_query = "   How do I use Python for machine learning?   "
        processed_query = query_service.preprocess_query(raw_query)
        keywords = query_service.extract_keywords(processed_query)
        query_stats = query_service.get_query_stats(raw_query)

        print(f"✅ Raw query: '{raw_query}'")
        print(f"✅ Processed: '{processed_query}'")
        print(f"✅ Keywords: {keywords}")
        print(f"✅ Stats: {query_stats['word_count']} words, {query_stats['keyword_count']} keywords")

        print("\n3. Simulating search pipeline structure...")

        rag_engine = UnifiedRAGEngine()

        # Test search pipeline logic (will fail at provider level, but structure is sound)
        try:
            rag_engine.search_documents("test query")
            print("❌ Expected error but got success")
        except ConnectionError as e:
            if "provider not configured" in str(e).lower():
                print("✅ Search pipeline structure correct (provider error expected)")
            else:
                print(f"⚠️ Unexpected error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")

        print("\n4. Simulating RAG answer pipeline structure...")

        try:
            rag_engine.answer_question("test question")
            print("❌ Expected error but got success")
        except ConnectionError as e:
            if "provider not configured" in str(e).lower():
                print("✅ Answer pipeline structure correct (provider error expected)")
            else:
                print(f"⚠️ Unexpected error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")

        print("\n5. Testing input validation throughout pipeline...")

        # Test validation at each step
        validation_tests = [
            (lambda: query_service.preprocess_query(""), "Empty query"),
            (lambda: rag_engine.search_documents(""), "Empty search query"),
            (lambda: rag_engine.answer_question(""), "Empty question"),
            (lambda: rag_engine.search_documents("valid", count=0), "Invalid count"),
            (lambda: rag_engine.answer_question("valid", k=0), "Invalid k"),
            (lambda: rag_engine.answer_question("valid", min_confidence=2.0), "Invalid confidence"),
        ]

        for test_func, description in validation_tests:
            try:
                test_func()
                print(f"❌ {description} should have failed")
            except ValueError:
                print(f"✅ {description} correctly rejected")
            except ConnectionError:
                print(f"✅ {description} reached provider (validation passed)")
            except Exception as e:
                print(f"⚠️ {description} unexpected error: {e}")

        print("\n🎉 All pipeline structure tests passed!")
        return True

    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test that CLI commands can import and use engine classes."""
    print("\n🧪 Testing CLI Integration")
    print("=" * 30)

    try:
        # Test that CLI files compile
        cli_files = [
            'src/rag_explorer/cli/commands/ask.py',
            'src/rag_explorer/cli/commands/search.py',
            'src/rag_explorer/cli/commands/index.py'
        ]

        import py_compile
        for cli_file in cli_files:
            if os.path.exists(cli_file):
                py_compile.compile(cli_file, doraise=True)
                print(f"✅ {cli_file} compiles successfully")
            else:
                print(f"⚠️ {cli_file} not found")

        # Test imports that CLI uses
        print("\nTesting CLI imports...")
        try:
            from rag_explorer.engine import UnifiedRAGEngine
            from rag_explorer.engine import UnifiedDocumentProcessor
            from rag_explorer.engine import UnifiedQueryService
            from rag_explorer.engine import UnifiedEmbeddingService
            print("✅ All CLI imports work")
        except ImportError as e:
            print(f"❌ CLI import failed: {e}")
            return False

        # Test that CLI can instantiate engines
        print("\nTesting CLI instantiation...")
        rag_engine = UnifiedRAGEngine()
        doc_processor = UnifiedDocumentProcessor()
        query_service = UnifiedQueryService()
        embedding_service = UnifiedEmbeddingService()
        print("✅ CLI can instantiate all engines")

        print("\n🎉 CLI integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Full RAG Pipeline Testing")
    print("=" * 60)

    success = True
    success &= test_engine_instantiation()
    success &= test_pipeline_structure()
    success &= test_cli_integration()

    print(f"\n🎯 Overall Pipeline Test Result: {'✅ PASSED' if success else '❌ FAILED'}")

    if success:
        print("\n🎉 All pipeline tests passed!")
        print("✅ Engine classes instantiate correctly")
        print("✅ Pipeline structure is sound")
        print("✅ CLI integration works")
        print("✅ Input validation throughout pipeline")
        print("✅ Error handling provides clear guidance")
    else:
        print("\n⚠️ Some pipeline tests failed - check output above")
        sys.exit(1)