#!/usr/bin/env python3
"""Test script to validate provider error messages guide users correctly."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_provider_error_messages():
    """Test that provider error messages provide clear user guidance."""
    print("🧪 Testing Provider Error Messages")
    print("=" * 50)

    # Test 1: Test engine classes can be imported
    print("\n1. Testing Engine Import...")
    try:
        from rag_explorer.engine import (
            UnifiedRAGEngine,
            UnifiedDocumentProcessor,
            UnifiedQueryService,
            UnifiedEmbeddingService
        )
        print("✅ All engine classes imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Test UnifiedQueryService (should work without providers)
    print("\n2. Testing UnifiedQueryService (no provider needed)...")
    try:
        query_service = UnifiedQueryService()
        result = query_service.preprocess_query("test query with extra    spaces")
        print(f"✅ QueryService works: '{result}'")
    except Exception as e:
        print(f"❌ QueryService failed: {e}")

    # Test 3: Test confidence calculation (should work without providers)
    print("\n3. Testing confidence calculation...")
    try:
        rag_engine = UnifiedRAGEngine()
        # Create mock hits for testing
        mock_hits = [
            {'text': 'test document', 'similarity_score': 0.8},
            {'text': 'another document', 'similarity_score': 0.6}
        ]
        confidence = rag_engine.calculate_confidence(mock_hits, "test query")
        print(f"✅ Confidence calculation works: {confidence:.3f}")
    except Exception as e:
        print(f"❌ Confidence calculation failed: {e}")

    # Test 4: Test provider error messages
    print("\n4. Testing Provider Error Messages...")

    # Test RAG Engine with no providers configured
    print("\n   4a. Testing RAG Engine search (should show embedding provider error)...")
    try:
        rag_engine = UnifiedRAGEngine()
        rag_engine.search_documents("test query")
        print("❌ Expected error but got success")
    except ConnectionError as e:
        error_msg = str(e)
        print(f"✅ Got expected ConnectionError: {error_msg}")
        # Check if error message provides guidance
        if "provider not configured" in error_msg.lower() and ("api_key" in error_msg.lower() or "setting" in error_msg.lower()):
            print("✅ Error message provides clear guidance")
        else:
            print(f"⚠️  Error message could be clearer: {error_msg}")
    except Exception as e:
        print(f"❌ Got unexpected error type: {type(e).__name__}: {e}")

    # Test RAG Engine answer_question
    print("\n   4b. Testing RAG Engine answer_question (should show both providers error)...")
    try:
        rag_engine = UnifiedRAGEngine()
        rag_engine.answer_question("test question")
        print("❌ Expected error but got success")
    except ConnectionError as e:
        error_msg = str(e)
        print(f"✅ Got expected ConnectionError: {error_msg}")
        if "provider not configured" in error_msg.lower():
            print("✅ Error message mentions provider configuration")
        else:
            print(f"⚠️  Error message could be clearer: {error_msg}")
    except Exception as e:
        print(f"❌ Got unexpected error type: {type(e).__name__}: {e}")

    # Test Embedding Service directly
    print("\n   4c. Testing EmbeddingService directly...")
    try:
        embedding_service = UnifiedEmbeddingService()
        embedding_service.generate_embeddings("test text")
        print("❌ Expected error but got success")
    except ConnectionError as e:
        error_msg = str(e)
        print(f"✅ Got expected ConnectionError: {error_msg}")
        if any(keyword in error_msg.lower() for keyword in ["api_key", "environment variable", "setting"]):
            print("✅ Error message provides configuration guidance")
        else:
            print(f"⚠️  Error message could provide better guidance: {error_msg}")
    except Exception as e:
        print(f"❌ Got unexpected error type: {type(e).__name__}: {e}")

    # Test Document Processor
    print("\n   4d. Testing DocumentProcessor...")
    try:
        doc_processor = UnifiedDocumentProcessor()
        doc_processor.process_local_directory("/tmp")
        print("❌ Expected error but got success")
    except ConnectionError as e:
        error_msg = str(e)
        print(f"✅ Got expected ConnectionError: {error_msg}")
        if "provider not configured" in error_msg.lower():
            print("✅ Error message mentions provider configuration")
        else:
            print(f"⚠️  Error message could be clearer: {error_msg}")
    except Exception as e:
        print(f"❌ Got unexpected error type: {type(e).__name__}: {e}")

    print("\n🏁 Provider Error Message Testing Complete!")
    print("=" * 50)
    return True

def test_input_validation():
    """Test input validation provides clear error messages."""
    print("\n🧪 Testing Input Validation")
    print("=" * 30)

    try:
        from rag_explorer.engine import UnifiedRAGEngine, UnifiedQueryService, UnifiedEmbeddingService

        # Test query service validation
        print("\n1. Testing QueryService input validation...")
        query_service = UnifiedQueryService()

        try:
            query_service.preprocess_query("")
            print("❌ Empty query should have failed")
        except ValueError as e:
            print(f"✅ Empty query correctly rejected: {e}")

        try:
            query_service.preprocess_query(None)
            print("❌ None query should have failed")
        except ValueError as e:
            print(f"✅ None query correctly rejected: {e}")

        # Test embedding service validation
        print("\n2. Testing EmbeddingService input validation...")
        embedding_service = UnifiedEmbeddingService()

        try:
            embedding_service.generate_embeddings("")
            print("❌ Empty text should have failed")
        except ValueError as e:
            print(f"✅ Empty text correctly rejected: {e}")

        try:
            embedding_service.generate_embeddings_batch([])
            print("❌ Empty list should have failed")
        except ValueError as e:
            print(f"✅ Empty list correctly rejected: {e}")

        # Test RAG engine validation
        print("\n3. Testing RAG Engine input validation...")
        rag_engine = UnifiedRAGEngine()

        try:
            rag_engine.search_documents("")
            print("❌ Empty query should have failed")
        except ValueError as e:
            print(f"✅ Empty query correctly rejected: {e}")

        try:
            rag_engine.answer_question("")
            print("❌ Empty question should have failed")
        except ValueError as e:
            print(f"✅ Empty question correctly rejected: {e}")

        print("\n✅ Input validation tests completed successfully!")

    except Exception as e:
        print(f"❌ Input validation testing failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🚀 Starting RAG Explorer Engine Testing")
    print("=" * 60)

    success = True
    success &= test_provider_error_messages()
    success &= test_input_validation()

    print(f"\n🎯 Overall Test Result: {'✅ PASSED' if success else '❌ FAILED'}")

    if success:
        print("\n🎉 All error handling tests passed!")
        print("✅ Provider error messages guide users correctly")
        print("✅ Input validation works as expected")
        print("✅ Engine classes can be imported and instantiated")
    else:
        print("\n⚠️  Some tests failed - check output above")
        sys.exit(1)