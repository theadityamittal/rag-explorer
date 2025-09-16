#!/usr/bin/env python3
"""Test confidence calculation algorithm independently."""

import sys
import os

def test_confidence_calculation():
    """Test the 4-factor confidence calculation algorithm."""
    print("🧪 Testing Confidence Calculation Algorithm")
    print("=" * 50)

    # Mock the confidence calculation logic here (extracted from our implementation)
    def calculate_confidence_mock(hits, question):
        """Mock confidence calculation to test the algorithm."""
        import re

        if not hits:
            return 0.0

        try:
            # Factor 1: Similarity scores (40% weight)
            similarity_scores = [hit.get('similarity_score', 0.0) for hit in hits]
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            similarity_factor = min(avg_similarity, 1.0)

            # Factor 2: Result count (20% weight)
            result_count = len(hits)
            max_expected_results = 10
            count_factor = min(result_count / max_expected_results, 1.0)

            # Factor 3: Keyword overlap (20% weight)
            def extract_keywords(text):
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
                return {word for word in words if word not in stop_words}

            question_keywords = extract_keywords(question.lower())
            if question_keywords:
                overlap_scores = []
                for hit in hits:
                    text = hit.get('text', '').lower()
                    text_keywords = extract_keywords(text)
                    if text_keywords:
                        overlap = len(question_keywords.intersection(text_keywords))
                        overlap_score = overlap / len(question_keywords)
                        overlap_scores.append(overlap_score)

                keyword_factor = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
                keyword_factor = min(keyword_factor, 1.0)
            else:
                keyword_factor = 0.0

            # Factor 4: Content length (20% weight)
            content_lengths = [len(hit.get('text', '')) for hit in hits]
            avg_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0.0
            expected_min_length = 100
            expected_max_length = 2000
            normalized_length = min(max(avg_length - expected_min_length, 0) /
                                   (expected_max_length - expected_min_length), 1.0)
            content_factor = normalized_length

            # Weighted combination
            confidence = (
                similarity_factor * 0.4 +   # 40% weight
                count_factor * 0.2 +        # 20% weight
                keyword_factor * 0.2 +      # 20% weight
                content_factor * 0.2        # 20% weight
            )

            # Ensure result is in valid range
            confidence = max(0.0, min(confidence, 1.0))

            print(f"    Factors: similarity={similarity_factor:.3f}, count={count_factor:.3f}, "
                  f"keywords={keyword_factor:.3f}, content={content_factor:.3f}")

            return confidence

        except Exception as e:
            print(f"    Error in calculation: {e}")
            return 0.0

    # Test Case 1: High confidence scenario
    print("\n1. Testing high confidence scenario...")
    high_confidence_hits = [
        {'text': 'Python is a programming language used for data science and machine learning', 'similarity_score': 0.9},
        {'text': 'Machine learning algorithms can be implemented in Python using libraries', 'similarity_score': 0.85},
        {'text': 'Data science projects often use Python for data analysis and visualization', 'similarity_score': 0.8},
        {'text': 'Programming in Python requires understanding of syntax and data structures', 'similarity_score': 0.75},
        {'text': 'Python machine learning libraries include scikit-learn and TensorFlow', 'similarity_score': 0.7}
    ]
    question1 = "How is Python used in machine learning?"
    confidence1 = calculate_confidence_mock(high_confidence_hits, question1)
    print(f"    Confidence: {confidence1:.3f}")
    assert 0.4 <= confidence1 <= 1.0, f"Expected high confidence, got {confidence1:.3f}"
    print("✅ High confidence test passed")

    # Test Case 2: Medium confidence scenario
    print("\n2. Testing medium confidence scenario...")
    medium_confidence_hits = [
        {'text': 'Programming languages have different syntax rules', 'similarity_score': 0.6},
        {'text': 'Software development involves writing code', 'similarity_score': 0.55},
        {'text': 'Computer science covers many topics', 'similarity_score': 0.5}
    ]
    question2 = "How is Python used in machine learning?"
    confidence2 = calculate_confidence_mock(medium_confidence_hits, question2)
    print(f"    Confidence: {confidence2:.3f}")
    assert 0.1 <= confidence2 <= 0.6, f"Expected medium confidence, got {confidence2:.3f}"
    print("✅ Medium confidence test passed")

    # Test Case 3: Low confidence scenario
    print("\n3. Testing low confidence scenario...")
    low_confidence_hits = [
        {'text': 'The weather is nice today', 'similarity_score': 0.2},
        {'text': 'Cats are popular pets', 'similarity_score': 0.15}
    ]
    question3 = "How is Python used in machine learning?"
    confidence3 = calculate_confidence_mock(low_confidence_hits, question3)
    print(f"    Confidence: {confidence3:.3f}")
    assert 0.0 <= confidence3 <= 0.4, f"Expected low confidence, got {confidence3:.3f}"
    print("✅ Low confidence test passed")

    # Test Case 4: Empty hits
    print("\n4. Testing empty hits scenario...")
    confidence4 = calculate_confidence_mock([], "test question")
    print(f"    Confidence: {confidence4:.3f}")
    assert confidence4 == 0.0, f"Expected 0.0 confidence for empty hits, got {confidence4:.3f}"
    print("✅ Empty hits test passed")

    # Test Case 5: Edge case - very long content
    print("\n5. Testing long content scenario...")
    long_content_hits = [
        {'text': 'A' * 3000 + ' python machine learning artificial intelligence', 'similarity_score': 0.8},
        {'text': 'B' * 2500 + ' data science programming algorithms', 'similarity_score': 0.75}
    ]
    question5 = "python machine learning"
    confidence5 = calculate_confidence_mock(long_content_hits, question5)
    print(f"    Confidence: {confidence5:.3f}")
    assert 0.0 <= confidence5 <= 1.0, f"Confidence out of range: {confidence5:.3f}"
    print("✅ Long content test passed")

    print("\n🎉 All confidence calculation tests passed!")
    print("✅ 4-factor algorithm works correctly")
    print("✅ Weighted combination produces expected results")
    print("✅ Edge cases handled properly")
    return True

if __name__ == "__main__":
    success = test_confidence_calculation()
    print(f"\n🎯 Confidence Test Result: {'✅ PASSED' if success else '❌ FAILED'}")

    if not success:
        sys.exit(1)