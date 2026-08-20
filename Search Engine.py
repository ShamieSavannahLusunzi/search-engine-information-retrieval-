import sqlite3
import math
import re
from collections import defaultdict


# ========== MUST USE THE SAME STEMMER AS YOUR INDEXER ==========
class PorterStemmer:
    """A simplified Porter Stemmer for demonstration"""

    def stem(self, word):
        # Basic stemming rules - in practice, use a proper stemmer
        word = word.lower()
        if word.endswith('sses'):
            word = word[:-2]
        elif word.endswith('ies'):
            word = word[:-2]
        elif word.endswith('ss'):
            pass
        elif word.endswith('s'):
            word = word[:-1]
        return word


# Initialize stemmer (SAME AS INDEXER)
stemmer = PorterStemmer()

# ========== MUST USE THE SAME STOP WORDS AS YOUR INDEXER ==========
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'i', 'you', 'we', 'they',
    'this', 'that', 'these', 'those', 'have', 'has', 'had', 'do',
    'does', 'did', 'but', 'or', 'if', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', 'now'
}


class SearchEngine:
    def __init__(self, db_path="indexer_part2_tfidf.db"):
        """Initialize search engine with database connection."""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def process_query(self, query_text):
        """Process user query EXACTLY like the indexer did."""
        print(f"[DEBUG] Raw query: '{query_text}'")

        # 1. Split and lowercase
        raw_terms = query_text.lower().split()
        processed_terms = []

        for term in raw_terms:
            # 2. Skip if stop word (SAME STOP WORDS)
            if term in STOP_WORDS:
                print(f"[DEBUG] Filtered stop word: '{term}'")
                continue

            # 3. Skip if too short (<2 characters)
            if len(term) < 2:
                print(f"[DEBUG] Filtered short term: '{term}'")
                continue

            # 4. Skip if starts with punctuation
            if term and term[0] in '!@#$%^&*()_+-=[]{}|;:",.<>?/~`\'"':
                print(f"[DEBUG] Filtered punctuation-start: '{term}'")
                continue

            # 5. Skip if all digits
            if term.isdigit():
                print(f"[DEBUG] Filtered numeric: '{term}'")
                continue

            # 6. Apply stemming (EXACT SAME STEMMER AS INDEXER)
            stemmed = stemmer.stem(term)
            if stemmed:
                processed_terms.append(stemmed)
                print(f"[DEBUG] Term '{term}' -> stemmed '{stemmed}'")
            else:
                print(f"[DEBUG] Term '{term}' -> empty after stemming")

        print(f"[DEBUG] Final processed terms: {processed_terms}")
        return processed_terms

    def get_documents_with_all_terms(self, query_terms):
        """Find documents containing ALL query terms."""
        if not query_terms:
            return []

        # First, verify each term exists in the index
        valid_terms = []
        for term in query_terms:
            self.cursor.execute(
                "SELECT TermId, DocumentFrequency FROM TermDictionary WHERE Term = ?",
                (term,)
            )
            result = self.cursor.fetchone()
            if result:
                term_id, df = result
                valid_terms.append(term)
                print(f"[DEBUG] Term '{term}' found in index (ID: {term_id}, df: {df})")
            else:
                # Try to find similar terms (for debugging)
                self.cursor.execute(
                    "SELECT Term FROM TermDictionary WHERE Term LIKE ?",
                    (term + '%',)
                )
                similar = self.cursor.fetchall()
                if similar:
                    print(f"[DEBUG] Term '{term}' not found, but similar terms exist: {[s[0] for s in similar]}")
                else:
                    print(f"[DEBUG] Term '{term}' not found in index and no similar terms")

        if not valid_terms:
            print("[WARNING] No valid terms found in index")
            return []

        # Build SQL query for documents containing ALL valid terms
        placeholders = ','.join('?' for _ in valid_terms)

        query = f"""
        SELECT p.DocId, d.DocumentName
        FROM Posting p
        JOIN TermDictionary t ON p.TermId = t.TermId
        JOIN DocumentDictionary d ON p.DocId = d.DocId
        WHERE t.Term IN ({placeholders})
        GROUP BY p.DocId
        HAVING COUNT(DISTINCT t.TermId) = ?
        """

        print(f"[DEBUG] Executing SQL for terms: {valid_terms}")
        self.cursor.execute(query, valid_terms + [len(valid_terms)])
        results = self.cursor.fetchall()

        print(f"[DEBUG] Found {len(results)} documents with ALL terms")
        return results

    def calculate_cosine_similarity(self, query_terms, doc_id):
        """Calculate cosine similarity between query and document."""
        # Get total number of documents
        self.cursor.execute("SELECT COUNT(*) FROM DocumentDictionary")
        N = self.cursor.fetchone()[0]

        # Build query vector (using idf weights)
        query_vector = {}
        query_norm_squared = 0

        for term in query_terms:
            # Get document frequency for term
            self.cursor.execute(
                "SELECT DocumentFrequency FROM TermDictionary WHERE Term = ?",
                (term,)
            )
            result = self.cursor.fetchone()

            if result:
                df = result[0]
                if df > 0 and N > 0:
                    idf = math.log(N / df)
                    query_vector[term] = idf
                    query_norm_squared += idf * idf
                else:
                    query_vector[term] = 0
            else:
                query_vector[term] = 0

        # Build document vector (using stored tf-idf)
        doc_vector = {}
        doc_norm_squared = 0

        for term in query_terms:
            self.cursor.execute("""
                SELECT p.tfidf 
                FROM Posting p
                JOIN TermDictionary t ON p.TermId = t.TermId
                WHERE t.Term = ? AND p.DocId = ?
            """, (term, doc_id))

            result = self.cursor.fetchone()
            if result and result[0] is not None:
                tfidf = float(result[0])
                doc_vector[term] = tfidf
                doc_norm_squared += tfidf * tfidf
            else:
                doc_vector[term] = 0

        # Calculate dot product
        dot_product = sum(query_vector[term] * doc_vector[term] for term in query_terms)

        # Calculate norms
        query_norm = math.sqrt(query_norm_squared) if query_norm_squared > 0 else 1
        doc_norm = math.sqrt(doc_norm_squared) if doc_norm_squared > 0 else 1

        # Avoid division by zero
        if query_norm == 0 or doc_norm == 0:
            return 0.0

        similarity = dot_product / (query_norm * doc_norm)
        return similarity

    def search(self, query_text):
        """Main search function."""
        print("=" * 60)
        print(f"Searching for: '{query_text}'")
        print("=" * 60)

        # Process query
        query_terms = self.process_query(query_text)

        if not query_terms:
            print("[ERROR] No valid query terms after processing.")
            return []

        # Find candidate documents
        candidates = self.get_documents_with_all_terms(query_terms)

        if not candidates:
            print(f"\n[INFO] No documents contain ALL terms: {query_terms}")

            # Fallback: find docs with ANY term
            placeholders = ','.join('?' for _ in query_terms)
            fallback_query = f"""
            SELECT DISTINCT p.DocId, d.DocumentName
            FROM Posting p
            JOIN TermDictionary t ON p.TermId = t.TermId
            JOIN DocumentDictionary d ON p.DocId = d.DocId
            WHERE t.Term IN ({placeholders})
            """

            self.cursor.execute(fallback_query, query_terms)
            candidates = self.cursor.fetchall()
            print(f"[DEBUG] Fallback: Found {len(candidates)} documents with ANY term")

        print(f"\nFound {len(candidates)} candidate document(s)")

        # Calculate similarity for each candidate
        results = []
        for doc_id, doc_name in candidates:
            similarity = self.calculate_cosine_similarity(query_terms, doc_id)
            results.append({
                'doc_id': doc_id,
                'doc_name': doc_name,
                'similarity': similarity
            })

        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results

    def display_results(self, results, max_display=20):
        """Display results in required format."""
        print(f"\n{'=' * 60}")
        if not results:
            print("NO RESULTS FOUND")
            print(f"{'=' * 60}")
            print("Total candidate documents retrieved: 0")
            print("Simpson algorithm")
            return

        display_count = min(max_display, len(results))
        print(f"TOP {display_count} RESULTS (Sorted by Cosine Similarity)")
        print(f"{'=' * 60}")

        for i, result in enumerate(results[:max_display], 1):
            print(f"{i:2}. Document: {result['doc_name']}")
            print(f"    Cosine Similarity: {result['similarity']:.6f}")
            print()

        print(f"Total candidate documents retrieved: {len(results)}")
        print("Simpson algorithm")

    def close(self):
        """Close database connection."""
        self.conn.close()


def check_database_directly():
    """Check database contents directly - CRITICAL for debugging."""
    print("\n" + "=" * 60)
    print("DIRECT DATABASE CHECK")
    print("=" * 60)

    try:
        conn = sqlite3.connect("indexer_part2_tfidf.db")
        cursor = conn.cursor()

        # Check document count
        cursor.execute("SELECT COUNT(*) FROM DocumentDictionary")
        doc_count = cursor.fetchone()[0]
        print(f"✓ Documents in database: {doc_count}")

        # Check term count
        cursor.execute("SELECT COUNT(*) FROM TermDictionary")
        term_count = cursor.fetchone()[0]
        print(f"✓ Unique terms in database: {term_count}")

        # Check posting count
        cursor.execute("SELECT COUNT(*) FROM Posting")
        posting_count = cursor.fetchone()[0]
        print(f"✓ Postings in database: {posting_count}")

        # Check for specific terms with stemming
        test_words = ['home', 'mortgage']
        print(f"\nChecking stems for '{test_words}':")

        for word in test_words:
            # Apply the SAME stemmer
            stemmed = stemmer.stem(word.lower())
            print(f"\n  Word: '{word}' -> Stemmed: '{stemmed}'")

            # Check if stemmed version exists
            cursor.execute(
                "SELECT Term, DocumentFrequency FROM TermDictionary WHERE Term = ?",
                (stemmed,)
            )
            result = cursor.fetchone()

            if result:
                term, df = result
                print(f"  ✓ Found in index: '{term}' (df={df})")

                # Show some documents containing this term
                cursor.execute("""
                    SELECT d.DocumentName, p.termfreq, p.tfidf
                    FROM Posting p
                    JOIN DocumentDictionary d ON p.DocId = d.DocId
                    JOIN TermDictionary t ON p.TermId = t.TermId
                    WHERE t.Term = ?
                    LIMIT 3
                """, (stemmed,))

                docs = cursor.fetchall()
                print(f"  Sample documents containing '{term}':")
                for doc_name, tf, tfidf in docs:
                    print(f"    - {doc_name} (tf={tf}, tfidf={tfidf:.4f})")
            else:
                print(f"  ✗ Not found in index")

                # Show similar terms
                cursor.execute(
                    "SELECT Term FROM TermDictionary WHERE Term LIKE ? LIMIT 5",
                    (stemmed[:3] + '%',)  # First 3 chars
                )
                similar = cursor.fetchall()
                if similar:
                    print(f"  Similar terms: {[s[0] for s in similar]}")

        # Show some sample terms from database
        print(f"\nSample of 15 terms from database:")
        cursor.execute("SELECT Term, DocumentFrequency FROM TermDictionary LIMIT 15")
        for term, df in cursor.fetchall():
            print(f"  '{term}' (in {df} documents)")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to run the search engine."""
    print("TF-IDF SEARCH ENGINE (Unit 4 Assignment)")
    print("=" * 60)

    # First, check what's in the database
    check_database_directly()

    # Initialize search engine
    engine = SearchEngine("indexer_part2_tfidf.db")

    try:
        # REQUIRED: Test with "home mortgage" first
        print("\n" + "=" * 60)
        print("REQUIRED TEST QUERY: 'home mortgage'")
        print("=" * 60)

        query = "home mortgage"
        results = engine.search(query)
        engine.display_results(results, max_display=20)

        # If no results, try alternative queries
        if not results:
            print("\n" + "=" * 60)
            print("TRYING ALTERNATIVE QUERIES")
            print("=" * 60)

            alt_queries = [
                "computer",
                "computer science",
                "data",
                "information retrieval",
                "system",
                "algorithm"
            ]

            for alt_query in alt_queries:
                print(f"\nTrying: '{alt_query}'")
                alt_results = engine.search(alt_query)
                if alt_results:
                    engine.display_results(alt_results, max_display=10)
                    print(f"\n✓ Found results for '{alt_query}'")
                    break
                else:
                    print(f"✗ No results for '{alt_query}'")

        # Optional: Let user enter their own query
        print("\n" + "=" * 60)
        print("OPTIONAL: CUSTOM QUERY")
        print("=" * 60)
        user_query = input("\nEnter your own query (or press Enter to skip): ").strip()

        if user_query:
            print(f"\nSearching for: '{user_query}'")
            user_results = engine.search(user_query)
            engine.display_results(user_results, max_display=20)

    finally:
        engine.close()
        print("\n" + "=" * 60)
        print("SEARCH COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
