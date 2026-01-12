"""Full pipeline test - Scraping, Parsing, Indexing, Q&A, Quality Evaluation."""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from pydantic import SecretStr
from src.scrapers.ir_scraper import IRScraper
from src.document.parser import DocumentParser
from src.vectorstore.chroma_store import ChromaStore
from src.llm.glm_client import GLMChat, OpenAIEmbeddings, GLMEmbeddings
from src.agents.quality_evaluator import IterativeAnswerer


def test_full_pipeline(company_name: str, website: str, test_questions: list):
    """Test the full NOVA pipeline."""
    print("\n" + "=" * 70)
    print(f"NOVA Full Pipeline Test - {company_name}")
    print("=" * 70)
    
    # Check API keys
    glm_api_key = os.getenv("GLM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not glm_api_key:
        print("❌ GLM_API_KEY not found in .env")
        return False
    
    print(f"\n✓ API Keys loaded")
    print(f"  - GLM: {'*' * 8}{glm_api_key[-4:]}")
    print(f"  - OpenAI: {'*' * 8}{openai_api_key[-4:] if openai_api_key else 'Not set'}")
    
    # Initialize components
    print("\n" + "-" * 70)
    print("STEP 1: Initialize Components")
    print("-" * 70)
    
    try:
        # LLM
        llm = GLMChat(
            api_key=SecretStr(glm_api_key),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4.7",  # Zhipu AI GLM-4.7 model
            temperature=0.7,
            timeout=120  # Shorter timeout for tests
        )
        print("  ✓ LLM initialized (GLM-4.7)")
        
        # Embeddings
        if openai_api_key:
            embeddings = OpenAIEmbeddings(
                api_key=SecretStr(openai_api_key),
                model="text-embedding-3-small"
            )
            print("  ✓ Embeddings initialized (OpenAI)")
        else:
            embeddings = GLMEmbeddings(
                api_key=SecretStr(glm_api_key),
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            print("  ✓ Embeddings initialized (GLM)")
        
        # Scraper
        scraper = IRScraper(timeout=30, max_depth=1)
        print("  ✓ Scraper initialized")
        
        # Parser
        parser = DocumentParser()
        print("  ✓ Parser initialized")
        
        # Vector store (use test collection)
        test_db_path = Path(__file__).parent.parent / "data" / "test_vectordb"
        vector_store = ChromaStore(
            persist_dir=test_db_path,
            embeddings=embeddings,
            collection_name=f"test_{company_name.lower().replace(' ', '_')}"
        )
        # Clear previous test data
        vector_store.clear_collection()
        print("  ✓ Vector store initialized (fresh collection)")
        
        # Quality evaluator
        answerer = IterativeAnswerer(
            llm=llm,
            target_score=8,
            max_iterations=2
        )
        print("  ✓ Quality evaluator initialized")
        
    except Exception as e:
        print(f"  ❌ Error initializing components: {e}")
        return False
    
    # Step 2: Scrape IR materials
    print("\n" + "-" * 70)
    print("STEP 2: Scrape IR Materials")
    print("-" * 70)
    
    def progress_cb(msg):
        print(f"  → {msg}")
    
    try:
        materials = scraper.scrape(website, progress_callback=progress_cb)
        print(f"\n  ✓ Found {len(materials)} materials")
        
        if not materials:
            print("  ⚠ No materials found. Testing with mock data...")
            return False
        
        # Show material types
        by_type = {}
        for m in materials:
            by_type.setdefault(m.material_type, []).append(m)
        
        print("\n  Material breakdown:")
        for mtype, items in sorted(by_type.items()):
            print(f"    - {mtype}: {len(items)}")
        
    except Exception as e:
        print(f"  ❌ Error scraping: {e}")
        return False
    
    # Step 3: Parse and index materials
    print("\n" + "-" * 70)
    print("STEP 3: Parse and Index Materials")
    print("-" * 70)
    
    all_chunks = []
    processed_count = 0
    failed_count = 0
    
    # Prioritize PDF materials for testing (they have actual content)
    pdf_materials = [m for m in materials if m.file_format == "pdf"]
    html_materials = [m for m in materials if m.file_format != "pdf"]
    
    # Take up to 5 PDFs and 5 HTML pages
    test_materials = pdf_materials[:5] + html_materials[:5]
    print(f"  Processing {len(test_materials)} materials ({len(pdf_materials[:5])} PDFs, {len(html_materials[:5])} HTML)...\n")
    
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    
    save_dir = Path(__file__).parent.parent / "data" / "test_processed"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, material in enumerate(test_materials):
        try:
            print(f"  [{i+1}/{len(test_materials)}] {material.title[:50]}...")
            
            material_metadata = {
                "url": material.url,
                "title": material.title,
                "material_type": material.material_type,
                "date": material.date or "",
                "file_format": material.file_format or "html",
                "company_name": company_name
            }
            
            async def parse():
                return await parser.parse_url(
                    url=material.url,
                    material_metadata=material_metadata,
                    save_dir=save_dir
                )
            
            chunks = asyncio.run(parse())
            
            if chunks:
                all_chunks.extend(chunks)
                processed_count += 1
                print(f"      ✓ Parsed {len(chunks)} chunks")
            else:
                print(f"      ⚠ No chunks extracted")
                failed_count += 1
                
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:50]}")
            failed_count += 1
    
    print(f"\n  Summary:")
    print(f"    - Processed: {processed_count}")
    print(f"    - Failed: {failed_count}")
    print(f"    - Total chunks: {len(all_chunks)}")
    
    if not all_chunks:
        print("  ❌ No chunks to index. Cannot continue.")
        return False
    
    # Index chunks
    print("\n  Indexing chunks...")
    try:
        indexed = vector_store.add_chunks(all_chunks, progress_callback=progress_cb)
        print(f"  ✓ Indexed {indexed} chunks")
    except Exception as e:
        print(f"  ❌ Error indexing: {e}")
        return False
    
    # Step 4: Test Q&A
    print("\n" + "-" * 70)
    print("STEP 4: Question & Answer Test")
    print("-" * 70)
    
    qa_results = []
    
    for i, question in enumerate(test_questions):
        print(f"\n  Question {i+1}: {question}")
        print("  " + "-" * 60)
        
        try:
            # Retrieve relevant chunks
            retrieved = vector_store.search(
                query=question,
                top_k=5,
                filter_company=company_name
            )
            
            if not retrieved:
                print("  ⚠ No relevant chunks found")
                qa_results.append({"question": question, "score": 0, "error": "No chunks"})
                continue
            
            print(f"  Retrieved {len(retrieved)} relevant chunks")
            
            # Build context
            context = "\n\n---\n\n".join([c.to_context_string() for c in retrieved])
            
            # Generate answer with quality evaluation
            print("  Generating answer...")
            answer, score, iterations = answerer.generate(
                question=question,
                context=context,
                progress_callback=lambda m: print(f"    → {m}")
            )
            
            qa_results.append({
                "question": question,
                "answer": answer,
                "score": score,
                "iterations": iterations
            })
            
            # Display answer
            print(f"\n  📊 Quality Score: {score}/10 (iterations: {iterations})")
            print(f"  {'✓ CEO-level quality' if score >= 8 else '⚠ Below CEO-level threshold'}")
            print(f"\n  Answer Preview:")
            print("  " + "-" * 60)
            # Show first 500 chars of answer
            preview = answer[:500] + "..." if len(answer) > 500 else answer
            for line in preview.split("\n"):
                print(f"  {line}")
            print("  " + "-" * 60)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            qa_results.append({"question": question, "score": 0, "error": str(e)})
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Pipeline Statistics:")
    print(f"  - Materials scraped: {len(materials)}")
    print(f"  - Materials processed: {processed_count}")
    print(f"  - Chunks indexed: {len(all_chunks)}")
    
    print(f"\n📝 Q&A Results:")
    total_score = 0
    ceo_level_count = 0
    
    for result in qa_results:
        score = result.get("score", 0)
        total_score += score
        if score >= 8:
            ceo_level_count += 1
        
        status = "✓" if score >= 8 else "⚠" if score >= 5 else "❌"
        print(f"  {status} Q: {result['question'][:40]}... → Score: {score}/10")
    
    avg_score = total_score / len(qa_results) if qa_results else 0
    
    print(f"\n📈 Quality Metrics:")
    print(f"  - Average score: {avg_score:.1f}/10")
    print(f"  - CEO-level answers: {ceo_level_count}/{len(qa_results)} ({100*ceo_level_count/len(qa_results):.0f}%)")
    
    # Test assertions
    print("\n" + "=" * 70)
    print("TEST ASSERTIONS")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Materials found
    tests_total += 1
    if len(materials) > 0:
        print("✓ PASS: Found IR materials")
        tests_passed += 1
    else:
        print("✗ FAIL: No materials found")
    
    # Test 2: Chunks indexed
    tests_total += 1
    if len(all_chunks) > 0:
        print("✓ PASS: Chunks indexed successfully")
        tests_passed += 1
    else:
        print("✗ FAIL: No chunks indexed")
    
    # Test 3: Answers generated
    tests_total += 1
    answers_generated = sum(1 for r in qa_results if r.get("answer"))
    if answers_generated > 0:
        print(f"✓ PASS: Generated {answers_generated} answers")
        tests_passed += 1
    else:
        print("✗ FAIL: No answers generated")
    
    # Test 4: Quality threshold
    tests_total += 1
    if avg_score >= 6:
        print(f"✓ PASS: Average quality score {avg_score:.1f}/10 >= 6")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Average quality score {avg_score:.1f}/10 < 6")
    
    # Test 5: At least one CEO-level answer
    tests_total += 1
    if ceo_level_count > 0:
        print(f"✓ PASS: {ceo_level_count} CEO-level answer(s) achieved")
        tests_passed += 1
    else:
        print("✗ FAIL: No CEO-level answers (score >= 8)")
    
    print(f"\n{'=' * 70}")
    print(f"FINAL: {tests_passed}/{tests_total} tests passed")
    print("=" * 70)
    
    return tests_passed >= 3  # At least 3 tests should pass


if __name__ == "__main__":
    # Test with Coca-Cola
    test_questions = [
        "What are Coca-Cola's key financial highlights from the latest quarterly report?",
        "What is Coca-Cola's strategic direction and growth initiatives?",
        "How is Coca-Cola performing in terms of revenue and profit margins?",
    ]
    
    success = test_full_pipeline(
        company_name="The Coca-Cola Company",
        website="https://www.coca-colacompany.com",
        test_questions=test_questions
    )
    
    sys.exit(0 if success else 1)
