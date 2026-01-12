"""Test script for IR scraper - Tests with Coca-Cola as example."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.ir_scraper import IRScraper, IRMaterial


def test_coca_cola():
    """Test scraping Coca-Cola IR materials."""
    print("\n" + "=" * 60)
    print("NOVA IR Scraper Test - Coca-Cola")
    print("=" * 60 + "\n")
    
    scraper = IRScraper(timeout=30, max_depth=1)
    
    # Coca-Cola's IR website
    test_urls = [
        "https://www.coca-colacompany.com",
        "https://investors.coca-colacompany.com",
    ]
    
    def progress_callback(msg):
        print(f"  → {msg}")
    
    all_materials = []
    
    for url in test_urls:
        print(f"\n📍 Testing: {url}")
        print("-" * 50)
        
        try:
            materials = scraper.scrape(
                company_website=url,
                progress_callback=progress_callback
            )
            
            all_materials.extend(materials)
            
            print(f"\n✓ Found {len(materials)} materials from {url}")
            
        except Exception as e:
            print(f"\n✗ Error scraping {url}: {e}")
    
    # Deduplicate
    seen = set()
    unique_materials = []
    for m in all_materials:
        if m.url not in seen:
            seen.add(m.url)
            unique_materials.append(m)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"TOTAL UNIQUE MATERIALS: {len(unique_materials)}")
    print("=" * 60)
    
    # Group by type
    by_type = {}
    for m in unique_materials:
        by_type.setdefault(m.material_type, []).append(m)
    
    print("\n📊 Materials by Type:")
    print("-" * 40)
    for mtype, items in sorted(by_type.items()):
        print(f"  {mtype}: {len(items)}")
    
    print("\n📄 Sample Materials (first 5 of each type):")
    print("-" * 40)
    
    for mtype, items in sorted(by_type.items()):
        print(f"\n  [{mtype.upper()}]")
        for m in items[:5]:
            title_short = m.title[:50] + "..." if len(m.title) > 50 else m.title
            format_str = f"[{m.file_format}]" if m.file_format else "[html]"
            print(f"    • {format_str} {title_short}")
            print(f"      URL: {m.url[:70]}...")
    
    # Test assertions
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Found any materials
    tests_total += 1
    if len(unique_materials) > 0:
        print("✓ PASS: Found IR materials")
        tests_passed += 1
    else:
        print("✗ FAIL: No materials found")
    
    # Test 2: Found multiple types
    tests_total += 1
    if len(by_type) >= 2:
        print(f"✓ PASS: Found {len(by_type)} different material types")
        tests_passed += 1
    else:
        print(f"✗ FAIL: Only {len(by_type)} material type(s) found")
    
    # Test 3: Found PDF documents
    tests_total += 1
    pdf_count = sum(1 for m in unique_materials if m.file_format == "pdf")
    if pdf_count > 0:
        print(f"✓ PASS: Found {pdf_count} PDF documents")
        tests_passed += 1
    else:
        print("✗ FAIL: No PDF documents found")
    
    # Test 4: Found presentations
    tests_total += 1
    if "presentation" in by_type or "quarterly_report" in by_type:
        print("✓ PASS: Found presentations or reports")
        tests_passed += 1
    else:
        print("✗ FAIL: No presentations or reports found")
    
    # Test 5: Found news/press releases
    tests_total += 1
    if "news" in by_type or any("press" in m.title.lower() for m in unique_materials):
        print("✓ PASS: Found news/press releases")
        tests_passed += 1
    else:
        print("✗ FAIL: No news/press releases found")
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    return tests_passed == tests_total


def test_joby_aviation():
    """Test scraping Joby Aviation IR materials."""
    print("\n" + "=" * 60)
    print("NOVA IR Scraper Test - Joby Aviation")
    print("=" * 60 + "\n")
    
    scraper = IRScraper(timeout=30, max_depth=1)
    
    test_urls = [
        "https://www.jobyaviation.com",
        "https://ir.jobyaviation.com",
    ]
    
    def progress_callback(msg):
        print(f"  → {msg}")
    
    all_materials = []
    
    for url in test_urls:
        print(f"\n📍 Testing: {url}")
        print("-" * 50)
        
        try:
            materials = scraper.scrape(
                company_website=url,
                progress_callback=progress_callback
            )
            all_materials.extend(materials)
            print(f"\n✓ Found {len(materials)} materials from {url}")
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    # Deduplicate
    seen = set()
    unique_materials = []
    for m in all_materials:
        if m.url not in seen:
            seen.add(m.url)
            unique_materials.append(m)
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(unique_materials)} unique materials found")
    print("=" * 60)
    
    # Group by type
    by_type = {}
    for m in unique_materials:
        by_type.setdefault(m.material_type, []).append(m)
    
    for mtype, items in sorted(by_type.items()):
        print(f"  {mtype}: {len(items)}")
    
    return len(unique_materials) > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NOVA IR Scraper")
    parser.add_argument(
        "--company", 
        type=str, 
        default="coca-cola",
        choices=["coca-cola", "joby", "all"],
        help="Company to test"
    )
    
    args = parser.parse_args()
    
    if args.company == "coca-cola":
        success = test_coca_cola()
    elif args.company == "joby":
        success = test_joby_aviation()
    else:
        success1 = test_coca_cola()
        success2 = test_joby_aviation()
        success = success1 and success2
    
    sys.exit(0 if success else 1)
