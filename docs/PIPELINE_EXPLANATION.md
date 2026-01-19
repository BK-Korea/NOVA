# NOVA 데이터 처리 파이프라인 상세 설명

## 전체 프로세스 개요

```
웹페이지 → 스크래핑 → 문서 분류 → 파싱 → 청킹 → 임베딩 → 벡터 DB 저장
```

---

## 1단계: 웹페이지 스크래핑 (IR Scraper)

### 위치: `src/scrapers/ir_scraper.py`

### 프로세스:

#### 1.1 IR 섹션 찾기 (`_find_ir_pages`)
```python
# 서브도메인 탐색
ir.jobyaviation.com
investors.jobyaviation.com

# 경로 탐색
/investors/
/investor-relations/
/financials/

# 홈페이지 크롤링
홈페이지에서 "Investor", "IR" 링크 찾기
```

#### 1.2 페이지 스크래핑 (`_scrape_page`)
```python
# HTML 페이지 다운로드
response = httpx.get(page_url)

# BeautifulSoup으로 파싱
soup = BeautifulSoup(response.text, "lxml")

# 모든 <a> 태그 찾기
for link in soup.find_all("a", href=True):
    href = link.get("href")  # 예: "/news/article.pdf"
    full_url = urljoin(page_url, href)  # 절대 URL로 변환
```

#### 1.3 링크 분류 (`_classify_link`)

**파일 확장자 확인:**
```python
FILE_EXTENSIONS = {
    ".pdf": "pdf",
    ".ppt": "ppt", 
    ".pptx": "pptx",
    ".doc": "doc",
    ".docx": "docx",
    ".xls": "xls",      # ← Excel 파일
    ".xlsx": "xlsx",   # ← Excel 파일
    ".html": "html",
    ".htm": "html",
}
```

**현재 처리되는 파일 타입:**
- ✅ **PDF**: `.pdf` → `file_format = "pdf"`
- ✅ **PowerPoint**: `.ppt`, `.pptx` → `file_format = "ppt"/"pptx"`
- ✅ **Word**: `.doc`, `.docx` → `file_format = "doc"/"docx"`
- ✅ **Excel**: `.xls`, `.xlsx` → `file_format = "xls"/"xlsx"` (분류만 됨, 파싱은 안 됨)
- ✅ **HTML**: `.html`, `.htm` 또는 확장자 없음 → `file_format = "html"`
- ❌ **음성파일**: `.mp3`, `.mp4`, `.wav` → **현재 분류되지 않음**
- ❌ **비디오**: `.mp4`, `.mov` → **현재 분류되지 않음**

**자료 타입 분류:**
```python
material_type = {
    "sec_filing": SEC 제출 자료 (자동 제외됨)
    "shareholder_letter": 주주 서한
    "quarterly_report": 분기 보고서
    "presentation": 프레젠테이션
    "news": 뉴스/보도자료
    "ir_pitch": IR 개요
    "other": 기타
}
```

**결과: `IRMaterial` 객체 생성**
```python
IRMaterial(
    url="https://ir.jobyaviation.com/presentation.pdf",
    title="Q3 2024 Earnings Presentation",
    material_type="presentation",
    file_format="pdf",
    date="2024-10-15",
    source_page="https://ir.jobyaviation.com/presentations/"
)
```

---

## 2단계: 문서 파싱 (Document Parser)

### 위치: `src/document/parser.py`

### 프로세스: `parse_url()`

#### 2.1 Content-Type 확인
```python
content_type = response.headers.get("content-type", "").lower()

if "html" in content_type:
    # HTML 파싱
elif "pdf" in content_type:
    # PDF 파싱
else:
    # 미지원 형식 → 스킵
```

#### 2.2 파일 타입별 처리

**HTML 파일:**
```python
# 1. BeautifulSoup으로 텍스트 추출
soup = BeautifulSoup(response.text, "lxml")
# 스크립트, 스타일, 네비게이션 제거
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
text = soup.get_text()

# 2. unstructured 라이브러리로 상세 파싱
elements = partition_html(text=response.text)

# 3. 청크 생성
chunks = _create_chunks_from_elements(elements, metadata)
```

**PDF 파일:**
```python
# 1. PDF 다운로드 (임시 저장)
temp_path = save_dir / "temp.pdf"
with open(temp_path, "wb") as f:
    f.write(response.content)

# 2. unstructured로 PDF 파싱
# 전략: "fast" → "auto" → "hi_res" 순서로 시도
elements = partition_pdf(
    filename=str(temp_path),
    strategy="fast",  # 실패하면 "auto", "hi_res" 시도
    extract_images_in_pdf=False
)

# 3. PyPDF2 fallback (unstructured 실패 시)
# 텍스트만 추출
```

**PowerPoint 파일:**
```python
# unstructured로 PPT 파싱
elements = partition_pptx(filename=str(file_path))

# 각 슬라이드의 텍스트 추출
```

**Word 파일:**
```python
# unstructured로 DOCX 파싱
from unstructured.partition.docx import partition_docx
elements = partition_docx(filename=str(file_path))
```

**Excel 파일:**
```python
# ⚠️ 현재 파싱되지 않음!
# FILE_EXTENSIONS에는 있지만 parser.parse_url()에서 처리 안 함
# Excel 파일은 스킵됨
```

**음성/비디오 파일:**
```python
# ⚠️ 현재 전혀 처리되지 않음!
# FILE_EXTENSIONS에도 없고, 파서에도 없음
# .mp3, .mp4 파일은 스크래퍼에서도 분류되지 않음
```

#### 2.3 청킹 (Chunking)

**청크 생성 규칙:**
```python
chunk_size = 1000  # 기본 1000자
chunk_overlap = 200  # 200자 겹침

# 예시:
# 청크 1: 0-1000자
# 청크 2: 800-1800자 (200자 겹침)
# 청크 3: 1600-2600자
```

**메타데이터 포함:**
```python
DocumentChunk(
    content="청크 텍스트 내용...",
    metadata={
        "url": "https://ir.jobyaviation.com/presentation.pdf",
        "title": "Q3 2024 Earnings",
        "material_type": "presentation",
        "file_format": "pdf",
        "date": "2024-10-15",
        "company_name": "Joby Aviation",
        "chunk_index": 0,
        "page_number": 1  # PDF인 경우
    }
)
```

---

## 3단계: 벡터 DB 저장 (ChromaDB)

### 위치: `src/vectorstore/chroma_store.py`

### 프로세스: `add_chunks()`

#### 3.1 임베딩 생성
```python
# OpenAI 또는 GLM API 호출
embeddings = self.embeddings.embed_documents(texts)
# texts = ["청크1", "청크2", ...]
# embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
# 각 벡터는 1536차원 (OpenAI) 또는 모델에 따라 다름
```

#### 3.2 ChromaDB에 저장
```python
self.collection.add(
    documents=["청크 텍스트1", "청크 텍스트2", ...],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    metadatas=[
        {"url": "...", "title": "...", ...},
        {"url": "...", "title": "...", ...},
        ...
    ],
    ids=["url1_chunk0", "url1_chunk1", ...]
)
```

#### 3.3 저장 위치
```
/Users/bk/NOVA/data/vectordb/
├── chroma.sqlite3  # 메타데이터 DB
└── [collection_id]/
    ├── data_level0.bin  # 벡터 데이터
    ├── header.bin
    └── ...
```

---

## 4단계: 검색 및 답변 생성

### 위치: `src/agents/graph.py` → `ask()`

#### 4.1 유사도 검색
```python
# 질문을 임베딩으로 변환
query_embedding = embeddings.embed_query("26년 현재기준 사업모델에 대해서...")

# 벡터 유사도 검색 (코사인 유사도)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,  # 상위 10개
    where={"company_name": "Joby Aviation"}  # 필터링
)

# RetrievedChunk 반환
RetrievedChunk(
    content="청크 내용...",
    metadata={"url": "...", "title": "..."},
    score=0.85  # 유사도 점수
)
```

#### 4.2 컨텍스트 구성
```python
context = "\n\n---\n\n".join([
    f"{chunk.citation}\nURL: {chunk.metadata['url']}\n{chunk.content}"
    for chunk in retrieved
])
```

#### 4.3 LLM 답변 생성
```python
# GLM-4.7에 컨텍스트와 질문 전달
answer = llm.generate(
    question="26년 현재기준 사업모델...",
    context=context  # URL 포함된 컨텍스트
)
```

---

## 현재 파일 타입별 처리 현황

| 파일 타입 | 스크래핑 | 파싱 | 임베딩 | 상태 |
|-----------|---------|------|--------|------|
| **HTML** | ✅ | ✅ | ✅ | 완전 지원 |
| **PDF** | ✅ | ✅ | ✅ | 완전 지원 |
| **PPT/PPTX** | ✅ | ✅ | ✅ | 완전 지원 |
| **DOC/DOCX** | ✅ | ✅ | ✅ | 완전 지원 |
| **XLS/XLSX** | ✅ | ❌ | ❌ | 분류만 됨, 파싱 안 됨 |
| **MP3/WAV** | ❌ | ❌ | ❌ | 미지원 |
| **MP4/Video** | ❌ | ❌ | ❌ | 미지원 |
| **ZIP** | ✅ | ❌ | ❌ | 분류만 됨 (SEC 파일) |

---

## 문제점 및 개선 필요 사항

### 1. Excel 파일 파싱 미지원
- 현재: Excel 파일은 스크래퍼에서 발견되지만 파서에서 처리 안 됨
- 해결: `unstructured.partition.xlsx` 추가 필요

### 2. 음성/비디오 파일 미지원
- 현재: `.mp3`, `.mp4` 파일이 스크래퍼에서도 분류되지 않음
- 해결: 
  - 스크래퍼에 음성/비디오 확장자 추가
  - 음성 → 텍스트 변환 (Whisper API 등)
  - 비디오 → 자막 추출 또는 음성 변환

### 3. ZIP 파일 처리
- 현재: ZIP 파일은 분류만 되고 파싱 안 됨
- 해결: ZIP 압축 해제 후 내부 파일 파싱

---

## 데이터 흐름 다이어그램

```
[웹페이지]
    ↓
[IR Scraper]
    ├─ 링크 발견 (PDF, PPT, HTML 등)
    ├─ 파일 확장자 확인
    └─ IRMaterial 생성
        ↓
[Document Parser]
    ├─ URL에서 다운로드
    ├─ Content-Type 확인
    ├─ 파일 타입별 파싱
    │   ├─ HTML → BeautifulSoup + unstructured
    │   ├─ PDF → partition_pdf (unstructured)
    │   ├─ PPT → partition_pptx
    │   └─ DOCX → partition_docx
    └─ DocumentChunk 생성 (텍스트 + 메타데이터)
        ↓
[ChromaDB]
    ├─ 임베딩 생성 (OpenAI/GLM API)
    ├─ 벡터 + 메타데이터 저장
    └─ 영구 저장 (SQLite + 바이너리)
        ↓
[검색 시]
    ├─ 질문 → 임베딩
    ├─ 벡터 유사도 검색
    └─ RetrievedChunk 반환 (URL 포함)
        ↓
[LLM 답변]
    └─ 컨텍스트 + URL → 보고서 생성
```

---

## 메타데이터 구조

### IRMaterial (스크래퍼 출력)
```python
{
    "url": "https://ir.jobyaviation.com/presentation.pdf",
    "title": "Q3 2024 Earnings Presentation",
    "material_type": "presentation",
    "file_format": "pdf",
    "date": "2024-10-15",
    "source_page": "https://ir.jobyaviation.com/presentations/"
}
```

### DocumentChunk (파서 출력)
```python
{
    "content": "청크 텍스트 내용...",
    "metadata": {
        "url": "https://ir.jobyaviation.com/presentation.pdf",
        "title": "Q3 2024 Earnings Presentation",
        "material_type": "presentation",
        "file_format": "pdf",
        "date": "2024-10-15",
        "company_name": "Joby Aviation",
        "chunk_index": 0,
        "page_number": 1,
        "element_type": "NarrativeText"
    }
}
```

### ChromaDB 저장 형식
```python
{
    "id": "url_chunk0",
    "document": "청크 텍스트...",
    "embedding": [0.1, 0.2, 0.3, ...],  # 1536차원 벡터
    "metadata": {
        "url": "...",
        "title": "...",
        "material_type": "...",
        ...
    }
}
```

---

## 캐싱 메커니즘

### 중복 방지
```python
# URL 기반 중복 체크
if self.vector_store.check_existing(material.url):
    logger.debug(f"Already indexed: {material.url}")
    continue  # 스킵
```

### 저장 위치
- **원본 파일**: `data/processed/` (PDF, PPT 등)
- **벡터 DB**: `data/vectordb/` (ChromaDB)
- **메타데이터**: SQLite DB 내부

---

## 비용 발생 지점

1. **임베딩 생성**: OpenAI API 호출 (청크당 ~$0.00003)
2. **LLM 답변**: GLM API 호출 (질문당 ~$0.01-0.05)
3. **스크래핑/파싱**: 무료 (로컬 처리)

---

## 개선 제안

### Excel 파일 지원 추가
```python
# parser.py에 추가
def _parse_xlsx(self, file_path, metadata):
    from unstructured.partition.xlsx import partition_xlsx
    elements = partition_xlsx(filename=str(file_path))
    return self._create_chunks_from_elements(elements, metadata)
```

### 음성 파일 지원 추가
```python
# 스크래퍼에 추가
FILE_EXTENSIONS = {
    ...
    ".mp3": "audio",
    ".mp4": "video",
    ".wav": "audio",
}

# 파서에 추가
async def parse_audio(self, url, metadata):
    # Whisper API 또는 로컬 STT 사용
    transcript = transcribe_audio(url)
    return self._simple_chunk(transcript, metadata)
```
