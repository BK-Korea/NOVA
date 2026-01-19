# 음성 파일 처리 가이드

## 가능 여부

**✅ 가능합니다!** 음성 파일을 텍스트로 변환(전사)하여 DB에 저장할 수 있습니다.

---

## 구현 방법

### 옵션 1: OpenAI Whisper API (추천)

**장점:**
- 매우 정확한 전사 (다국어 지원)
- API 사용이 간단
- 저렴한 비용

**단점:**
- API 비용 발생
- 인터넷 연결 필요

**비용:**
- **$0.006 per minute** (분당 약 0.6센트)
- 예시:
  - 10분 음성 = $0.06
  - 1시간 음성 = $0.36
  - 10시간 음성 = $3.60

### 옵션 2: 로컬 Whisper (무료)

**장점:**
- 완전 무료
- 오프라인 가능

**단점:**
- 느림 (CPU/GPU 의존)
- 모델 다운로드 필요 (약 1-3GB)
- 리소스 많이 사용

---

## 구현 코드 예시

### OpenAI Whisper API 사용

```python
import openai
from pathlib import Path

async def transcribe_audio(
    audio_url: str,
    api_key: str,
    language: str = "ko"  # 한국어
) -> str:
    """
    음성 파일을 텍스트로 변환
    
    Args:
        audio_url: 음성 파일 URL
        api_key: OpenAI API 키
        language: 언어 코드 (ko, en, ja 등)
    
    Returns:
        전사된 텍스트
    """
    # 1. 음성 파일 다운로드
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(audio_url)
        audio_data = response.content
    
    # 2. 임시 파일로 저장
    temp_path = Path("/tmp/audio.mp3")
    with open(temp_path, "wb") as f:
        f.write(audio_data)
    
    # 3. Whisper API 호출
    client = openai.OpenAI(api_key=api_key)
    
    with open(temp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="text"
        )
    
    # 4. 임시 파일 삭제
    temp_path.unlink()
    
    return transcript
```

### 비용 계산 예시

```python
def calculate_transcription_cost(duration_minutes: float) -> float:
    """
    전사 비용 계산
    
    Args:
        duration_minutes: 음성 길이 (분)
    
    Returns:
        비용 (USD)
    """
    COST_PER_MINUTE = 0.006
    return duration_minutes * COST_PER_MINUTE

# 예시
print(f"10분 음성: ${calculate_transcription_cost(10):.2f}")
print(f"1시간 음성: ${calculate_transcription_cost(60):.2f}")
```

---

## 전체 파이프라인 통합

### 1. 스크래퍼에 음성 파일 추가

```python
# src/scrapers/ir_scraper.py
FILE_EXTENSIONS = {
    # ... 기존 확장자들
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".mp4": "video",  # 비디오도 음성 추출 가능
    ".webm": "video",
}
```

### 2. 파서에 음성 처리 추가

```python
# src/document/parser.py
async def parse_audio(
    self,
    url: str,
    material_metadata: Dict[str, Any],
    save_dir: Optional[Path] = None
) -> List[DocumentChunk]:
    """
    음성 파일을 텍스트로 변환하여 청크 생성
    """
    import openai
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY not found for audio transcription")
        return []
    
    try:
        # 1. 음성 파일 다운로드
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=300)  # 5분 타임아웃
            audio_data = response.content
        
        # 2. 임시 파일 저장
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            import hashlib
            filename = hashlib.md5(url.encode()).hexdigest()[:10] + ".mp3"
            temp_path = save_dir / filename
            
            with open(temp_path, "wb") as f:
                f.write(audio_data)
        else:
            import tempfile
            temp_path = Path(tempfile.mktemp(suffix=".mp3"))
            with open(temp_path, "wb") as f:
                f.write(audio_data)
        
        # 3. Whisper API로 전사
        client = openai.OpenAI(api_key=api_key)
        
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko",  # 한국어 (자동 감지도 가능)
                response_format="text"
            )
        
        # 4. 전사된 텍스트를 청크로 변환
        chunks = self._simple_chunk(
            transcript,
            {**material_metadata, "file_format": "audio", "transcription": True},
            temp_path
        )
        
        # 5. 임시 파일 삭제 (선택적)
        # temp_path.unlink()
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error transcribing audio {url}: {e}")
        return []
```

### 3. parse_url()에 통합

```python
# src/document/parser.py - parse_url() 메서드에 추가
elif "audio" in content_type or url.endswith((".mp3", ".wav", ".m4a")):
    return await self.parse_audio(url, material_metadata, save_dir)
elif "video" in content_type or url.endswith((".mp4", ".webm")):
    # 비디오에서도 음성 추출 가능
    return await self.parse_audio(url, material_metadata, save_dir)
```

---

## 비용 분석

### 시나리오별 비용

| 시나리오 | 음성 길이 | 전사 비용 | 임베딩 비용 | 총 비용 |
|---------|----------|----------|-----------|---------|
| **짧은 발표** | 10분 | $0.06 | $0.01 | **$0.07** |
| **분기 실적 발표** | 30분 | $0.18 | $0.03 | **$0.21** |
| **연례 총회** | 1시간 | $0.36 | $0.06 | **$0.42** |
| **10개 회사, 각 30분** | 5시간 | $1.80 | $0.30 | **$2.10** |

### 기존 비용과 비교

| 작업 | 기존 비용 | 음성 추가 시 |
|------|----------|------------|
| PDF 100개 파싱 | $0.00 (무료) | $0.00 |
| 임베딩 1000 청크 | $0.03 | $0.03 |
| 음성 10분 전사 | - | **$0.06** |
| **총합** | **$0.03** | **$0.09** |

**결론**: 음성 파일 추가 시 비용이 증가하지만, 매우 저렴합니다!

---

## 대안: 로컬 Whisper (무료)

### 설치

```bash
pip install openai-whisper
```

### 사용

```python
import whisper

def transcribe_local(audio_path: str) -> str:
    """
    로컬 Whisper로 전사 (무료, 느림)
    """
    model = whisper.load_model("base")  # 또는 "small", "medium", "large"
    result = model.transcribe(audio_path, language="ko")
    return result["text"]
```

**모델 크기별 성능:**
- `tiny`: 가장 빠름, 정확도 낮음
- `base`: 균형잡힘 (권장)
- `small`: 더 정확
- `medium`: 매우 정확
- `large`: 가장 정확, 매우 느림

---

## 권장 사항

### 프로덕션 환경
- **OpenAI Whisper API 사용** (빠르고 정확)
- 비용이 매우 저렴함 ($0.006/분)
- 자동 언어 감지 지원

### 개발/테스트 환경
- **로컬 Whisper 사용** (무료)
- 오프라인 테스트 가능
- 비용 걱정 없음

---

## 구현 체크리스트

- [ ] 스크래퍼에 음성 확장자 추가 (`.mp3`, `.wav`, `.m4a`)
- [ ] 파서에 `parse_audio()` 메서드 추가
- [ ] `parse_url()`에 음성 파일 분기 추가
- [ ] OpenAI API 키 확인
- [ ] 비용 모니터링 설정
- [ ] 에러 핸들링 (타임아웃, 파일 크기 제한 등)

---

## 주의사항

1. **파일 크기 제한**: OpenAI Whisper는 최대 25MB
2. **타임아웃**: 긴 음성 파일은 타임아웃 설정 필요
3. **비용 모니터링**: 사용량 추적 권장
4. **언어 설정**: 한국어는 `language="ko"` 명시

---

## 예상 사용량

**일반적인 IR 자료:**
- 뉴스 발표: 5-10분
- 분기 실적 발표: 20-30분
- 연례 총회: 1-2시간

**월간 예상 비용:**
- 10개 회사 × 5개 음성 × 20분 = 1000분
- 비용: 1000분 × $0.006 = **$6.00/월**

매우 저렴합니다! 🎉
