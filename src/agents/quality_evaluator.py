"""Quality evaluation and answer refinement for CEO-level reports."""
import logging
from typing import Dict, Any, List, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
import re

from ..vectorstore.chroma_store import RetrievedChunk

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """Evaluates answer quality and provides improvement suggestions."""

    EVALUATION_PROMPT = """당신은 글로벌 투자은행의 **리서치 디렉터**입니다.
애널리스트가 작성한 기업 IR 자료 분석 보고서를 **CEO에게 보고하기 전에** 품질을 평가합니다.

## 평가 기준 (각 2점, 총 10점)

1. **구체성 (Specificity)**: 구체적인 수치(금액, 비율, 날짜)가 포함되어 있는가?
   - 0점: 수치 없음
   - 1점: 일부 수치 있음
   - 2점: 충분한 수치와 데이터

2. **출처 명확성 (Citation Quality)**: 모든 사실에 정확한 출처가 있는가?
   - 0점: 출처 없음
   - 1점: 일부 출처
   - 2점: 모든 주요 사실에 출처

3. **원문 인용 (Direct Quotes)**: 핵심 문장이 직접 인용되어 있는가?
   - 0점: 원문 인용 없음
   - 1점: 1-2개 인용
   - 2점: 3개 이상 중요 원문 인용

4. **분석 깊이 (Depth)**: 단순 나열이 아닌 심층 분석이 있는가?
   - 0점: 단순 나열
   - 1점: 기본 분석
   - 2점: 인사이트와 함의 분석

5. **완결성 (Completeness)**: 질문에 완전히 답변했는가?
   - 0점: 부분적 답변
   - 1점: 대부분 답변
   - 2점: 완전하고 포괄적

## 출력 형식 (정확히 따를 것)

```
SCORES:
- Specificity: X/2
- Citation: X/2
- Quotes: X/2
- Depth: X/2
- Completeness: X/2
- TOTAL: X/10

WEAKNESSES:
1. [가장 부족한 점]
2. [두 번째 부족한 점]
3. [세 번째 부족한 점]

IMPROVEMENTS:
1. [구체적인 개선 지시 1]
2. [구체적인 개선 지시 2]
3. [구체적인 개선 지시 3]
```"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def evaluate(self, question: str, answer: str) -> Tuple[int, str, List[str]]:
        """
        Evaluate answer quality.

        Returns:
            Tuple of (score, weaknesses_text, improvement_suggestions)
        """
        messages = [
            SystemMessage(content=self.EVALUATION_PROMPT),
            HumanMessage(content=f"""## 질문
{question}

## 분석 보고서
{answer}

---
위 보고서를 평가하세요.""")
        ]

        response = self.llm.invoke(messages)
        evaluation = response.content

        # Parse score
        score = self._parse_score(evaluation)

        # Parse improvements
        improvements = self._parse_improvements(evaluation)

        return score, evaluation, improvements

    def _parse_score(self, evaluation: str) -> int:
        """Extract total score from evaluation."""
        match = re.search(r'TOTAL:\s*(\d+)/10', evaluation)
        if match:
            return int(match.group(1))
        return 5  # Default if parsing fails

    def _parse_improvements(self, evaluation: str) -> List[str]:
        """Extract improvement suggestions."""
        improvements = []
        in_improvements = False

        for line in evaluation.split('\n'):
            if 'IMPROVEMENTS:' in line:
                in_improvements = True
                continue
            if in_improvements:
                match = re.match(r'\d+\.\s*(.+)', line.strip())
                if match:
                    improvements.append(match.group(1))
                elif line.strip() == '' and improvements:
                    break

        return improvements[:3]  # Max 3 improvements


class AnswerRefiner:
    """Refines answer based on improvement suggestions."""

    REFINEMENT_PROMPT = """당신은 기업 IR 자료 분석 전문가입니다.
기존 분석 보고서를 **개선 지시사항**에 따라 **대폭 강화**해야 합니다.

## 개선 원칙

1. **기존 내용 유지**: 기존의 정확한 정보는 그대로 유지
2. **약점 보완**: 지적된 약점을 구체적으로 개선
3. **수치 추가**: 문서에서 관련 수치를 더 찾아 추가
4. **원문 인용 추가**: 핵심 문장을 직접 인용
5. **분석 심화**: 단순 나열을 넘어 의미와 함의 분석
6. **레퍼런스 추가**: 각 핵심 내용 바로 아래에 하이퍼링크 추가

## 출력 형식

- **마크다운 형식** 유지
- 다음 구조를 반드시 따르세요:
  1. 사업의 일반현황
  2. 주요연혁
  3. 주요투자 유치이력
  4. 주요인사
  5. 비즈니스모델
  6. 인증/기체개발현황
  7. 주요 재무현황

- 레퍼런스는 각 섹션의 핵심 내용 바로 아래에 하이퍼링크로 배치
- 데이터 없으면 "없음" 명시
- 모든 수치에 출처 필수"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def refine(
        self,
        question: str,
        current_answer: str,
        context: str,
        improvements: List[str]
    ) -> str:
        """Refine the answer based on improvement suggestions."""

        improvements_text = "\n".join([f"{i+1}. {imp}" for i, imp in enumerate(improvements)])

        messages = [
            SystemMessage(content=self.REFINEMENT_PROMPT),
            HumanMessage(content=f"""## 원본 질문
{question}

## 현재 보고서
{current_answer}

## 개선 지시사항
{improvements_text}

## 참고할 IR 자료 컨텍스트
{context}

---
위 개선 지시사항을 반영하여 보고서를 **대폭 강화**하세요.
더 많은 수치, 더 많은 원문 인용, 더 깊은 분석을 포함하세요.""")
        ]

        response = self.llm.invoke(messages)
        return response.content


class IterativeAnswerer:
    """
    Generates high-quality answers through iterative refinement.

    Process:
    1. Generate initial draft
    2. Evaluate quality (score 1-10)
    3. If score < target, refine and repeat
    4. Return final answer when target reached or max iterations hit
    """

    INITIAL_PROMPT = """당신은 기업 IR 자료를 분석하는 **시니어 투자 리서치 애널리스트**입니다.
CEO와 투자사에게 보고할 **정교한 분석 보고서**를 작성합니다.

## 보고서 작성 규칙

### 필수 형식
- **마크다운 형식**으로 작성
- 각 섹션은 명확하게 구분
- **레퍼런스는 본문 끝이 아니라, 각 핵심 내용 바로 아래에 하이퍼링크로 표시**
- 하이퍼링크 형식: `[출처명](URL)`
- 데이터가 없으면 **"없음"**으로 명시

### 보고서 구조 (반드시 이 순서대로)

## 1. 사업의 일반현황
- **주요기업정보 본사위치**: 구체적 주소
- **CEO**: 이름과 임기
- **임직원수**: 전체 인원 및 (R&D인력) 별도 표시
- **최신기준 3년간 매출액**: 연도별 매출 (3년이 안되면 있는 것만)
- **주요제품군**: 제품/서비스 목록
- **보유역량**: 핵심 기술/능력
- **주요고객사**: 상세하게 (가능하면 계약 규모 포함)

## 2. 주요연혁
- **시간순서대로** 작성 (최신순 또는 과거순 일관성 유지)
- 인증과 기체개발 투자 관점에서 중요 사건 강조
- 각 연혁 항목에 연도 명시

## 3. 주요투자 유치이력
- **레퍼런스와 숫자 정합성이 중요**
- 각 투자 라운드별로:
  - 투자 시기
  - 투자 금액
  - 리드 투자자
  - 참여 투자자
  - 기업 가치(Valuation)
  - 레퍼런스 링크

## 4. 주요인사
각 인사별로:
- **이름**
- **직위**
- **대학** (학위 포함)
- **주요경력** (이전 주요 직책)

## 5. 비즈니스모델
- **맥킨지 수준의 조사와 인사이트 필요**
- 수익 구조
- 가치 제안
- 시장 포지셔닝
- 경쟁 우위
- 성장 전략

## 6. 인증/기체개발현황
- **용어가 어려우니 용어에 대한 설명도 상세하게**
- 각 인증/개발 항목에:
  - 용어 설명
  - 현재 상태
  - 일정/마일스톤
  - 중요도

## 7. 주요 재무현황
- **숫자와 레퍼런스가 중요**
- **주요 매출액**: 연도별, 분기별 (가능하면)
- **주요 소비비용**: 주요 비용 항목
- **Going Concern** 또는 재무적으로 CEO/투자사가 꼭 알아야할 내용:
  - 현금 흐름
  - 자본 구조
  - 주요 재무 리스크
  - 재무 전망

## 레퍼런스 표기 규칙
- 각 섹션의 핵심 내용 **바로 아래**에 하이퍼링크로 표시
- 형식: `[자료명](URL)` 또는 `[출처: 자료명](URL)`
- 마지막에 한꺼번에 모으지 말고, 관련 내용 근처에 배치

## 품질 기준
- 모든 수치에 출처 필수
- 구체적인 숫자와 날짜 포함
- 데이터 없으면 "없음" 명시
- 전문적이면서도 이해하기 쉬운 문체"""

    def __init__(
        self,
        llm: BaseChatModel,
        target_score: int = 8,
        max_iterations: int = 2
    ):
        self.llm = llm
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.evaluator = QualityEvaluator(llm)
        self.refiner = AnswerRefiner(llm)

    def generate(
        self,
        question: str,
        context: str,
        progress_callback=None
    ) -> Tuple[str, int, int]:
        """
        Generate high-quality answer through iterative refinement.

        Returns:
            Tuple of (final_answer, final_score, iterations_used)
        """
        # Step 1: Generate initial draft
        if progress_callback:
            progress_callback("Generating initial analysis...")

        messages = [
            SystemMessage(content=self.INITIAL_PROMPT),
            HumanMessage(content=f"""## IR 자료 컨텍스트

{context}

---

## 분석 요청

{question}

---

위 문서를 기반으로 **상세한 분석 보고서**를 작성하세요.
구체적인 수치, 직접 인용, 명확한 출처를 포함하세요.""")
        ]

        try:
            response = self.llm.invoke(messages)
            current_answer = response.content
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            # Return a basic answer if LLM fails
            return f"[Error generating answer: {str(e)[:100]}]", 0, 0

        # Debug: Check initial answer
        logger.info(f"Initial answer length: {len(current_answer)} chars")
        if len(current_answer) < 100:
            logger.warning(f"Initial answer seems too short: {current_answer[:200]}")

        # Step 2: Iterative refinement
        for iteration in range(self.max_iterations):
            if progress_callback:
                progress_callback(f"Evaluating quality (iteration {iteration + 1})...")

            # Evaluate
            score, evaluation, improvements = self.evaluator.evaluate(
                question, current_answer
            )

            if progress_callback:
                progress_callback(f"Current score: {score}/10")

            # Check if target reached
            if score >= self.target_score:
                return current_answer, score, iteration + 1

            # Refine if not at target
            if improvements and iteration < self.max_iterations - 1:
                if progress_callback:
                    progress_callback(f"Refining answer based on feedback...")

                current_answer = self.refiner.refine(
                    question=question,
                    current_answer=current_answer,
                    context=context,
                    improvements=improvements
                )

        # Final evaluation
        final_score, _, _ = self.evaluator.evaluate(question, current_answer)

        return current_answer, final_score, self.max_iterations
