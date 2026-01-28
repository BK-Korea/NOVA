"""Company matching using LangChain and LLM."""
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Company information."""
    name: str
    official_name: str
    website: str
    ir_website: Optional[str] = None  # Dedicated IR/Investor Relations website
    ticker: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None


class CompanyMatcher:
    """
    Match user query to official company information using LLM.

    Uses LangChain to intelligently match company names/queries
    to official company names and websites.
    """

    MATCH_PROMPT = """당신은 기업 정보 전문가입니다. 사용자의 입력을 분석하여 정확한 기업 정보를 찾아야 합니다.

사용자 입력: "{query}"

## 작업

1. **기업 식별**: 입력에서 언급된 기업을 식별
2. **공식 이름 확인**: 해당 기업의 공식 정식 명칭 확인
3. **웹사이트 찾기**: 기업의 **실제 존재하는 공식 웹사이트 URL** 확인
   - **중요**: 반드시 실제로 존재하고 접근 가능한 웹사이트만 제공하세요
   - 잘못된 도메인이나 존재하지 않는 사이트를 만들지 마세요
   - 일반적인 패턴: company.com, company.co.kr, company.io, company.team 등
   - 예시: "beta" → "BETA Technologies" → https://beta.team (실제 존재)
4. **IR 웹사이트 찾기**: 투자자관계(IR) 전용 웹사이트 확인 (예: ir.company.com, investors.company.com)
5. **티커 심볼**: 상장사인 경우 주식 티커 심볼 확인

## 주의사항
- 약어/별칭을 공식 정식 명칭으로 변환 (예: "애플" → "Apple Inc.")
- 다국어 기업명을 영어 공식 명칭으로 변환
- 가장 최신의 기업명 사용 (M&A, 이름 변경 등 반영)
- 티커는 주요 거래소의 것 사용 (NYSE, NASDAQ, KOSPI 등)
- IR 웹사이트가 별도로 있으면 반드시 포함 (ir.company.com, investors.company.com 등)
- **웹사이트 URL은 반드시 실제로 존재하는 공식 사이트만 제공하세요**
- **추측하거나 존재하지 않는 도메인을 만들지 마세요**
- **일반적인 패턴**: company.com, company.co.kr, company.io, company.team, company.aero 등
- **예시**: "beta" → "BETA Technologies" → https://beta.team (실제 존재하는 사이트)
- **절대 추측하거나 존재하지 않는 도메인을 만들지 마세요**
- **실제로 알려진 공식 웹사이트만 제공하세요**

## 출력 형식 (JSON)

```json
{{
  "name": "사용자 입력 기업명",
  "official_name": "공식 정식 기업명",
  "website": "https://www.official-website.com",
  "ir_website": "https://ir.official-website.com 또는 null",
  "ticker": "티커 또는 null",
  "country": "본사 소재 국가",
  "description": "간단한 기업 설명 (1-2문장)"
}}
```

위 JSON 형식으로만 답변하세요. 다른 설명 없이 JSON만 출력."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def match(self, query: str) -> Dict[str, Any]:
        """
        Match user query to company information.

        Args:
            query: User's company search query

        Returns:
            Dict with 'company_info' (CompanyInfo) or 'error' (str)
        """
        try:
            messages = [
                SystemMessage(content=self.MATCH_PROMPT),
                HumanMessage(content=f'사용자 입력: "{query}"')
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Try to extract JSON if there's extra text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            company_info = CompanyInfo(
                name=query,
                official_name=data.get("official_name", query),
                website=data.get("website", ""),
                ir_website=data.get("ir_website"),
                ticker=data.get("ticker"),
                country=data.get("country"),
                description=data.get("description")
            )

            # Validate required fields
            if not company_info.website:
                return {
                    "error": "Could not determine official website for this company."
                }
            
            # Validate website format
            if not company_info.website.startswith(("http://", "https://")):
                company_info.website = f"https://{company_info.website}"
            
            # Basic validation: check if website looks reasonable
            # Don't accept obviously wrong domains
            website_lower = company_info.website.lower()
            if any(suspicious in website_lower for suspicious in ["example.com", "placeholder", "unknown", "n/a", "none", "test.com"]):
                return {
                    "error": f"Invalid website format detected: {company_info.website}. Please provide a valid official website."
                }

            return {"company_info": company_info}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response content: {content}")
            return {"error": f"Failed to parse company information: {e}"}

        except Exception as e:
            logger.error(f"Error matching company: {e}", exc_info=True)
            # Provide more user-friendly error messages
            error_msg = str(e)
            if "Authentication failed" in error_msg:
                return {"error": "GLM API authentication failed. Please check your GLM_API_KEY in .env file."}
            elif "Rate limit" in error_msg or "429" in error_msg or "rate limit" in error_msg.lower():
                return {
                    "error": "API rate limit exceeded (동시 사용 수가 한도를 초과했습니다).\n"
                             "잠시 기다린 후 다시 시도해주세요. (보통 1-2분 후 재시도 가능)\n"
                             "또는 Zhipu AI 고객 서비스에 연락하여 한도를 늘려주세요."
                }
            elif "HTTP" in error_msg or "Status" in error_msg:
                return {"error": f"API request failed: {error_msg}. Please check your API key and network connection."}
            else:
                return {"error": f"Failed to match company: {error_msg}"}
