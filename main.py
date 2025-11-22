# FastAPI main.py 예시
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import psycopg2
# 최신 SDK 사용법: google.generativeai 대신 from google import genai
from google import genai 
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (로컬 개발용. Render에서는 무시됨)
load_dotenv()

# --- 1. 환경 변수 로드 ---
# Render에서 설정한 환경 변수 이름을 사용합니다.
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. FastAPI 앱 초기화 ---
app = FastAPI()

# --- 3. Gemini 클라이언트 초기화 ---
# 모델 객체를 가져오는 과정(오류 유발)을 생략하고, client 객체만 초기화합니다.
client = None 

if GEMINI_API_KEY:
    try:
        # 클라이언트 객체를 생성하면서 API 키를 전달합니다.
        client = genai.Client(api_key=GEMINI_API_KEY) 
        
        # NOTE: model = client.models.get(...) 또는 client.get_model(...) 코드를 제거했습니다.
        # 이 코드가 버전 호환성 오류를 일으키므로, client만 초기화하고 API 호출 시 모델 이름을 직접 전달합니다.

        print("Gemini Client Initialized Successfully.")
    except Exception as e:
        # 초기화 오류 발생 시 로그에 기록합니다.
        print(f"Gemini Client Initialization Error: {e}")
        client = None # 오류 발생 시 client를 None으로 설정하여 AI 서비스 접근을 막음
else:
    print("FATAL: GEMINI_API_KEY environment variable is not set.")

# --- 4. DB 연결 함수 ---
def get_db_connection():
    # DATABASE_URL이 설정되지 않았다면 연결 시도하지 않음
    if not DATABASE_URL:
        print("FATAL: DATABASE_URL environment variable is not set.")
        return None
        
    try:
        # DATABASE_URL 문자열 전체를 사용하여 연결 시도
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        # DB 연결 실패 시 에러를 기록합니다.
        print(f"Database Connection Error: {e}")
        return None

# --- 5. 요청/응답 모델 정의 ---
class PromptRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: str

# --- 6. API 엔드포인트 정의 (Node.js 서버가 호출할 주소) ---
@app.post("/generate_ai_response")
def generate_ai_response(request: PromptRequest):
    # AI client가 초기화되지 않았을 경우 (API 키 누락 등)
    if not client:
        raise HTTPException(status_code=503, detail="AI Service not available (Check GEMINI_API_KEY).")

    conn = get_db_connection()
    # DB 연결에 실패했을 경우 (DATABASE_URL 누락 또는 오류)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error (Check DATABASE_URL).")

    try:
        # 1. DB에서 이전 대화 기록 불러오기 (선택 사항: 맥락 유지)
        # 여기에 DB 로직 추가
        
        # 2. Gemini API 호출
        full_prompt = "You are a helpful assistant. " + request.prompt
        
        # client 객체의 models를 통해 generate_content를 호출하고, 모델 이름을 인수로 전달합니다.
        response = client.models.generate_content( 
            model="gemini-1.5-flash", 
            contents=full_prompt
        )
        ai_response_text = response.text

        # 3. DB에 대화 기록 저장 (사용자 입력 + AI 응답)
        # 여기에 DB 저장 로직 추가

        # 4. Node.js 서버로 최종 응답 반환
        return {"response": ai_response_text}

    except Exception as e:
        # AI 생성 또는 DB 작업 실패 시
        print(f"AI Generation or DB operation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during AI processing.")
    finally:
        # 작업이 끝나면 DB 연결을 안전하게 닫습니다.
        if conn:
            conn.close()