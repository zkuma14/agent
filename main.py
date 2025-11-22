# FastAPI main.py 예시
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import psycopg2
from google import genai  # 최신 SDK 사용법: google.genai 대신 from google import genai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (로컬 개발용. Render에서는 무시됨)
load_dotenv()

# --- 1. 환경 변수 로드 ---
# Render의 환경 변수 이름을 사용합니다.
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. FastAPI 앱 초기화 ---
app = FastAPI()

# --- 3. Gemini 클라이언트 초기화 ---
# 'genai.configure' 대신 'genai.Client'를 사용하여 오류를 해결합니다.
client = None  # 클라이언트 객체
model = None   # 모델 객체

if GEMINI_API_KEY:
    try:
        # 클라이언트 객체를 생성하면서 API 키를 전달합니다.
        client = genai.Client(api_key=GEMINI_API_KEY) 
        
        # 모델 객체를 client를 통해 가져옵니다.
        # 이 방식으로 'model' 변수에 모델 객체를 할당합니다.
        model = client.get_model("gemini-2.5-flash")

        print("Gemini Client Initialized Successfully.")
    except Exception as e:
        # 초기화 오류 발생 시 로그에 기록합니다.
        print(f"Gemini Client Initialization Error: {e}")
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
    # AI 서비스가 초기화되지 않았을 경우 (API 키 누락 등)
    if not model:
        raise HTTPException(status_code=503, detail="AI Service not available (Check GEMINI_API_KEY).")

    conn = get_db_connection()
    # DB 연결에 실패했을 경우 (DATABASE_URL 누락 또는 오류)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error (Check DATABASE_URL).")

    try:
        # 1. DB에서 이전 대화 기록 불러오기 (선택 사항)
        # 여기에 DB에서 과거 대화 기록을 불러오는 로직을 추가하여 'full_prompt'에 맥락을 제공할 수 있습니다.

        # 2. Gemini API 호출
        full_prompt = "You are a helpful assistant. " + request.prompt
        # model 객체의 generate_content 메서드를 사용하여 응답을 생성합니다.
        response = model.generate_content(full_prompt)
        ai_response_text = response.text

        # 3. DB에 대화 기록 저장 (선택 사항)
        # 여기에 사용자 입력과 AI 응답을 DB에 INSERT하는 로직을 추가합니다.

        # 4. Node.js 서버로 최종 응답 반환
        return {"response": ai_response_text}

    except Exception as e:
        print(f"AI Generation or DB operation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during AI processing.")
    finally:
        # 작업이 끝나면 DB 연결을 안전하게 닫습니다.
        if conn:
            conn.close()