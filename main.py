# FastAPI main.py 예시
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import psycopg2
import google.generativeai as genai # 최신 Gemini SDK로 변경
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (로컬 개발용)
load_dotenv()

# Node.js와 동일한 DB 접속 정보 환경 변수를 사용
DATABASE_URL = os.getenv("DATABASE_URL") # Render에서 설정한 환경 변수 이름으로 변경 권장

# --- 1. 환경 변수 로드 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. FastAPI 앱 초기화 ---
app = FastAPI()

# --- 3. Gemini 클라이언트 초기화 ---
model = None
if GEMINI_API_KEY:
    try:
        # 환경 변수를 사용하여 Gemini 클라이언트 초기화
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash') # 모델 이름은 필요에 따라 변경
        print("Gemini Client Initialized Successfully.")
    except Exception as e:
        print(f"Gemini Client Initialization Error: {e}")
else:
    print("FATAL: GEMINI_API_KEY environment variable is not set.")

# --- 4. DB 연결 함수 ---
def get_db_connection():
    try:
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

# --- 6. API 엔드포인트 정의 ---
@app.post("/generate_ai_response")
def generate_ai_response(request: PromptRequest):
    if not model:
        raise HTTPException(status_code=503, detail="AI Service not available.")

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error.")

    try:
        # 1. DB에서 이전 대화 기록 불러오기 (선택 사항: 맥락 유지)
        # 커서 설정 및 쿼리 실행...

        # 2. Gemini API 호출
        full_prompt = "You are a helpful assistant. " + request.prompt
        response = model.generate_content(full_prompt)
        ai_response_text = response.text

        # 3. DB에 대화 기록 저장 (사용자 입력 + AI 응답)
        # conn을 사용해서 INSERT 쿼리 실행...

        # 4. Node.js 서버로 응답 반환
        return {"response": ai_response_text}

    except Exception as e:
        print(f"AI Generation or DB operation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during AI processing.")
    finally:
        # try...except 블록이 끝나면 항상 DB 연결을 닫습니다.
        if conn:
            conn.close()