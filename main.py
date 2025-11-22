# FastAPI main.py 최종 예시

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from google import genai 
from google.genai import types
import os
import psycopg2
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (로컬 개발용)
# Render에 배포 시에는 Render 환경 변수 설정을 사용합니다.
load_dotenv()

# --- 1. 환경 변수 로드 ---
# Render에서 설정한 환경 변수 이름을 사용합니다.
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. FastAPI 앱 초기화 ---
app = FastAPI()

@app.get("/")
def read_root():
    """
    서버의 상태를 확인하기 위한 헬스 체크 엔드포인트입니다.
    """
    return {"status": "AI Agent Server is Running", "message": "Access /generate_ai_response for AI service."}

# --- 3. Gemini 클라이언트 초기화 ---
client = None 

if GEMINI_API_KEY:
    try:
        # 클라이언트 객체를 생성하면서 API 키를 전달합니다.
        client = genai.Client(api_key=GEMINI_API_KEY) 
        print("Gemini Client Initialized Successfully.")
    except Exception as e:
        print(f"Gemini Client Initialization Error: {e}")
        client = None
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
async def generate_ai_response(request: PromptRequest):
    # AI client가 초기화되지 않았을 경우 (API 키 누락 등)
    if not client:
        raise HTTPException(status_code=503, detail="AI Service not available (Check GEMINI_API_KEY).")

    conn = get_db_connection()
    # DB 연결에 실패했을 경우 (DATABASE_URL 누락 또는 오류)
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error (Check DATABASE_URL).")

    try:
        # --- 1. DB에서 이전 대화 기록 불러오기 (컨텍스트 유지) ---
        cur = conn.cursor()
        
        # 특정 user_id와 session_id에 해당하는 최근 대화 5개를 최신 순으로 가져옵니다.
        cur.execute(
            """
            SELECT prompt, response FROM conversations
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (request.user_id, request.session_id)
        )
        history_records = cur.fetchall()
        cur.close()

        # 대화 기록을 Gemini API의 'contents' 형식에 맞게 변환합니다.
        # 최신 대화가 뒤로 오도록 순서를 역전시킵니다.
        chat_history = []
        for prompt, response in reversed(history_records):
            # 사용자의 질문 (role: user)
            chat_history.append({"role": "user", "parts": [{"text": prompt}]})
            # AI의 답변 (role: model)
            chat_history.append({"role": "model", "parts": [{"text": response}]})
            
        # --- 2. Gemini API 호출 ---
        
        # System Instruction: 에이전트의 역할 정의
        system_instruction = "You are a helpful and friendly assistant. Use the provided chat history to answer the current prompt concisely."
        
        # history와 현재 프롬프트를 contents에 포함시킵니다.
        # contents는 [chat_history, current_prompt] 순서입니다.
        contents = chat_history + [{"role": "user", "parts": [{"text": request.prompt}]}]
        
        # GenerateContentConfig를 사용하여 system_instruction을 전달합니다.
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )

        # client.models.generate_content_async를 사용하여 비동기 호출을 권장합니다.
        # FastAPI 엔드포인트가 'async' 함수이므로 await를 사용합니다.
        response = await client.models.generate_content_async( 
            model="gemini-2.5-flash", 
            contents=contents, # 컨텍스트가 포함된 전체 대화 목록
            config=config      # 시스템 역할 설정
        )
        ai_response_text = response.text

        # --- 3. DB에 대화 기록 저장 (사용자 입력 + AI 응답) ---
        save_cur = conn.cursor()
        
        save_cur.execute(
            """
            INSERT INTO conversations (user_id, session_id, prompt, response)
            VALUES (%s, %s, %s, %s)
            """,
            (request.user_id, request.session_id, request.prompt, ai_response_text)
        )
        conn.commit() # 변경사항을 DB에 최종 반영
        save_cur.close()

        # --- 4. Node.js 서버로 최종 응답 반환 ---
        return {"response": ai_response_text}

    except Exception as e:
        # AI 생성 또는 DB 작업 실패 시
        print(f"AI Generation or DB operation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during AI processing.")
    finally:
        # 작업이 끝나면 DB 연결을 안전하게 닫습니다.
        if conn:
            conn.close()
