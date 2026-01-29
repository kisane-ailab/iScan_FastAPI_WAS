"""
데이터베이스 연결 모듈
MySQL 비동기 연결 풀 관리
"""
import aiomysql
from typing import Optional, Dict, List, Any
from app.core.config import get_database_config
import logging

logger = logging.getLogger(__name__)

# 전역 연결 풀
_pool: Optional[aiomysql.Pool] = None


async def init_db_pool():
    """데이터베이스 연결 풀 초기화"""
    global _pool
    
    if _pool is not None:
        return
    
    try:
        db_config = get_database_config()
        
        _pool = await aiomysql.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            db=db_config["database"],
            charset=db_config["charset"],
            minsize=1,
            maxsize=db_config["pool_size"],
            autocommit=True
        )
        
        logger.info(f"✅ 데이터베이스 연결 풀 초기화 완료: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        print(f"✅ 데이터베이스 연결 풀 초기화 완료: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 풀 초기화 실패: {e}")
        print(f"❌ 데이터베이스 연결 풀 초기화 실패: {e}")
        raise


async def close_db_pool():
    """데이터베이스 연결 풀 종료"""
    global _pool
    
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
        logger.info("데이터베이스 연결 풀 종료 완료")
        print("데이터베이스 연결 풀 종료 완료")


async def get_db_connection():
    """데이터베이스 연결 가져오기"""
    global _pool
    
    if _pool is None:
        await init_db_pool()
    
    return await _pool.acquire()


async def release_db_connection(conn):
    """데이터베이스 연결 반환"""
    global _pool
    
    if _pool is not None:
        _pool.release(conn)


async def execute_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    """쿼리 실행 및 결과 반환"""
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchall()
            return result
    except Exception as e:
        logger.error(f"❌ 쿼리 실행 실패: {query[:100]}... 에러: {e}")
        print(f"❌ 쿼리 실행 실패: {e}")
        raise
    finally:
        if conn:
            await release_db_connection(conn)


async def execute_one(query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
    """단일 결과 쿼리 실행"""
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchone()
            return result
    except Exception as e:
        logger.error(f"❌ 쿼리 실행 실패: {query[:100]}... 에러: {e}")
        print(f"❌ 쿼리 실행 실패: {e}")
        raise
    finally:
        if conn:
            await release_db_connection(conn)


async def execute_update(query: str, params: tuple = None) -> int:
    """UPDATE/INSERT/DELETE 쿼리 실행 및 영향받은 행 수 반환"""
    conn = None
    try:
        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            affected_rows = await cursor.execute(query, params)
            await conn.commit()
            return affected_rows
    except Exception as e:
        logger.error(f"❌ 쿼리 실행 실패: {query[:100]}... 에러: {e}")
        print(f"❌ 쿼리 실행 실패: {e}")
        if conn:
            await conn.rollback()
        raise
    finally:
        if conn:
            await release_db_connection(conn)

