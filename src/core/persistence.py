import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from api.schemas import JobResponse, JobStatus

class JobStore(ABC):
    @abstractmethod
    def create_job(self, job_id: str, initial_data: Dict[str, Any]):
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def update_job(self, job_id: str, updates: Dict[str, Any]):
        pass
    
    @abstractmethod
    def list_active_jobs(self) -> int:
        pass


class SQLiteJobStore(JobStore):
    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        # Create table if not exists
        # We assume single process for now, or safe separate connections
        conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result JSON,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def create_job(self, job_id: str, initial_data: Dict[str, Any]):
        conn = self._get_conn()
        cursor = conn.cursor()
        # Initial data usually has "status"
        status = initial_data.get("status", "pending")
        result = json.dumps(initial_data.get("result")) if initial_data.get("result") else None
        
        cursor.execute(
            "INSERT INTO jobs (job_id, status, result, error) VALUES (?, ?, ?, ?)",
            (job_id, status, result, initial_data.get("error"))
        )
        conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        data = dict(row)
        if data["result"]:
            try:
                data["result"] = json.loads(data["result"])
            except:
                data["result"] = None
        return data

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        fields = []
        values = []
        
        for k, v in updates.items():
            if k == "result":
                v = json.dumps(v)
            fields.append(f"{k} = ?")
            values.append(v)
            
        values.append(job_id)
        
        sql = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
        cursor.execute(sql, values)
        conn.commit()

    def list_active_jobs(self) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'processing')")
        return cursor.fetchone()[0]
