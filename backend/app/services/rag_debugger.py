# app/services/rag_debugger.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from app.core.logger import log

@dataclass
class DebugEntry:
    stage: str
    data: dict

class RagDebugger:
    def __init__(self):
        self.entries: List[DebugEntry] = []

    def add(self, stage: str, **data):
        self.entries.append(DebugEntry(stage, data))

    def dump(self):
        log.info("🧠 ===== RAG DEBUG REPORT =====")

        for entry in self.entries:
            log.info(f"\n--- [{entry.stage}] ---")
            for k, v in entry.data.items():
                log.info(f"{k}: {v}")

        log.info("🧠 ===== END DEBUG REPORT =====")