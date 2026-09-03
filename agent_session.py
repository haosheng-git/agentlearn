import json
import os
from dataclasses import asdict
from pathlib import Path

from agent_messages import Message, ToolCall


class SessionStoreError(RuntimeError):
    """Session 加载或保存失败。"""

def truncate_messages(
        messages: list[Message],
        max_messages: int,
) -> list[Message]:
    """删除最旧完整轮次，不从一轮对话中间切开。"""
    if(
        not isinstance(max_messages,int)
        or isinstance(max_messages,bool)
        or max_messages < 1
    ):
        raise ValueError(
            "max_messages 必须是大于 0 的整数"
        )

    copied_messages = list(messages)

    if len(copied_messages) <= max_messages:
        return copied_messages

    turns: list[list[Message]] = []
    current_turn: list[Message] = []

    for message in copied_messages:
        if message.role == "user" and current_turn:
            turns.append(current_turn)
            current_turn = []

        current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    kept_messages: list[Message] = []

    for turn in reversed(turns):
      would_exceed_limit = (
          kept_messages
          and len(turn) + len(kept_messages) > max_messages
      )  

      if would_exceed_limit:
          break

      kept_messages = list(turn) + kept_messages

      if len(kept_messages) >= max_messages:
          break

    return kept_messages

class JsonSessionStore:
    def __init__(self,path: str | Path):
        self.path = Path(path) 

    @staticmethod
    def _message_to_dict(message: Message) -> dict:
        return asdict(message)

    @staticmethod
    def _message_from_dict(data: dict) -> Message:
        return Message(
            role=data["role"],
            content=data["content"],
            tool_call_id=data.get("tool_call_id"),
            tool_calls=[
                ToolCall(
                    id=item["id"],
                    name=item["name"],
                    arguments=item["arguments"],
                )
                for item in data.get("tool_calls", [])
            ],
        )

    def load(self,session_id: str) -> list[Message]:
        if not self.path.exists():
            return []

        latest_messages: list[Message] = []

        try:
            with self.path.open("r",encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    record = json.loads(line)

                    if record.get("session_id") != session_id:
                        continue

                    latest_messages = [
                        self._message_from_dict(item)
                        for item in record["messages"]
                    ]

        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise SessionStoreError(
                "无法加载 session"
            ) from error

        return list(latest_messages)

    def save(
            self,
            session_id: str,
            messages: list[Message],
    ) -> None:
        record = {
            "session_id": session_id,
            "messages": [
                self._message_to_dict(message)
                for message in messages
            ],
        }

        try:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                file.flush()
                os.fsync(file.fileno())
        except (OSError,TypeError) as error:
            raise SessionStoreError(
                "无法保存 session"
            ) from error

    def clear(self, session_id: str) -> None:
        self.save(session_id, [])

    def export(self,session_id: str) -> list[dict]:
        return [
            self._message_to_dict(message)
            for message in self.load(session_id)
        ]

