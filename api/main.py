import os
import re
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rapidfuzz import fuzz


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("uk-agent-api")


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@uk_agent_db:5432/uk_agent",
)

ADDRESS_CACHE_LIMIT = 20000
DEFAULT_MIN_SCORE = 72
MAX_ADDRESS_CANDIDATES = 5


class AppState:
    pool: asyncpg.Pool | None = None
    address_cache: list[dict[str, Any]] = []


state = AppState()


DROP_WORDS = {
    "город", "г", "г.",
    "улица", "ул", "ул.",
    "проспект", "пр", "пр.", "пр-кт", "пр-т",
    "переулок", "пер", "пер.",
    "дом", "д", "д.",
    "квартира", "кв", "кв.",
    "апартамент", "апартаменты",
    "корпус", "корп", "корп.",
    "строение", "стр", "стр.",
    "шоссе", "ш", "ш.",
    "бульвар", "бул", "бул.",
    "площадь", "пл", "пл.",
    "проезд",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9/.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(value: Any) -> str:
    text = normalize_text(value)
    tokens = [token for token in text.split() if token not in DROP_WORDS]
    return " ".join(tokens)


def make_address_line(row: dict[str, Any]) -> str:
    if row.get("full_address"):
        return str(row["full_address"])

    parts = []

    if row.get("city"):
        parts.append(str(row["city"]))

    if row.get("street"):
        parts.append(str(row["street"]))

    if row.get("house"):
        parts.append(f"д. {row['house']}")

    if row.get("apartment"):
        parts.append(f"кв. {row['apartment']}")

    return ", ".join(parts)


def extract_address_hints(raw_text: str) -> dict[str, Optional[str]]:
    text = normalize_text(raw_text)

    house = None
    apartment = None

    house_match = re.search(
        r"(?:дом|д\.?)\s*([0-9]+[а-яa-z]?(?:/[0-9]+[а-яa-z]?)?)",
        text,
    )
    if house_match:
        house = house_match.group(1)

    apartment_match = re.search(
        r"(?:квартира|кв\.?|апартамент|апартаменты)\s*([0-9]+[а-яa-z]?)",
        text,
    )
    if apartment_match:
        apartment = apartment_match.group(1)

    numbers = re.findall(
        r"\b[0-9]+[а-яa-z]?(?:/[0-9]+[а-яa-z]?)?\b",
        text,
    )

    if not house and numbers:
        house = numbers[0]

    if not apartment and len(numbers) >= 2:
        apartment = numbers[1]

    return {
        "house": house,
        "apartment": apartment,
    }


def score_address(
    query_norm: str,
    row: dict[str, Any],
    hints: dict[str, Optional[str]],
) -> int:
    target = row["_search_text"]

    score = max(
        fuzz.WRatio(query_norm, target),
        fuzz.token_set_ratio(query_norm, target),
        fuzz.partial_ratio(query_norm, target),
    )

    house_hint = normalize_text(hints.get("house"))
    apartment_hint = normalize_text(hints.get("apartment"))

    row_house = normalize_text(row.get("house"))
    row_apartment = normalize_text(row.get("apartment"))

    if house_hint and row_house:
        if house_hint == row_house:
            score += 8
        else:
            score -= 8

    if apartment_hint and row_apartment:
        if apartment_hint == row_apartment:
            score += 5
        else:
            score -= 4

    return max(0, min(100, int(score)))


async def load_address_cache() -> int:
    if state.pool is None:
        raise RuntimeError("Database pool is not initialized")

    async with state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT address_id, city, street, house, apartment, full_address
            FROM addresses
            LIMIT $1
            """,
            ADDRESS_CACHE_LIMIT,
        )

    cache = []

    for row in rows:
        item = dict(row)
        item["_address_line"] = make_address_line(item)
        item["_search_text"] = normalize_address(item["_address_line"])
        cache.append(item)

    state.address_cache = cache
    logger.info("Loaded %s addresses into cache", len(cache))

    return len(cache)


async def get_address_or_404(address_id: str) -> dict[str, Any]:
    if state.pool is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")

    async with state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT address_id, city, street, house, apartment, full_address
            FROM addresses
            WHERE address_id = $1
            LIMIT 1
            """,
            address_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail="Address not found")

    return dict(row)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
        command_timeout=10,
    )

    await load_address_cache()

    yield

    if state.pool is not None:
        await state.pool.close()


app = FastAPI(
    title="UK Agent API",
    version="1.0.0",
    description="FastAPI tools for Dialog AI agent.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResolveAddressRequest(BaseModel):
    raw_text: str = Field(..., description="Адрес в свободной форме от пользователя")
    min_score: int = Field(DEFAULT_MIN_SCORE, ge=0, le=100)
    limit: int = Field(MAX_ADDRESS_CANDIDATES, ge=1, le=20)


class AddressCandidate(BaseModel):
    address_id: str
    score: int
    address: str
    city: Optional[str] = None
    street: Optional[str] = None
    house: Optional[str] = None
    apartment: Optional[str] = None


class ResolveAddressResponse(BaseModel):
    address_match_status: str
    address_id: Optional[str] = None
    normalized_address: Optional[str] = None
    address_candidates: str = ""
    api_error: bool = False
    api_error_text: str = ""


class CustomerContextRequest(BaseModel):
    address_id: str
    include_debts: bool = True
    include_tickets: bool = True
    include_notifications: bool = True
    limit: int = Field(5, ge=1, le=20)


class TicketStatusRequest(BaseModel):
    address_id: str
    limit: int = Field(5, ge=1, le=20)


class CreateTicketRequest(BaseModel):
    address_id: str
    reason: str
    current_comment: Optional[str] = None
    status: str = "новая"
    estimated_finish_at: Optional[str] = None


def to_candidate(row: dict[str, Any], score: int) -> AddressCandidate:
    return AddressCandidate(
        address_id=str(row["address_id"]),
        score=score,
        address=row["_address_line"],
        city=row.get("city"),
        street=row.get("street"),
        house=row.get("house"),
        apartment=row.get("apartment"),
    )


def candidates_to_text(candidates: list[AddressCandidate]) -> str:
    return "\n".join(
        [
            (
                f"{index + 1}. {candidate.address} "
                f"| address_id={candidate.address_id} "
                f"| score={candidate.score}"
            )
            for index, candidate in enumerate(candidates)
        ]
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    db_ok = False
    error = None

    try:
        if state.pool is not None:
            async with state.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
    except Exception as exc:
        error = str(exc)

    return {
        "ok": db_ok,
        "service": "uk-agent-api",
        "db_ok": db_ok,
        "address_cache_size": len(state.address_cache),
        "error": error,
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/tools")
async def tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "resolve_address",
                "method": "POST",
                "path": "/tools/resolve_address",
                "description": "Найти address_id по сырому адресу пользователя.",
                "response_fields": {
                    "status": "address_match_status",
                    "address_id": "address_id",
                    "normalized_address": "normalized_address",
                    "candidates_text": "address_candidates",
                    "api_error": "api_error",
                    "api_error_text": "api_error_text",
                },
            },
            {
                "name": "get_customer_context",
                "method": "POST",
                "path": "/tools/get_customer_context",
                "description": "Получить долги, заявки и уведомления по address_id.",
            },
            {
                "name": "get_ticket_status",
                "method": "POST",
                "path": "/tools/get_ticket_status",
                "description": "Получить последние заявки по address_id.",
            },
            {
                "name": "create_ticket",
                "method": "POST",
                "path": "/tools/create_ticket",
                "description": "Создать заявку по подтверждённому address_id.",
            },
        ]
    }


@app.post("/tools/resolve_address", response_model=ResolveAddressResponse)
async def resolve_address(payload: ResolveAddressRequest) -> ResolveAddressResponse:
    try:
        query_norm = normalize_address(payload.raw_text)

        if not query_norm:
            return ResolveAddressResponse(
                address_match_status="not_found",
                address_id=None,
                normalized_address=None,
                address_candidates="",
                api_error=False,
                api_error_text="",
            )

        hints = extract_address_hints(payload.raw_text)

        scored: list[tuple[int, dict[str, Any]]] = []

        for row in state.address_cache:
            score = score_address(query_norm, row, hints)

            if score >= payload.min_score:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)

        candidates = [
            to_candidate(row, score)
            for score, row in scored[:payload.limit]
        ]

        if not candidates:
            return ResolveAddressResponse(
                address_match_status="not_found",
                address_id=None,
                normalized_address=None,
                address_candidates="",
                api_error=False,
                api_error_text="",
            )

        best = candidates[0]
        resolved = False

        if len(candidates) == 1 and best.score >= 84:
            resolved = True

        elif len(candidates) >= 2:
            score_gap = candidates[0].score - candidates[1].score

            if best.score >= 88 and score_gap >= 8:
                resolved = True

        address_candidates = candidates_to_text(candidates)

        if resolved:
            return ResolveAddressResponse(
                address_match_status="found",
                address_id=best.address_id,
                normalized_address=best.address,
                address_candidates=address_candidates,
                api_error=False,
                api_error_text="",
            )

        return ResolveAddressResponse(
            address_match_status="multiple",
            address_id=None,
            normalized_address=None,
            address_candidates=address_candidates,
            api_error=False,
            api_error_text="",
        )

    except Exception as exc:
        logger.exception("resolve_address failed")

        return ResolveAddressResponse(
            address_match_status="error",
            address_id=None,
            normalized_address=None,
            address_candidates="",
            api_error=True,
            api_error_text=str(exc),
        )


@app.post("/tools/get_customer_context")
async def get_customer_context(payload: CustomerContextRequest) -> dict[str, Any]:
    if state.pool is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")

    address = await get_address_or_404(payload.address_id)

    result: dict[str, Any] = {
        "ok": True,
        "api_error": False,
        "api_error_text": "",
        "address_id": payload.address_id,
        "address": make_address_line(address),
        "address_data": address,
        "debts": [],
        "tickets": [],
        "notifications": [],
    }

    try:
        async with state.pool.acquire() as conn:
            if payload.include_debts:
                debts = await conn.fetch(
                    """
                    SELECT address_id, amount_text, debt_months
                    FROM debts
                    WHERE address_id = $1
                    LIMIT $2
                    """,
                    payload.address_id,
                    payload.limit,
                )

                result["debts"] = [dict(row) for row in debts]

            if payload.include_tickets:
                tickets = await conn.fetch(
                    """
                    SELECT created_at, address_id, reason, current_comment, status, estimated_finish_at
                    FROM tickets
                    WHERE address_id = $1
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    payload.address_id,
                    payload.limit,
                )

                result["tickets"] = [dict(row) for row in tickets]

            if payload.include_notifications:
                notifications = await conn.fetch(
                    """
                    SELECT notification_type, address_id, comment, starts_at, ends_at
                    FROM notifications
                    WHERE address_id = $1
                    ORDER BY starts_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    payload.address_id,
                    payload.limit,
                )

                result["notifications"] = [dict(row) for row in notifications]

        return result

    except Exception as exc:
        logger.exception("get_customer_context failed")

        return {
            "ok": False,
            "api_error": True,
            "api_error_text": str(exc),
            "address_id": payload.address_id,
            "address": "",
            "address_data": {},
            "debts": [],
            "tickets": [],
            "notifications": [],
        }


@app.post("/tools/get_ticket_status")
async def get_ticket_status(payload: TicketStatusRequest) -> dict[str, Any]:
    if state.pool is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")

    try:
        await get_address_or_404(payload.address_id)

        async with state.pool.acquire() as conn:
            tickets = await conn.fetch(
                """
                SELECT created_at, address_id, reason, current_comment, status, estimated_finish_at
                FROM tickets
                WHERE address_id = $1
                ORDER BY created_at DESC NULLS LAST
                LIMIT $2
                """,
                payload.address_id,
                payload.limit,
            )

        rows = [dict(row) for row in tickets]

        return {
            "ok": True,
            "found": bool(rows),
            "api_error": False,
            "api_error_text": "",
            "address_id": payload.address_id,
            "tickets": rows,
            "tickets_text": format_tickets_text(rows),
            "message": "Заявки найдены." if rows else "По этому адресу заявок не найдено.",
        }

    except Exception as exc:
        logger.exception("get_ticket_status failed")

        return {
            "ok": False,
            "found": False,
            "api_error": True,
            "api_error_text": str(exc),
            "address_id": payload.address_id,
            "tickets": [],
            "tickets_text": "",
            "message": "Ошибка при получении статуса заявок.",
        }


@app.post("/tools/create_ticket")
async def create_ticket(payload: CreateTicketRequest) -> dict[str, Any]:
    if state.pool is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")

    try:
        await get_address_or_404(payload.address_id)

        created_at = datetime.utcnow().isoformat() + "Z"

        async with state.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tickets (
                    created_at,
                    address_id,
                    reason,
                    current_comment,
                    status,
                    estimated_finish_at
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING created_at, address_id, reason, current_comment, status, estimated_finish_at
                """,
                created_at,
                payload.address_id,
                payload.reason,
                payload.current_comment,
                payload.status,
                payload.estimated_finish_at,
            )

        ticket = dict(row)

        return {
            "ok": True,
            "created": True,
            "api_error": False,
            "api_error_text": "",
            "ticket": ticket,
            "ticket_text": format_tickets_text([ticket]),
            "message": "Заявка создана.",
        }

    except Exception as exc:
        logger.exception("create_ticket failed")

        return {
            "ok": False,
            "created": False,
            "api_error": True,
            "api_error_text": str(exc),
            "ticket": None,
            "ticket_text": "",
            "message": "Ошибка при создании заявки.",
        }


def format_tickets_text(tickets: list[dict[str, Any]]) -> str:
    if not tickets:
        return ""

    lines = []

    for index, ticket in enumerate(tickets, start=1):
        created_at = ticket.get("created_at") or "дата не указана"
        reason = ticket.get("reason") or "причина не указана"
        status = ticket.get("status") or "статус не указан"
        comment = ticket.get("current_comment") or ""
        estimated_finish_at = ticket.get("estimated_finish_at") or ""

        line = f"{index}. {created_at}: {reason}. Статус: {status}."

        if comment:
            line += f" Комментарий: {comment}."

        if estimated_finish_at:
            line += f" Ожидаемое завершение: {estimated_finish_at}."

        lines.append(line)

    return "\n".join(lines)


@app.post("/admin/reload-address-cache")
async def reload_address_cache() -> dict[str, Any]:
    try:
        count = await load_address_cache()

        return {
            "ok": True,
            "api_error": False,
            "api_error_text": "",
            "address_cache_size": count,
        }

    except Exception as exc:
        logger.exception("reload_address_cache failed")

        return {
            "ok": False,
            "api_error": True,
            "api_error_text": str(exc),
            "address_cache_size": len(state.address_cache),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
