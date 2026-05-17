from collections.abc import Awaitable, Callable
from secrets import compare_digest

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from .settings import Settings


PUBLIC_PATHS = {
    "/api/v1/health",
    "/api/v1/access",
    "/api/v1/auth/verify",
    "/api/v1/jobs/events",
}


def token_from_request(request: Request) -> str:
    header_token = request.headers.get("x-sceneworks-token", "").strip()
    if header_token:
        return header_token

    authorization = request.headers.get("authorization", "").strip()
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix) :].strip()

    return ""


def is_authorized(request: Request, settings: Settings) -> bool:
    if not settings.access_token:
        return True

    request_token = token_from_request(request).encode("utf-8")
    expected_token = settings.access_token.encode("utf-8")
    return compare_digest(request_token, expected_token)


async def access_control_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
    settings: Settings,
) -> Response:
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    if is_authorized(request, settings):
        return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={
            "detail": "SceneWorks access token required",
            "authRequired": True,
        },
    )
