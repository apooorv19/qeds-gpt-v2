"""Version routes."""

from fastapi import APIRouter

from api.models import VersionResponse

router = APIRouter(prefix="/version", tags=["version"])


@router.get("", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Return application and API version metadata."""

    return VersionResponse(
        app_name="QEDS-GPT",
        version="1.0.0",
        api_version="v1",
    )
