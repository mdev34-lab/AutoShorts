from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VideoMetadata:
    title: str | None = None
    artist: str = "AutoShorts"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    genre: str = "Explainer"
    description: str | None = None
    comment: str | None = None
    encoder: str = "AutoShorts"

    def to_ffmpeg_args(self) -> list[str]:
        args = []
        for key, value in self.__dict__.items():
            if value is not None:
                args.extend(["-metadata", f"{key}={value}"])
        return args
