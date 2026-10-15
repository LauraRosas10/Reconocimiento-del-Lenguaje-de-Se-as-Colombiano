import re
import unicodedata


def normalize_label(label: str) -> str:
    decomposed = unicodedata.normalize("NFKD", label.casefold())
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", "", without_marks)


DIRECT_VIDEO_LABELS = {
    "bien": "0024",
    "bienvenido": "0034",
    "buenas noches": "0003",
    "buenas tardes": "0002",
    "buenos dias": "0001",
    "con gusto": "0033",
    "gracias": "0000",
    "hola": "0030",
    "mal": "0025",
    "maso menos": "0027",
    "perdon": "0035",
    "permiso": "0036",
    "por favor": "0032",
}

SEMANTIC_VIDEO_LABELS = {
    "como esta": "0050",
    "chao": "0051",
    "denada": "0052",
    "lo siento": "0053",
}

VIDEO_LABEL_TO_CLASS = {
    normalize_label(label): class_name
    for label, class_name in (DIRECT_VIDEO_LABELS | SEMANTIC_VIDEO_LABELS).items()
}
DIRECT_CLASSES = tuple(sorted(set(DIRECT_VIDEO_LABELS.values())))
SEMANTIC_CLASSES = tuple(sorted(set(SEMANTIC_VIDEO_LABELS.values())))
COMBINED_CLASSES = tuple(sorted((*DIRECT_CLASSES, *SEMANTIC_CLASSES)))


def map_video_label(label: str) -> str | None:
    return VIDEO_LABEL_TO_CLASS.get(normalize_label(label))
