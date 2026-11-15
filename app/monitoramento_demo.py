"""Protótipo didático: CSV sintético e comparação histórica, sem uso clínico.

Só biblioteca padrão. Não lê arquivos, não grava dados e não faz chamadas de rede.
As faixas de entrada e os parâmetros são convenções de teste, não limites clínicos.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import math
import re
from statistics import median
from typing import MutableMapping, Optional


MAX_BYTES = 256 * 1024
MAX_ROWS = 2000
MAX_CELL = 64
HEADER = ("timestamp", "signal", "value", "unit")
WINDOW = 10
MIN_HISTORY = 5
REVIEW_SCORE = 3.5
MAX_GAP = timedelta(hours=6)
ALGORITHM_VERSION = "rolling-median-mad-demo-v1"


@dataclass(frozen=True)
class SignalSpec:
    label: str
    unit: str
    minimum: float
    maximum: float
    scale_floor: float


# Sanidade numérica do protótipo, NÃO intervalos de normalidade.
SIGNALS = {
    "heart_rate": SignalSpec("Frequência cardíaca", "bpm", 20, 250, 3),
    "systolic_bp": SignalSpec("Pressão sistólica", "mmHg", 40, 300, 5),
    "diastolic_bp": SignalSpec("Pressão diastólica", "mmHg", 20, 200, 3),
    "spo2": SignalSpec("SpO₂", "%", 50, 100, 1),
    "temperature": SignalSpec("Temperatura", "degC", 25, 45, 0.15),
}
TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", re.ASCII
)
NUMBER_RE = re.compile(r"\d{1,6}(?:\.\d{1,3})?", re.ASCII)


class ValidationError(ValueError):
    """Mensagem segura: não reproduz conteúdo livre recebido no CSV."""


@dataclass(frozen=True)
class Reading:
    timestamp: datetime
    signal: str
    value: float
    unit: str
    row: int


@dataclass(frozen=True)
class Dataset:
    readings: tuple[Reading, ...]
    fingerprint: str
    notices: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    reading: Reading
    history_count: int
    baseline: Optional[float]
    scale: Optional[float]
    delta: Optional[float]
    score: Optional[float]
    review: bool
    status: str


def fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def selection_key(source: str, payload: bytes) -> str:
    """A origem também invalida a seleção, mesmo com bytes idênticos."""
    return fingerprint(source.encode("utf-8") + b"\0" + payload)


def sync_selection(state: MutableMapping, key: Optional[str]) -> None:
    """Resultados pertencem somente aos dados atualmente selecionados."""
    if state.get("_mon_selected") != key:
        state.pop("_mon_preview", None)
        state.pop("_mon_results", None)
        state["_mon_selected"] = key


def _timestamp(value: str, row: int) -> datetime:
    if not TIMESTAMP_RE.fullmatch(value):
        raise ValidationError(
            f"Linha {row}: timestamp deve incluir segundos e fuso, "
            "como 2026-01-01T12:00:00Z."
        )
    # datetime.fromisoformat normaliza offsets inválidos como +02:60; não permitir.
    if value[-1] != "Z":
        hours, minutes = int(value[-5:-3]), int(value[-2:])
        if minutes > 59 or hours > 14 or (hours == 14 and minutes != 0):
            raise ValidationError(f"Linha {row}: fuso horário inválido.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        result = result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise ValidationError(f"Linha {row}: data ou hora inválida.") from None
    if not 2000 <= result.year <= 2100:
        raise ValidationError(f"Linha {row}: use datas fictícias entre 2000 e 2100.")
    return result


def parse_csv(payload: bytes) -> Dataset:
    """Validação estrita; qualquer erro rejeita o arquivo inteiro, sem correções."""
    if not isinstance(payload, bytes):
        raise ValidationError("A entrada deve ser um CSV em bytes UTF-8.")
    if not payload or len(payload) > MAX_BYTES:
        raise ValidationError("O CSV deve ter de 1 byte a 256 KiB.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError("Use um CSV de texto UTF-8, com ou sem BOM.") from None
    if "\x00" in text:
        raise ValidationError("O CSV contém um caractere nulo não permitido.")

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    readings = []
    previous = None
    last_by_signal = {}
    seen = set()
    gaps = defaultdict(int)
    counts = defaultdict(int)
    try:
        if tuple(next(reader, [])) != HEADER:
            raise ValidationError(
                "Cabeçalho obrigatório, nesta ordem: timestamp,signal,value,unit. "
                "Não inclua identificadores ou colunas extras. Use vírgulas."
            )
        for row, fields in enumerate(reader, start=2):
            if len(readings) >= MAX_ROWS:
                raise ValidationError("Limite de 2.000 medições por arquivo excedido.")
            if len(fields) != len(HEADER) or any(len(f) > MAX_CELL for f in fields):
                raise ValidationError(f"Linha {row}: exigidas 4 células de até 64 caracteres.")
            if any(not field or field != field.strip() for field in fields):
                raise ValidationError(f"Linha {row}: célula vazia ou espaço extra não permitido.")
            timestamp_text, signal, value_text, unit = fields
            timestamp = _timestamp(timestamp_text, row)
            if signal not in SIGNALS:
                raise ValidationError(f"Linha {row}: sinal desconhecido; consulte o esquema.")
            spec = SIGNALS[signal]
            if unit != spec.unit:
                raise ValidationError(f"Linha {row}: unidade incompatível com o sinal.")
            if not NUMBER_RE.fullmatch(value_text):
                raise ValidationError(
                    f"Linha {row}: valor deve ser decimal positivo, com ponto e até 3 casas."
                )
            value = float(value_text)
            if not math.isfinite(value) or not spec.minimum <= value <= spec.maximum:
                raise ValidationError(f"Linha {row}: valor fora da faixa de entrada do protótipo.")
            if previous is not None and timestamp < previous:
                raise ValidationError(f"Linha {row}: medições fora da ordem temporal UTC.")
            identity = (timestamp, signal)
            if identity in seen:
                raise ValidationError(f"Linha {row}: sinal e instante duplicados.")
            seen.add(identity)
            if signal in last_by_signal and timestamp - last_by_signal[signal] > MAX_GAP:
                gaps[signal] += 1
            last_by_signal[signal] = timestamp
            counts[signal] += 1
            previous = timestamp
            readings.append(Reading(timestamp, signal, value, unit, row))
    except csv.Error:
        raise ValidationError("CSV malformado ou célula excessivamente longa.") from None
    if not readings:
        raise ValidationError("O CSV precisa conter pelo menos uma medição.")

    notices = []
    for signal in SIGNALS:
        if signal in counts and counts[signal] <= MIN_HISTORY:
            notices.append(
                f"{SIGNALS[signal].label}: menos de 6 medições; histórico insuficiente "
                "para comparar esse sinal."
            )
        if gaps[signal]:
            notices.append(
                f"{SIGNALS[signal].label}: {gaps[signal]} intervalo(s) maior(es) que 6 h; "
                "a referência reinicia após cada intervalo."
            )
    return Dataset(tuple(readings), fingerprint(payload), tuple(notices))


def analyze(dataset: Dataset) -> tuple[Comparison, ...]:
    """Mediana/MAD das 10 medições ANTERIORES do mesmo sinal; mínimo de 5.

    A medição atual é adicionada à janela só depois da comparação. Após uma
    lacuna >6h, reinicia o aquecimento. Não interpola nem mistura sinais.
    """
    history = defaultdict(lambda: deque(maxlen=WINDOW))
    last_seen = {}
    results = []
    for reading in dataset.readings:
        series = history[reading.signal]
        if reading.signal in last_seen and reading.timestamp - last_seen[reading.signal] > MAX_GAP:
            series.clear()
        count = len(series)
        if count < MIN_HISTORY:
            result = Comparison(
                reading, count, None, None, None, None, False, "Histórico insuficiente"
            )
        else:
            baseline = float(median(series))
            mad = float(median(abs(value - baseline) for value in series))
            scale = max(1.4826 * mad, SIGNALS[reading.signal].scale_floor)
            delta = reading.value - baseline
            score = abs(delta) / scale
            review = score > REVIEW_SCORE
            result = Comparison(
                reading, count, baseline, scale, delta, score, review,
                "Revisar variação" if review else "Sem alerta estatístico",
            )
        results.append(result)
        series.append(reading.value)
        last_seen[reading.signal] = reading.timestamp
    return tuple(results)


def preview_rows(dataset: Dataset) -> list[dict]:
    return [
        {"Instante (UTC)": r.timestamp.isoformat(), "Sinal": SIGNALS[r.signal].label,
         "Valor": r.value, "Unidade": r.unit}
        for r in dataset.readings
    ]


def comparison_rows(results: tuple[Comparison, ...]) -> list[dict]:
    return [
        {"Instante (UTC)": r.reading.timestamp.isoformat(),
         "Sinal": SIGNALS[r.reading.signal].label, "Valor": r.reading.value,
         "Unidade": r.reading.unit, "Histórico (n)": r.history_count,
         "Mediana anterior": r.baseline, "Escala": r.scale,
         "Diferença": r.delta, "Score": r.score, "Resultado": r.status}
        for r in results
    ]


def chart_rows(results: tuple[Comparison, ...], signal: str) -> list[dict]:
    """Pontos reais do CSV validado; nenhum valor interpolado ou substituído.

    Cada lacuna >6h começa outro segmento para não conectar dados distantes.
    Não há cadência presumida: intervalos menores não representam ausência
    de amostras conhecida. A referência permanece None durante o aquecimento.
    """
    rows = []
    previous = None
    segment = 0
    for result in results:
        reading = result.reading
        if reading.signal != signal:
            continue
        if previous is not None and reading.timestamp - previous > MAX_GAP:
            segment += 1
        rows.append({
            "Instante": reading.timestamp, "Instante UTC": reading.timestamp.isoformat(),
            "Valor": reading.value,
            "Mediana anterior": result.baseline, "Score": result.score,
            "Revisar": result.review, "Segmento": segment, "Unidade": reading.unit,
        })
        previous = reading.timestamp
    return rows
