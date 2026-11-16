"""CSV acadêmico estrito, em memória; sem I/O, inferência de plano ou rede."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import re

from episodio_internacao_demo import (
    Episode, MovementEvent, MOVEMENT_ID, Observation, ProspectivePlan,
    TreatmentState, TREATMENT_ID,
)

SCHEMA_VERSION = "internacao-csv-v1"
MAX_BYTES = 128 * 1024
MAX_ROWS = 500
MAX_CELL = 64
HEADER = (
    "episode_id", "record_type", "channel", "timestamp", "value", "unit",
    "plan_id", "valid_from", "valid_until", "as_of", "source",
)
SOURCES = {"simulado", "atividade_autorizada"}
VALUES = {
    "prescricao": {"ativo": TreatmentState.ACTIVE, "pausado": TreatmentState.PAUSED},
    "movimento": {"no_leito": MovementEvent.IN_BED, "saida_leito": MovementEvent.LEFT_BED,
                  "retorno_leito": MovementEvent.RETURNED_TO_BED},
}
UNITS = {"prescricao": "estado", "movimento": "evento"}
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})", re.ASCII)
EPISODE_CODE = re.compile(r"EPISODIO-[A-Z0-9_-]{1,24}", re.ASCII)
PLAN_CODE = re.compile(r"PLANO-[A-Z0-9_-]{1,32}", re.ASCII)


class CSVValidationError(ValueError):
    """Erros curtos: nunca reproduzem células ou nomes de arquivos recebidos."""


@dataclass(frozen=True)
class ParsedEpisode:
    episode: Episode
    row_count: int
    declared_source: str
    fingerprint: str


def _instant(text: str, row: int, field: str) -> datetime:
    if not TIMESTAMP.fullmatch(text):
        raise CSVValidationError(f"Linha {row}: {field} exige data, segundos e fuso ISO 8601.")
    if not text.endswith("Z"):
        hours, minutes = int(text[-5:-3]), int(text[-2:])
        if hours > 14 or minutes > 59 or (hours == 14 and minutes):
            raise CSVValidationError(f"Linha {row}: fuso inválido em {field}.")
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise CSVValidationError(f"Linha {row}: data ou hora inválida em {field}.") from None
    if not 2000 <= value.year <= 2100:
        raise CSVValidationError(f"Linha {row}: use datas entre 2000 e 2100.")
    return value


def parse_episode_csv(payload: bytes) -> ParsedEpisode:
    """Rejeita o arquivo inteiro se o esquema falhar; não ordena ou preenche dados.

    Ausência de histórico, observação ou plano é preservada e será avaliada como
    dados insuficientes pelas regras existentes. Não escolhe uma linha histórica
    como observação atual e não cria um plano a partir do histórico.
    """
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_BYTES:
        raise CSVValidationError("Use um CSV de 1 byte a 128 KiB.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CSVValidationError("O CSV deve usar texto UTF-8.") from None
    if "\x00" in text:
        raise CSVValidationError("O CSV contém caractere nulo não permitido.")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    history = {channel: [] for channel in VALUES}
    current = {channel: None for channel in VALUES}
    plans = {channel: None for channel in VALUES}
    plan_ids = set()
    seen = set()
    identity = None
    previous = None
    count = 0
    try:
        if tuple(next(reader, [])) != HEADER:
            raise CSVValidationError("Cabeçalho incompatível. Use o esquema de Internação, sem colunas extras.")
        for cells in reader:
            row = reader.line_num
            count += 1
            if count > MAX_ROWS:
                raise CSVValidationError("Limite de 500 registros excedido.")
            if len(cells) != len(HEADER) or any(len(cell) > MAX_CELL for cell in cells):
                raise CSVValidationError(f"Linha {row}: exigidas 11 células de até 64 caracteres.")
            if any(cell != cell.strip() for cell in cells):
                raise CSVValidationError(f"Linha {row}: remova espaços extras nas células.")
            episode_id, kind, channel, when, raw, unit, plan_id, start, end, cutoff, source = cells
            if not EPISODE_CODE.fullmatch(episode_id):
                raise CSVValidationError(f"Linha {row}: use um código acadêmico EPISODIO-..., sem identificadores pessoais.")
            if kind not in {"history", "observation", "plan"} or channel not in VALUES:
                raise CSVValidationError(f"Linha {row}: tipo de registro ou canal não reconhecido.")
            if unit != UNITS[channel]:
                raise CSVValidationError(f"Linha {row}: unidade incompatível com o canal.")
            if source not in SOURCES:
                raise CSVValidationError(f"Linha {row}: origem deve ser simulado ou atividade_autorizada.")
            timestamp = _instant(when, row, "timestamp")
            as_of = _instant(cutoff, row, "as_of")
            if timestamp > as_of:
                raise CSVValidationError(f"Linha {row}: registro posterior ao corte as_of.")
            row_identity = (episode_id, as_of, source)
            if identity is not None and identity != row_identity:
                raise CSVValidationError(f"Linha {row}: use um único episódio, corte as_of e origem por arquivo.")
            identity = row_identity
            if previous is not None and timestamp < previous:
                raise CSVValidationError(f"Linha {row}: registros fora da ordem temporal UTC.")
            previous = timestamp
            key = (channel, kind, timestamp)
            if key in seen:
                raise CSVValidationError(f"Linha {row}: registro duplicado no mesmo canal e instante.")
            seen.add(key)
            if raw == "indisponivel" and kind != "plan":
                value = None
            elif raw in VALUES[channel]:
                value = VALUES[channel][raw]
            else:
                raise CSVValidationError(f"Linha {row}: valor incompatível com o canal ou tipo de registro.")
            if kind == "plan":
                if not PLAN_CODE.fullmatch(plan_id) or not start or not end:
                    raise CSVValidationError(f"Linha {row}: plano exige código PLANO-..., valid_from e valid_until.")
                if plans[channel] is not None or plan_id in plan_ids:
                    raise CSVValidationError(f"Linha {row}: use somente um plano por canal e códigos de plano distintos.")
                valid_from = _instant(start, row, "valid_from")
                valid_until = _instant(end, row, "valid_until")
                if not timestamp <= valid_from < valid_until:
                    raise CSVValidationError(f"Linha {row}: registro do plano deve anteceder sua vigência, com início menor que fim.")
                subject_id = TREATMENT_ID if channel == "prescricao" else MOVEMENT_ID
                plans[channel] = ProspectivePlan(plan_id, episode_id, subject_id, timestamp, valid_from, valid_until, value)
                plan_ids.add(plan_id)
            else:
                if any((plan_id, start, end)):
                    raise CSVValidationError(f"Linha {row}: campos de plano devem ficar vazios em histórico e observação.")
                observation = Observation(timestamp, value)
                if kind == "history":
                    history[channel].append(observation)
                else:
                    if current[channel] is not None:
                        raise CSVValidationError(f"Linha {row}: use somente uma observação atual por canal.")
                    current[channel] = observation
    except csv.Error:
        raise CSVValidationError("CSV malformado. Revise aspas, separadores e tamanho das células.") from None
    if not count or identity is None:
        raise CSVValidationError("O CSV precisa conter pelo menos um registro.")
    for channel in VALUES:
        observed = current[channel]
        if observed is not None:
            if any(old.at >= observed.at for old in history[channel]):
                raise CSVValidationError("O histórico precisa anteceder a observação atual do mesmo canal.")
            if plans[channel] is not None and plans[channel].recorded_at >= observed.at:
                raise CSVValidationError("O plano precisa ter sido registrado antes da observação atual.")
    episode_id, as_of, source = identity
    episode = Episode(
        episode_id, as_of, tuple(history["prescricao"]), current["prescricao"], plans["prescricao"],
        tuple(history["movimento"]), current["movimento"], plans["movimento"],
    )
    return ParsedEpisode(episode, count, source, hashlib.sha256(payload).hexdigest())
