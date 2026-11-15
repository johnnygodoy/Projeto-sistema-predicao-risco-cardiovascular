"""Episódio de internação inteiramente fictício para comparação de registros.

Biblioteca padrão, sem I/O, modelo clínico, vídeo, notificações ou dados reais.
As regras são convenções determinísticas de demonstração, não critérios clínicos.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Generic, Optional, TypeVar


EPISODE_ID = "EPISODIO-SIMULADO-01"
TREATMENT_ID = "ITEM-FICTICIO-A"
MOVEMENT_ID = "MOVIMENTO-FICTICIO"
ALGORITHM_VERSION = "episodio-demo-v2"
MAX_OBSERVATION_AGE = timedelta(minutes=15)  # Cadência técnica do exemplo, não clínica.


class TreatmentState(str, Enum):
    ACTIVE = "Ativo"
    PAUSED = "Pausado"


class MovementEvent(str, Enum):
    IN_BED = "No leito"
    LEFT_BED = "Saída do leito"
    RETURNED_TO_BED = "Retorno ao leito"


class Scenario(str, Enum):
    EXPECTED = "Plano A: mudança prevista"
    UNEXPECTED = "Plano B: mesma mudança fora do plano"
    UNAVAILABLE = "Dados atuais indisponíveis"
    NO_PLAN = "Plano prospectivo indisponível"


class Status(str, Enum):
    ALIGNED = "Sem divergências"
    REVIEW = "Alertas"
    UNAVAILABLE = "Dados insuficientes"


T = TypeVar("T", TreatmentState, MovementEvent)


@dataclass(frozen=True)
class Observation(Generic[T]):
    at: datetime
    value: Optional[T]


@dataclass(frozen=True)
class ProspectivePlan(Generic[T]):
    plan_id: str
    episode_id: str
    subject_id: str
    recorded_at: datetime
    valid_from: datetime
    valid_until: datetime
    expected: T


@dataclass(frozen=True)
class Episode:
    episode_id: str
    as_of: datetime
    treatment_history: tuple[Observation[TreatmentState], ...]
    treatment_current: Optional[Observation[TreatmentState]]
    treatment_plan: Optional[ProspectivePlan[TreatmentState]]
    movement_history: tuple[Observation[MovementEvent], ...]
    movement_current: Optional[Observation[MovementEvent]]
    movement_plan: Optional[ProspectivePlan[MovementEvent]]


@dataclass(frozen=True)
class Assessment:
    subject: str
    status: Status
    rule: str
    reason: str
    previous: str
    observed: str
    expected: str
    changed: Optional[bool]

    @property
    def alert(self) -> bool:
        return self.status is Status.REVIEW


@dataclass(frozen=True)
class Review:
    treatment: Assessment
    movement: Assessment

    @property
    def assessments(self) -> tuple[Assessment, Assessment]:
        return (self.treatment, self.movement)

    @property
    def alert_count(self) -> int:
        return sum(item.alert for item in self.assessments)

    @property
    def unavailable_count(self) -> int:
        return sum(item.status is Status.UNAVAILABLE for item in self.assessments)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 15, hour, minute, tzinfo=timezone.utc)


def build_demo(scenario: Scenario) -> Episode:
    """A e B compartilham histórico e observação; só o plano anterior difere."""
    if not isinstance(scenario, Scenario):
        raise ValueError("Selecione um cenário fictício conhecido.")
    expected = scenario is not Scenario.UNEXPECTED
    unavailable = scenario is Scenario.UNAVAILABLE
    plan_label = "A" if expected else "B"
    treatment_plan = ProspectivePlan(
        f"PLANO-{plan_label}-ITEM", EPISODE_ID, TREATMENT_ID, _at(9), _at(10),
        _at(10, 10), TreatmentState.PAUSED if expected else TreatmentState.ACTIVE,
    )
    movement_plan = ProspectivePlan(
        f"PLANO-{plan_label}-MOVIMENTO", EPISODE_ID, MOVEMENT_ID, _at(9), _at(10),
        _at(10, 10), MovementEvent.LEFT_BED if expected else MovementEvent.IN_BED,
    )
    return Episode(
        episode_id=EPISODE_ID,
        as_of=_at(10),
        treatment_history=(
            Observation(_at(8), TreatmentState.ACTIVE),
            Observation(_at(9, 55), TreatmentState.ACTIVE),
        ),
        treatment_current=Observation(_at(10), None if unavailable else TreatmentState.PAUSED),
        treatment_plan=None if scenario is Scenario.NO_PLAN else treatment_plan,
        movement_history=(
            Observation(_at(8), MovementEvent.IN_BED),
            Observation(_at(9, 55), MovementEvent.IN_BED),
        ),
        movement_current=Observation(_at(10), None if unavailable else MovementEvent.LEFT_BED),
        movement_plan=None if scenario is Scenario.NO_PLAN else movement_plan,
    )


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _label(value: object) -> str:
    return value.value if isinstance(value, (TreatmentState, MovementEvent)) else "Indisponível"


def _movement_is_consistent(records: tuple[Observation[MovementEvent], ...]) -> bool:
    """Validação de sequência do exemplo; não detecta movimento em imagem."""
    state = None
    for record in records:
        if record.value is MovementEvent.IN_BED:
            state = "in"
        elif record.value is MovementEvent.LEFT_BED and state == "in":
            state = "out"
        elif record.value is MovementEvent.RETURNED_TO_BED and state == "out":
            state = "in"
        else:
            return False
    return True


def _compare(
    episode: Episode, subject: str, subject_id: str, enum_type: type[T],
    history: tuple[Observation[T], ...], current: Optional[Observation[T]],
    plan: Optional[ProspectivePlan[T]],
) -> Assessment:
    previous = _label(history[-1].value) if history else "Indisponível"
    observed = _label(current.value) if current else "Indisponível"
    expected = _label(plan.expected) if plan else "Indisponível"

    def unavailable(rule: str, reason: str) -> Assessment:
        return Assessment(subject, Status.UNAVAILABLE, rule, reason,
                          previous, observed, expected, None)

    if current is None or current.value is None:
        return unavailable("D01", "Não há dado atual; ausência não significa estado normal.")
    if not isinstance(current.value, enum_type):
        return unavailable("D02", "O registro atual não pertence ao tipo de dado esperado.")
    if not _aware(episode.as_of) or not _aware(current.at):
        return unavailable("D03", "Instantes precisam incluir fuso horário.")
    age = episode.as_of - current.at
    if not timedelta(0) <= age <= MAX_OBSERVATION_AGE:
        return unavailable("D04", "Registro futuro ou antigo para a janela técnica de 15 minutos.")
    if not history or any(not isinstance(r.value, enum_type) or not _aware(r.at) for r in history):
        return unavailable("D05", "Histórico anterior ausente, incompleto ou de tipo incompatível.")
    times = tuple(record.at for record in history) + (current.at,)
    if any(a >= b for a, b in zip(times, times[1:])):
        return unavailable("D06", "Histórico deve estar em ordem estrita e anteceder o dado atual.")
    if enum_type is MovementEvent and not _movement_is_consistent(history + (current,)):
        return unavailable("D07", "Sequência de movimentos inconsistente no exemplo tipado.")
    if plan is None:
        return unavailable("P01", "Sem plano prospectivo, não é possível julgar a mudança inesperada.")
    if plan.episode_id != episode.episode_id or plan.subject_id != subject_id:
        return unavailable("P02", "Plano pertence a outro episódio ou item.")
    if not isinstance(plan.expected, enum_type):
        return unavailable("P03", "Plano usa um tipo de registro incompatível.")
    if not all(_aware(t) for t in (plan.recorded_at, plan.valid_from, plan.valid_until)):
        return unavailable("P04", "Os instantes do plano precisam incluir fuso horário.")
    if not (plan.recorded_at <= plan.valid_from < plan.valid_until and plan.recorded_at < current.at):
        return unavailable("P05", "Plano inválido ou não registrado antes da observação.")
    if not plan.valid_from <= current.at < plan.valid_until:
        return unavailable("P06", "O plano não cobre o instante observado.")

    changed = current.value != history[-1].value
    if current.value == plan.expected:
        return Assessment(subject, Status.ALIGNED, "C01",
                          "Registro coincide com o plano anterior; não há divergência neste exemplo.",
                          previous, observed, expected, changed)
    return Assessment(subject, Status.REVIEW, "C02",
                      "Registro difere do plano anterior válido; conferir os registros no painel interno.",
                      previous, observed, expected, changed)


def review_episode(episode: Episode) -> Review:
    """Cada canal é independente. Somente C02 gera alerta de divergência."""
    return Review(
        _compare(episode, "Prescrição", TREATMENT_ID, TreatmentState,
                 episode.treatment_history, episode.treatment_current, episode.treatment_plan),
        _compare(episode, "Movimento", MOVEMENT_ID, MovementEvent,
                 episode.movement_history, episode.movement_current, episode.movement_plan),
    )


def timeline_rows(episode: Episode) -> list[dict[str, str]]:
    rows = []
    for label, history, current in (
        ("Prescrição", episode.treatment_history, episode.treatment_current),
        ("Movimento", episode.movement_history, episode.movement_current),
    ):
        for record in history:
            rows.append({"Instante (UTC)": record.at.isoformat(), "Canal": label,
                         "Origem": "Histórico", "Registro": _label(record.value)})
        if current is not None:
            rows.append({"Instante (UTC)": current.at.isoformat(), "Canal": label,
                         "Origem": "Atual", "Registro": _label(current.value)})
    return sorted(rows, key=lambda row: (row["Instante (UTC)"], row["Canal"]))


def plan_rows(episode: Episode) -> list[dict[str, str]]:
    rows = []
    for label, plan in (("Prescrição", episode.treatment_plan),
                        ("Movimento", episode.movement_plan)):
        rows.append({
            "Canal": label,
            "Plano": plan.plan_id if plan else "Indisponível",
            "Registrado antes (UTC)": plan.recorded_at.isoformat() if plan else "Indisponível",
            "Início inclusivo (UTC)": plan.valid_from.isoformat() if plan else "Indisponível",
            "Fim exclusivo (UTC)": plan.valid_until.isoformat() if plan else "Indisponível",
            "Esperado no plano": _label(plan.expected) if plan else "Indisponível",
        })
    return rows


def review_rows(review: Review) -> list[dict[str, str]]:
    return [{"Canal": item.subject, "Anterior": item.previous, "Observado": item.observed,
             "Esperado": item.expected, "Resultado": item.status.value, "Regra": item.rule,
             "Motivo": item.reason} for item in review.assessments]


@dataclass(frozen=True)
class ReviewSentence:
    subject: str
    status: Status
    plan: str
    record: str
    difference: str
    next_action: str
    provenance: str


@dataclass(frozen=True)
class ReviewSummary:
    status: Status
    text: str
    items: tuple[ReviewSentence, ReviewSentence]


def _clock(value: object) -> str:
    if not _aware(value):
        return "horário indisponível"
    return value.astimezone(timezone.utc).strftime("%H:%M UTC")


def _review_action(assessment: Assessment, treatment: bool) -> str:
    """Conferência de evidências apenas; nunca recomenda conduta terapêutica."""
    channel = "prescrição" if treatment else "movimentação"
    if assessment.status is Status.UNAVAILABLE:
        if assessment.rule.startswith("P"):
            return f"Confira a versão, o horário e a vigência do plano de {channel} registrado."
        if assessment.rule in ("D05", "D06", "D07"):
            return f"Confira se o histórico de {channel} está completo e em ordem temporal."
        return f"Confira se há um registro de {channel} válido, com data, hora e fuso do período analisado."
    if assessment.alert:
        if treatment:
            return "Confira se existe atualização da prescrição registrada para esse horário."
        return "Confira a confirmação de movimentação e o horário do registro."
    if treatment:
        return "Confira a versão e o horário da prescrição usada na comparação."
    return "Confira o horário e a origem do registro de movimentação."


def summarize_review(episode: Episode, review: Review) -> ReviewSummary:
    """Frases fixas ligadas à evidência; sem geração livre, chatbot ou decisão clínica."""
    if review != review_episode(episode):
        raise ValueError("A revisão não corresponde aos registros e ao plano atuais.")
    if review.alert_count:
        status = Status.REVIEW
    elif review.unavailable_count:
        status = Status.UNAVAILABLE
    else:
        status = Status.ALIGNED
    compatible = sum(item.status is Status.ALIGNED for item in review.assessments)
    alert_label = "divergência" if review.alert_count == 1 else "divergências"
    compatible_label = "comparação" if compatible == 1 else "comparações"
    text = (f"{review.alert_count} {alert_label} para revisão, {compatible} {compatible_label} "
            f"sem divergências e {review.unavailable_count} com dados insuficientes.")
    items = []
    for assessment, plan, current, treatment in (
        (review.treatment, episode.treatment_plan, episode.treatment_current, True),
        (review.movement, episode.movement_plan, episode.movement_current, False),
    ):
        if plan is None:
            plan_text = "Plano prospectivo indisponível."
            plan_origin = "plano indisponível"
        else:
            plan_text = (f"{assessment.expected}. Registrado às {_clock(plan.recorded_at)}, "
                         f"com vigência de {_clock(plan.valid_from)} a {_clock(plan.valid_until)} (fim exclusivo).")
            plan_origin = plan.plan_id
        observed_time = _clock(current.at) if current else "horário indisponível"
        record_text = (f"Anterior: {assessment.previous}. Atual: {assessment.observed} "
                       f"({observed_time}).")
        if assessment.status is Status.UNAVAILABLE:
            difference = f"Não foi possível comparar. {assessment.reason}"
        elif assessment.alert:
            difference = (f"O plano previa {assessment.expected}; o registro informa {assessment.observed}.")
        elif assessment.changed:
            difference = "A mudança em relação ao histórico está prevista no plano."
        else:
            difference = "Registro e plano coincidem, sem mudança em relação ao histórico."
        items.append(ReviewSentence(
            assessment.subject, assessment.status, plan_text, record_text, difference,
            _review_action(assessment, treatment),
            f"Regra {assessment.rule} · {plan_origin} · versão {ALGORITHM_VERSION}",
        ))
    return ReviewSummary(status, text, (items[0], items[1]))
