"""Revisão local de UM arquivo oficial BIDMC; sem rede, gravação ou diagnóstico.

Nenhuma linha da amostra é incluída neste módulo. A autenticação por hash
precede o parsing. Não é um importador genérico de dados clínicos.
"""

from dataclasses import dataclass
import csv
import hashlib
import io
import math
import re
from typing import Optional


MODE_LABEL = "Amostra de pesquisa BIDMC (PhysioNet)"
FILENAME = "bidmc_01_Numerics.csv"
EXPECTED_BYTES = 7644
EXPECTED_SHA256 = "27d43794cf2e7150db10e8bfe61d2bfdb26873a8f1629ffc5fbee5c9a4eb6ef5"
SOURCE_URL = "https://physionet.org/content/bidmc/1.0.0/"
FILE_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_01_Numerics.csv"
CHECKSUM_URL = "https://physionet.org/files/bidmc/1.0.0/SHA256SUMS.txt"
DOI_URL = "https://doi.org/10.13026/C2208R"
LICENSE_URL = "https://physionet.org/content/bidmc/view-license/1.0.0/"
HEADER = ("Time [s]", "HR", "PULSE", "RESP", "SpO2")
EXPECTED_ROWS = 481
SIGNALS = ("HR", "SpO2", "RESP", "PULSE")
UNITS = {"HR": "bpm", "SpO2": "%", "RESP": "resp/min", "PULSE": "bpm"}
EXPECTED_MISSING = {"HR": 0, "SpO2": 13, "RESP": 0, "PULSE": 13}
NUMBER = re.compile(r"\d{1,6}(?:\.\d{1,3})?", re.ASCII)


class SampleError(ValueError):
    """Erros não reproduzem nomes arbitrários, linhas ou dados recebidos."""


@dataclass(frozen=True)
class ResearchSample:
    seconds: tuple[int, ...]
    series: tuple[tuple[Optional[float], ...], ...]
    fingerprint: str

    def values(self, signal: str) -> tuple[Optional[float], ...]:
        return self.series[SIGNALS.index(signal)]


def verify_and_parse(filename: str, payload: bytes) -> ResearchSample:
    if filename != FILENAME:
        raise SampleError("Selecione somente bidmc_01_Numerics.csv, com o nome original.")
    if not isinstance(payload, bytes) or len(payload) != EXPECTED_BYTES:
        raise SampleError("O tamanho não corresponde ao arquivo oficial autorizado.")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise SampleError("SHA-256 diferente do oficial. Não edite nem converta o CSV; use a cópia original.")
    return _parse_verified_bytes(payload)


def _parse_verified_bytes(payload: bytes) -> ResearchSample:
    """Interno: nunca usar como entrada de UI; chamar verify_and_parse primeiro."""
    if len(payload) > 32768:
        raise SampleError("Arquivo acima do limite interno de leitura.")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise SampleError("Codificação inesperada para a amostra oficial.") from None
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    seconds = []
    columns = {signal: [] for signal in SIGNALS}
    try:
        if tuple(cell.strip() for cell in next(reader, [])) != HEADER:
            raise SampleError("Estrutura diferente da amostra oficial.")
        for row in reader:
            if len(seconds) >= EXPECTED_ROWS or len(row) != 5 or any(len(cell) > 32 for cell in row):
                raise SampleError("Dimensões inesperadas para a amostra oficial.")
            if not row[0].isascii() or not row[0].isdigit() or int(row[0]) != len(seconds):
                raise SampleError("A escala deve conter segundos relativos consecutivos de 0 a 480.")
            seconds.append(int(row[0]))
            for signal, cell in zip(HEADER[1:], row[1:]):
                cell = cell.strip()
                if cell == "NaN":
                    value = None
                elif NUMBER.fullmatch(cell):
                    value = float(cell)
                    if not math.isfinite(value):
                        raise SampleError("Valor não finito inesperado na amostra.")
                else:
                    raise SampleError("Valor inesperado na amostra oficial.")
                columns[signal].append(value)
    except csv.Error:
        raise SampleError("CSV inválido; mantenha o arquivo oficial sem alterações.") from None
    if len(seconds) != EXPECTED_ROWS:
        raise SampleError("São esperados 481 instantes relativos, de 0 a 480 segundos.")
    if any(sum(v is None for v in columns[s]) != EXPECTED_MISSING[s] for s in SIGNALS):
        raise SampleError("Contagem de ausências diferente da amostra oficial.")
    return ResearchSample(tuple(seconds), tuple(tuple(columns[s]) for s in SIGNALS),
                          hashlib.sha256(payload).hexdigest())


def coverage_rows(sample: ResearchSample) -> list[dict]:
    result = []
    for signal in SIGNALS:
        values = sample.values(signal)
        present = sum(value is not None for value in values)
        result.append({"Sinal": signal, "Unidade": UNITS[signal], "Presentes": present,
                       "Ausentes": len(values) - present,
                       "Cobertura (%)": round(100 * present / len(values), 2)})
    return result


def signal_rows(sample: ResearchSample, signal: str) -> list[dict]:
    """Preserva None e separa trechos após cada ausência, sem fabricar pontos."""
    values = sample.values(signal)
    result = []
    segment = 0
    after_gap = False
    for second, value in zip(sample.seconds, values):
        if value is None:
            after_gap = True
        elif after_gap:
            segment += 1
            after_gap = False
        result.append({"Segundo relativo": second, "Valor": value, "Trecho": segment,
                       "Sinal": signal, "Unidade": UNITS[signal]})
    return result
