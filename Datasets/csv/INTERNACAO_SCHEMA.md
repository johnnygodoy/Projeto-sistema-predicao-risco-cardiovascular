# CSV de Internação

Use **Internação → Fonte → Upload CSV** na execução local. Selecione o arquivo e clique em **Comparar com o plano**. Os cenários da página continuam disponíveis. Este esquema é diferente do CSV de sinais usado em Monitoramento.

Exemplos inteiramente sintéticos, sem dados de pessoas:

- `internacao_alertas.csv`: 8 registros; 2 alertas, 0 comparações sem divergências e 0 com dados insuficientes.
- `internacao_controle.csv`: os mesmos registros observados, com outro plano anterior; 0 alertas, 2 comparações sem divergências e 0 com dados insuficientes.
- Para verificar ausência de plano, remova as duas linhas `plan` de uma cópia: resultado esperado de 0 alertas e 2 comparações com dados insuficientes. O programa não cria plano a partir do histórico.

## Esquema exato

CSV UTF-8 (BOM opcional), separado por vírgulas, com estas 11 colunas na ordem indicada:

```text
episode_id,record_type,channel,timestamp,value,unit,plan_id,valid_from,valid_until,as_of,source
```

| Campo | Regra |
| --- | --- |
| `episode_id` | Código acadêmico `EPISODIO-` seguido de 1–24 letras maiúsculas, números, `_` ou `-`. Não use nome, documento ou identificador de paciente. Um único episódio por arquivo. |
| `record_type` | `history` para registro anterior, `observation` para observação atual explícita ou `plan` para expectativa registrada antes da observação. |
| `channel` | `prescricao` ou `movimento`. O arquivo contempla somente um item terapêutico abstrato A e um canal de movimentação, sem medicamento ou dose. |
| `timestamp` | Data e hora do registro. Na linha `plan`, significa o instante em que o plano foi registrado, não sua execução. |
| `value` | Prescrição: `ativo` ou `pausado`. Movimento: `no_leito`, `saida_leito` ou `retorno_leito`. Em histórico/observação, `indisponivel` declara ausência; não pode ser usado como expectativa de plano. |
| `unit` | `estado` para prescrição e `evento` para movimento. São categorias tipadas, não grandezas clínicas ou ângulos. |
| `plan_id` | Somente em `plan`: código `PLANO-` seguido de 1–32 letras maiúsculas, números, `_` ou `-`. No máximo um plano por canal, com códigos distintos. |
| `valid_from` / `valid_until` | Somente em `plan`: início inclusivo e fim exclusivo de sua vigência. Nas demais linhas, os dois campos e `plan_id` devem ficar vazios. |
| `as_of` | Corte temporal da revisão, explícito e igual em todas as linhas. Nenhum `timestamp` pode estar no futuro desse corte. |
| `source` | Origem declarada: `simulado` ou `atividade_autorizada`, igual em todas as linhas. Essa declaração não comprova autorização nem detecta dados sensíveis. |

Datas exigem segundos e fuso ISO 8601, como `2026-01-15T10:00:00Z` ou `2026-01-15T07:00:00-03:00`, entre 2000 e 2100. A comparação usa UTC. Não há datas sem fuso, frações de segundo, correção de unidades, preenchimento automático nem reordenação.

## Ordem e plano prospectivo

- Todas as linhas devem vir em ordem temporal não decrescente de `timestamp` em UTC. Empates entre canais são permitidos; registros duplicados do mesmo tipo/canal/instante são rejeitados.
- Aceita-se no máximo uma observação atual por canal. Todo histórico desse canal deve antecedê-la estritamente. A última linha histórica **não** vira observação atual automaticamente.
- O registro do plano deve ocorrer até o início de sua vigência e antes da observação atual. Início e fim devem ser distintos e crescentes. Planos retroativos são rejeitados.
- Um plano válido mas cuja vigência não cobre a observação gera **Dados insuficientes**, não alerta de mudança inesperada.
- Se faltarem histórico, observação atual ou plano, a falta é preservada pelas regras. Não significa normalidade nem deve ser preenchida com um valor inventado.
- A observação atual pode anteceder `as_of`. O núcleo existente exige atualidade de até 15 minutos; acima disso, retorna dados insuficientes. Esse intervalo é uma convenção de teste, não um limiar clínico.

## Limites e privacidade

Máximo de 128 KiB, 500 registros e 64 caracteres por célula. O cabeçalho é exato: não inclua colunas extras, texto livre, identificadores pessoais ou fórmulas. Espaços extras nas células, tipos desconhecidos ou qualquer erro de estrutura rejeitam o arquivo inteiro com uma mensagem curta.

O upload só é exibido com bind explícito `127.0.0.1` ou `::1`. Exemplo:

```text
python -m streamlit run app/app.py --server.address=127.0.0.1
```

Não exponha a execução por túnel ou proxy público. O Streamlit recebe o arquivo antes da validação de 128 KiB; o código verifica o tamanho informado antes de copiar seus bytes e confere o tamanho real novamente. O processamento é em memória, sem gravação do CSV, cache em disco ou envio externo. Limpar, remover/trocar o arquivo, mudar a fonte ou perder a configuração local invalida os resultados; uma nova comparação exige novo clique.

Use somente dados sintéticos ou material de atividade acadêmica autorizado e não identificável. Este parser não autentica essa origem. O resumo usa regras determinísticas para conferir registros e planos: não é prontuário, monitoramento contínuo, chatbot, diagnóstico ou recomendação terapêutica.
