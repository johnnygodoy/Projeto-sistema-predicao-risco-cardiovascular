# Áudio local

Página **🔊 Áudio** integrada ao menu nativo do Streamlit 1.32. Entradas na barra
lateral; player, transcrição e resultados no centro. Uso acadêmico, apenas com
gravações de teste, como voz sintética ou roteiro encenado, sem consultas reais
ou dados de pacientes.

## Requisitos

- Python 3.10–3.12 e o ambiente virtual usado pelo projeto; ensaio realizado em 3.12.
- Streamlit 1.32.0 e os requisitos públicos já instalados.
- faster-whisper 1.2.1 opcional, no **mesmo Python do Streamlit**.
- Pesos Whisper/CTranslate2 de fonte confiável em pasta local: `model.bin`,
  `config.json` e `tokenizer.json`. O app não baixa modelos.
- WAV RIFF, PCM16 mono, 16.000 Hz, de 0,25 segundo a **12 minutos**, até **24 MiB**.
  Exatamente um bloco `fmt ` PCM de 16 bytes antes de um bloco `data`.
- Metadados opcionais `LIST/INFO`, `JUNK` e `PAD ` são aceitos até 64 KiB somados,
  com limites de quantidade, tamanho e padding. São descartados; nunca apresentados
  nem entregues ao motor. Player e ASR recebem áudio sem esses metadados.
  Blocos desconhecidos, cabeçalho estendido, dados duplicados ou truncados são rejeitados.

Doze minutos nesse formato ocupam 23.040.044 bytes, aproximadamente 21,97 MiB,
mais eventuais metadados dentro do limite. Os limites de bytes e duração são
independentes. Um arquivo de cerca de 14 MiB cabe, desde que também tenha formato
e duração válidos. Tamanho sozinho não confirma que um WAV seja compatível.

Não é necessário instalar Azure ou alterar os requisitos básicos. As outras
páginas continuam funcionando sem faster-whisper. O BIDMC deste projeto é uma
fonte de sinais numéricos, não de áudio.

## Preparar o ambiente no Windows

Na raiz do projeto, ative a venv que você já utiliza. Exemplo se ela se chama `venv`:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import sys, streamlit, numpy, pyarrow; print(sys.executable); print('Streamlit', streamlit.__version__); print('NumPy', numpy.__version__); print('PyArrow', pyarrow.__version__)"
python -m pip check
```

Confira o Python exibido. Não troque nem reinstale o NumPy de um ambiente que está
compatível. O Streamlit 1.32 requer NumPy abaixo de 2. Se `pip check` apontar um
conflito ou o comando de importação falhar, resolva o problema mostrado antes de
continuar. `pip check` sozinho não detecta toda incompatibilidade binária.

Para instalar somente o recurso opcional, mantendo a restrição do Streamlit no
mesmo comando do resolvedor:

```powershell
python -m pip install --dry-run "streamlit==1.32.0" "faster-whisper==1.2.1"
python -m pip install "streamlit==1.32.0" "faster-whisper==1.2.1"
python -m pip check
```

Revise o plano antes do segundo comando. A verificação desse conjunto passou em
uma venv Linux isolada com NumPy 1.26.4 e PyArrow 15.0.2. Isso não verifica nem modifica sua venv
Windows. Não é preciso copiar código de outro projeto: um ambiente compatível e
pesos locais existentes podem ser reutilizados diretamente.

## Pesos locais

Se já possui os pesos compatíveis, informe sua pasta absoluta na barra lateral.
Não baixe de novo. Também pode configurar a pasta antes de iniciar:

```powershell
$env:FIAP_WHISPER_MODEL_DIR = "C:\modelos\faster-whisper-tiny"
```

**Somente se ainda não houver pesos:** obtenha o modelo oficial uma vez, fora do
repositório do projeto. O comando abaixo faz download de cerca de 78 MB, sem
executar inferência remota. Não é executado pelo aplicativo:

```powershell
$env:FIAP_WHISPER_MODEL_DIR = Join-Path $env:USERPROFILE "modelos\faster-whisper-tiny"
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-tiny', revision='d90ca5fe260221311c53c58e660288d3deb8d356', local_dir=os.environ['FIAP_WHISPER_MODEL_DIR'], allow_patterns=['model.bin','config.json','tokenizer.json','vocabulary.txt'], token=False)"
```

Use tiny **multilíngue** para Português. `tiny.en` aceita apenas Inglês. O aplicativo
exige pasta absoluta existente, nunca URL ou nome genérico de modelo. A falta do
tokenizer também interrompe a execução para evitar fallback de download.

## Executar e usar

```powershell
python -m streamlit run app/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

1. Abra **🔊 Áudio** e importe o WAV de teste.
2. Clique **Validar e ouvir**. Confira formato, duração e o áudio no player.
3. Informe a pasta do modelo e selecione o idioma.
4. Clique **Transcrever localmente**. Confira os segmentos ouvindo o arquivo.

O uploader só aparece com bind literal `127.0.0.1` ou `::1`. Não exponha a execução
por túnel ou proxy público. O Streamlit recebe o arquivo antes da validação de
24 MiB; o player só recebe o WAV após sua aprovação técnica. Isso não valida a
origem nem a ausência de dados pessoais no conteúdo.

Se um `.wav` for apenas um **ponteiro Git LFS**, a página explica que ele não contém
áudio e pede o arquivo completo. A detecção usa a estrutura exata de um ponteiro,
não o nome, a extensão ou um tamanho fixo. Não converta o texto do ponteiro em áudio.

Gravações de até 12 minutos podem ser usadas **inteiras, sem corte manual**.
Quando for necessário converter o formato, use seu FFmpeg já instalado:

```powershell
ffmpeg -n -i entrada.wav -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact audio_teste.wav
```

O comando não recorta nem presume que a fala começa no segundo zero. A opção
`-n` impede sobrescrever um arquivo existente. Se a duração exceder 12 minutos,
o app rejeitará o arquivo inteiro, sem truncar. Só nesse caso é necessário escolher
um trecho menor, ouvindo a origem antes. O app não executa FFmpeg,
não converte arquivos e não decodifica contêineres genéricos; entrega ao motor um
array PCM já validado.

## Processamento de áudio longo

O arquivo completo é lido uma vez pelo pipeline **local**. O filtro Silero VAD
incluído no faster-whisper seleciona trechos de fala antes da ASR. Usam-se os
parâmetros padrão: limiar 0,5, silêncio mínimo de 2.000 ms e margem de 400 ms.
Não há modelo adicional baixado em execução. Os eventos de telemetria do ONNX
Runtime são desativados antes de criar a sessão do filtro.

O próprio faster-whisper faz suas janelas e restaura os tempos ao arquivo original
com alinhamento de palavras (`vad_filter=True`, `word_timestamps=True`). O app não
faz cortes manuais, sobreposições, deduplicação ou filtros por palavras do resultado.
Um segmento pode conter mais de uma fala e incluir pausas; seus tempos continuam
aproximados. O limite curto de geração por janela foi removido; o limite finito
nativo do Whisper e os limites globais de saída continuam em vigor.

VAD não comprova ausência de fala e pode perder falas baixas, curtas ou atípicas.
A página informa a duração retida e mostra explicitamente quando o filtro não
detecta fala, sem inventar transcrição ou sentimento. Não é detector de tosse ou doença.

A execução é síncrona e pode levar vários minutos, com maior uso de CPU e memória.
Há no máximo 1.000 segmentos e 60.000 caracteres. Qualquer estouro descarta o
resultado, sem exibir uma transcrição parcial como completa.

O orçamento de processamento é de **15 minutos, cooperativo**: conferido em torno
do carregamento/preparação e entre segmentos retornados. Uma chamada nativa C++ ou
NumPy em andamento não pode ser interrompida por esse mecanismo e pode ultrapassar
o prazo. Não é um timeout rígido nem há cancelamento imediato. Se a execução não
responder, interromper o servidor com Ctrl+C também encerra a sessão do aplicativo.

Durante a execução, a barra indica a **posição do último segmento retornado**, não
uma medição de cobertura ou percentual de trabalho. Ela pode ficar parada em
intervalos sem segmentos, inclusive silêncio. Somente ao concluir todo o gerador
o resultado completo é guardado na sessão; mudanças de arquivo/idioma/modelo e
falhas continuam descartando os resultados antigos.

## O que é apresentado

- Transcrição real com tempos aproximados; sem texto de exemplo usado como resultado.
- Seis grupos de termos lexicais, com expressão original e pistas de negação ou
  incerteza. Janela de até cinco palavras anteriores, na mesma oração/segmento.
  São menções de texto, nunca sintomas confirmados ou negados.
- Polaridade lexical opcional: contagem explícita de palavras, não emoção, saúde
  mental, tom de voz ou diagnóstico. Ausência de palavras não significa neutralidade.

A ASR pode errar, omitir palavras ou alucinar texto. O resultado não alimenta o
score cardiovascular. Não há diarização, detector de doença, análise de tosse
ou integração com vídeo nesta página.

Áudio/transcrição ficam na memória da sessão, sem banco, arquivo temporário de mídia,
cache compartilhado ou envio a APIs. Limpar, remover/trocar arquivo, mudar pasta/idioma,
revalidar ou falhar em uma nova inferência descarta resultados antigos. Isso não
promete apagamento seguro da memória do sistema nem de cópias feitas fora do app.

## Verificação

A suíte completa passou em **110 testes** na integração, incluindo a amostra
oficial BIDMC externa. Ela inclui testes determinísticos, Streamlit 1.32 AppTest e, quando
fornecida, a cópia oficial externa BIDMC. A amostra deve permanecer fora do projeto:

```powershell
$env:FIAP_BIDMC_SAMPLE = "C:\dados\bidmc_01_Numerics.csv"
python -m unittest discover -s tests -v
```

AppTest 1.32 não controla o upload nativo. Seus testes injetam o arquivo na fronteira
do widget e usam dublês declarados para as saídas positivas de ASR. Eles verificam
validação até 12 minutos, fala apenas no final, tempos absolutos, progresso,
orçamento cooperativo, metadados válidos/inválidos, estados, navegação e falhas;
não demonstram qualidade do reconhecimento.

**Smoke real separado:** faster-whisper 1.2.1 + tiny na revisão acima reconheceram
um arquivo inglês de 7,395 s gerado localmente com a voz sintética Flite `slt`.
As 23 palavras coincidiram com a referência e duas negações foram preservadas.
É um único exemplo, sem validação de acurácia geral ou clínica. A referência foi
comparada depois da inferência, nunca fornecida como prompt ou hotword.

**Controle importante:** o Whisper chamado diretamente, sem VAD, alucinou “You”
sobre silêncio. O caminho do app rejeita PCM inteiramente zerado antes da ASR.
No ensaio separado com VAD, 12 minutos de silêncio e 30 segundos de ruído sintético
não produziram segmentos. Isso não garante o comportamento com todos os sons.

**Smoke longo separado:** um WAV sintético inglês de 8 minutos, 15.360.044 bytes,
com falas em 0,5–4,2 s, 65–68,665 s e 470–473,405 s, passou pelo motor local real.
As três falas foram reconhecidas, inclusive a última, sem corte em 60 segundos;
as duas negações foram preservadas. O arquivo curto também continuou funcionando.
Não se trata de consulta real, corpus de avaliação ou validação do PriMock.

Na primeira tentativa, **sem VAD**, surgiram 13 fragmentos “you” inventados nas
pausas e o último segmento começou em 450 s em vez de 470 s. Esse resultado negativo
foi mantido como evidência. Ativar só VAD eliminou os fragmentos, mas ainda juntou
intervalos distantes. Com VAD e alinhamento de palavras, na **mesma gravação**, não
houve fragmentos extras e a última fala ficou em 470,18–473,12 s. As negações e o
texto curto foram preservados; não foram removidas palavras por filtros de string.

Isso é um único ensaio sintético, não acurácia geral nem precisão de cronômetro.
Os segmentos ainda podem abranger pausas e exigem conferência pelo player. O consumo
de tempo e memória medido nesse executor Linux não prevê desempenho no Windows.

Também foi convertido um WAV de teste com o FFmpeg padrão, que adiciona LIST/INFO:
o arquivo foi aceito, os metadados foram removidos e os bytes PCM permaneceram iguais.
Testes de duplicatas, tamanhos, padding, truncamento e metadados excessivos rejeitam
as entradas sem chamar a ASR.

A inferência de ensaio usou pesos locais, Silero incluído no pacote, flags offline e conexões Python bloqueadas,
sem tentativas registradas. Não houve captura de pacotes. **Português falado, upload
nativo no navegador e Windows ainda não foram validados.** Pesos e áudios não fazem
parte deste repositório.

Fontes técnicas:
- [faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1)
- [Pesos oficiais na revisão fixa](https://huggingface.co/Systran/faster-whisper-tiny/tree/d90ca5fe260221311c53c58e660288d3deb8d356)
- [Streamlit 1.32: file_uploader](https://docs.streamlit.io/1.32.0/develop/api-reference/widgets/st.file_uploader)
