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
- WAV RIFF canônico, PCM16 mono, 16.000 Hz, de 0,25 a 60 segundos, até 2 MiB.
  Somente blocos `fmt ` de 16 bytes e `data`; sem metadados/cabeçalho estendido.

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
2 MiB; o player só recebe o WAV após sua aprovação técnica. Isso não valida a
origem nem a ausência de dados pessoais no conteúdo.

Se um `.wav` for apenas um **ponteiro Git LFS**, a página explica que ele não contém
áudio e pede o arquivo completo. A detecção usa a estrutura exata de um ponteiro,
não o nome, a extensão ou um tamanho fixo. Não converta o texto do ponteiro em áudio.

Ouça a origem e escolha antes um trecho com fala, de até 60 segundos. Não presuma
que os primeiros segundos contêm fala. Para converter esse trecho de teste com
FFmpeg já instalado:

```powershell
ffmpeg -n -i trecho.wav -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact audio_teste.wav
```

O comando não recorta: `trecho.wav` deve conter o intervalo escolhido. A opção
`-n` impede sobrescrever um arquivo existente. Se a duração ainda exceder 60 s,
escolha um trecho menor; o app o rejeitará sem truncar. O app não executa FFmpeg,
não converte arquivos e não decodifica contêineres genéricos; entrega ao motor um
array PCM já validado.

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

A suíte completa passou em **95 testes** na integração, incluindo a amostra
oficial BIDMC externa. Ela inclui testes determinísticos, Streamlit 1.32 AppTest e, quando
fornecida, a cópia oficial externa BIDMC. A amostra deve permanecer fora do projeto:

```powershell
$env:FIAP_BIDMC_SAMPLE = "C:\dados\bidmc_01_Numerics.csv"
python -m unittest discover -s tests -v
```

AppTest 1.32 não controla o upload nativo. Seus testes injetam o arquivo na fronteira
do widget e usam dublês declarados para as saídas positivas de ASR. Eles verificam
validação, estados, navegação e falhas; não demonstram qualidade do reconhecimento.

**Smoke real separado:** faster-whisper 1.2.1 + tiny na revisão acima reconheceram
um arquivo inglês de 7,395 s gerado localmente com a voz sintética Flite `slt`.
As 23 palavras coincidiram com a referência e duas negações foram preservadas.
É um único exemplo, sem validação de acurácia geral ou clínica. A referência foi
comparada depois da inferência, nunca fornecida como prompt ou hotword.

**Controle importante:** o Whisper chamado diretamente alucinou “You” sobre silêncio.
O caminho do aplicativo rejeitou o mesmo WAV com PCM inteiramente zerado antes da
ASR. Essa proteção não detecta silêncio ruidoso, música ou outros sons sem fala.

A inferência de ensaio usou pesos locais, flags offline e conexões Python bloqueadas,
sem tentativas registradas. Não houve captura de pacotes. **Português falado, upload
nativo no navegador e Windows ainda não foram validados.** Pesos e áudios não fazem
parte deste repositório.

Fontes técnicas:
- [faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1)
- [Pesos oficiais na revisão fixa](https://huggingface.co/Systran/faster-whisper-tiny/tree/d90ca5fe260221311c53c58e660288d3deb8d356)
- [Streamlit 1.32: file_uploader](https://docs.streamlit.io/1.32.0/develop/api-reference/widgets/st.file_uploader)
