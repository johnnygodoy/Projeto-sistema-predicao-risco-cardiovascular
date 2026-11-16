# Áudio local

Página **🔊 Áudio** integrada ao menu nativo do Streamlit 1.32. Entradas na barra
lateral; player, transcrição e resultados no centro. Uso acadêmico, apenas com
gravações de teste, como voz sintética ou roteiro encenado, sem consultas reais
ou dados de pacientes.

## Requisitos

- Python 3.10–3.12 e o ambiente virtual usado pelo projeto; ensaio realizado em 3.12.
- Streamlit 1.32.0 e os requisitos públicos já instalados.
- faster-whisper 1.2.1 opcional, no **mesmo Python do Streamlit**.
- Apenas para MP3: FFmpeg local já instalado e disponível no `PATH` do servidor.
  Não há nova dependência Python. Sem FFmpeg, o caminho WAV continua funcionando.
- Pesos Whisper/CTranslate2 de fonte confiável em `models/whisper-tiny` na raiz
  local do projeto: `model.bin`, `config.json` e `tokenizer.json`. O app não baixa modelos.
- WAV RIFF, PCM16 mono, 16.000 Hz, de 0,25 segundo a **12 minutos**, até **24 MiB**.
  Exatamente um bloco `fmt ` PCM de 16 bytes antes de um bloco `data`.
- Metadados opcionais `LIST/INFO`, `JUNK` e `PAD ` são aceitos até 64 KiB somados,
  com limites de quantidade, tamanho e padding. São descartados; nunca apresentados
  nem entregues ao motor. Player e ASR recebem áudio sem esses metadados.
  Blocos desconhecidos, cabeçalho estendido, dados duplicados ou truncados são rejeitados.
- MP3 Layer III (MPEG 1, 2 ou 2.5), mono ou estéreo, taxa de origem fixa entre
  8.000 e 48.000 Hz. Bitrate variável é aceito. Até 24 MiB no arquivo e até
  12 minutos **depois da decodificação**, com mínimo de 0,25 segundo.
  O aplicativo normaliza para o mesmo PCM16 mono a 16.000 Hz usado pelo WAV.
  Outros formatos, como M4A, AAC e OGG, ainda não são aceitos.

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

## Pesos locais dentro do projeto

Não é mais necessário escolher a pasta na tela. O padrão é
`models/whisper-tiny`, resolvido pela localização do código, independentemente da
pasta de onde o Streamlit for iniciado. É uma pasta **local**, ignorada pelo Git.
O modelo cardiovascular já existente em `model/model.pkl` não é alterado nem ignorado.

Se os pesos já estão no destino, não copie novamente. Caso estejam em outra pasta
local, na raiz do projeto copie apenas os arquivos conhecidos com o roteiro abaixo.
Ele verifica os arquivos obrigatórios antes de criar o destino, recusa um destino
já existente e nunca sobrescreve arquivos. Não usa curingas nem baixa arquivos.

```powershell
$origem = "C:\modelos\faster-whisper-tiny"  # pasta local que já contém seus pesos
$destino = Join-Path (Get-Location) "models\whisper-tiny"
if (-not (Test-Path -LiteralPath "app\audio_local.py" -PathType Leaf)) { throw "Execute na raiz do projeto." }
$obrigatorios = @("model.bin", "config.json", "tokenizer.json")
$opcionais = @("vocabulary.txt", "preprocessor_config.json")
foreach ($nome in $obrigatorios) {
    $arquivo = Join-Path $origem $nome
    if (-not (Test-Path -LiteralPath $arquivo -PathType Leaf)) { throw "Arquivo ausente: $nome" }
    if ((Get-Item -LiteralPath $arquivo).Length -eq 0) { throw "Arquivo vazio: $nome" }
}
if (Test-Path -LiteralPath $destino) { throw "Destino já existe. Confira os arquivos; nada será sobrescrito." }
$arquivos = @($obrigatorios) + @($opcionais | Where-Object { Test-Path -LiteralPath (Join-Path $origem $_) -PathType Leaf })
[System.IO.Directory]::CreateDirectory($destino) | Out-Null
foreach ($nome in $arquivos) {
    [System.IO.File]::Copy((Join-Path $origem $nome), (Join-Path $destino $nome), $false)
}
```

Apenas esse roteiro manual copia arquivos. O aplicativo não cria a pasta, não move
pesos e não escreve no disco. Não inclua os pesos no Git nem use `git add -f` nessa
pasta. Se uma cópia falhar, o aplicativo continuará indisponível até a preparação
ser concluída; não substitua arquivos de origem nem sobrescreva o destino às cegas.

Para reutilizar outra pasta sem copiar, a configuração opcional **do servidor**
continua disponível antes de iniciar o aplicativo:

```powershell
$env:FIAP_WHISPER_MODEL_DIR = "C:\modelos\faster-whisper-tiny"
```

Essa variável tem prioridade sobre o padrão. Para voltar à pasta do projeto,
remova a variável na sessão PowerShell que iniciará o Streamlit:

```powershell
Remove-Item Env:\FIAP_WHISPER_MODEL_DIR -ErrorAction SilentlyContinue
```

O override exige pasta absoluta local existente; URL, caminho de rede e nome
como `tiny` continuam recusados. Não existe campo de caminho na interface.

Use tiny **multilíngue** para Português; `tiny.en` aceita apenas Inglês. Na ausência
dos arquivos obrigatórios ou dos módulos opcionais, a página mostra uma única
mensagem curta de preparação e desabilita a transcrição. Validar e ouvir continuam
disponíveis. Essa verificação de presença não garante compatibilidade binária ou
integridade dos pesos; o carregamento real continua validado no momento da inferência.

## Executar e usar

```powershell
python -m streamlit run app/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

1. Abra **🔊 Áudio** e importe o WAV ou MP3 de teste.
2. Clique **Validar e ouvir**. Confira formato, duração e o áudio no player.
3. Selecione o **idioma falado no áudio**; os pesos locais são encontrados automaticamente.
4. Clique **Transcrever localmente**. Confira os segmentos ouvindo o arquivo.

Para um áudio em inglês, selecione **Inglês**: a transcrição permanece em inglês.
A escolha de Português não traduz o áudio. Tradução para português não faz parte
desta página.

O uploader só aparece com bind literal `127.0.0.1` ou `::1`. Não exponha a execução
por túnel ou proxy público. O Streamlit recebe o arquivo antes da validação de
24 MiB; o player só recebe o WAV após sua aprovação técnica. Isso não valida a
origem nem a ausência de dados pessoais no conteúdo.

Se um `.wav` for apenas um **ponteiro Git LFS**, a página explica que ele não contém
áudio e pede o arquivo completo. A detecção usa a estrutura exata de um ponteiro,
não o nome, a extensão ou um tamanho fixo. Não converta o texto do ponteiro em áudio.

Gravações dentro do limite decodificado de 12 minutos podem ser usadas
**inteiras, sem corte manual**. O padding MP3 pode exceder esse limite, como
explicado abaixo; WAV canônico de exatamente 12 minutos é aceito.
Quando for necessário converter o formato, use seu FFmpeg já instalado:

```powershell
ffmpeg -n -i entrada.wav -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 -fflags +bitexact audio_teste.wav
```

O comando não recorta nem presume que a fala começa no segundo zero. A opção
`-n` impede sobrescrever um arquivo existente. Se a duração exceder 12 minutos,
o app rejeitará o arquivo inteiro, sem truncar. Só nesse caso é necessário escolher
um trecho menor, ouvindo a origem antes. O caminho WAV continua sem conversão
automática. Para MP3, a conversão local descrita abaixo prepara o mesmo array PCM
validado antes de entregá-lo ao motor, sem alterar o arquivo original.

## Importação MP3

Selecione o MP3 diretamente, sem renomear sua extensão para WAV. Antes de abrir
o decodificador, o aplicativo verifica bytes e percorre todos os quadros Layer III,
rejeitando formato misto, quadros truncados, bytes adicionais e duração excessiva.
Essa etapa verifica estrutura, não qualidade ou segurança semântica do conteúdo.

São aceitos tags iniciais ID3v2.3/2.4 sem footer e um tag final ID3v1, até 64 KiB
somados. Os tags são delimitados, descartados e nunca apresentados ou enviados ao
decodificador. ID3v2.2, footer, APEv2, streams free-format e outros metadados não
são suportados. Arquivos válidos fora desse subconjunto podem ser recusados.

O FFmpeg é opcional e executado em subprocesso com argumentos fixos, sem shell,
sem caminhos fornecidos pelo upload e sem arquivos temporários de mídia. A entrada
é explicitamente MP3/Layer III, e os únicos protocolos permitidos são pipes em
memória. Não há download, URL de mídia ou chamada a serviço de conversão.
A saída é limitada a 23.040.000 bytes PCM e novamente validada. O player recebe
somente o WAV resultante; nunca recebe o MP3 ainda não validado.

A espera pelo decodificador tem prazo de 60 segundos, seguido pelo encerramento
do processo e das threads. Não é promessa de 60 segundos exatos para a operação
inteira. Timeout, estouro de saída ou erro descartam tudo, sem transcrição parcial.
Há limite de alocação individual do FFmpeg de 64 MiB, **não** um teto da memória
total do processo. Isso é isolamento de processo, não um sandbox de segurança
do sistema operacional. Use um FFmpeg atualizado de fonte confiável.

MP3 é um formato com perdas. A conversão não melhora automaticamente a transcrição,
e delay/padding do codificador pode mudar ligeiramente a duração e os tempos.
Por isso, um MP3 cuja origem tem exatamente 12 minutos pode ultrapassar o limite
decodificado e ser recusado. O app não remove amostras para fazê-lo caber.
WAV canônico de exatamente 12 minutos continua aceito.

No mesmo terminal que inicia o Streamlit, confira a disponibilidade:

```powershell
ffmpeg -version
```

Se o comando não for encontrado, disponibilize no `PATH` seu FFmpeg já instalado
e reinicie o servidor; não basta ter PyAV no Python. O aplicativo não instala
FFmpeg nem altera o `PATH`. O botão de validação apresenta uma mensagem curta
quando o executável não está disponível.

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
o resultado completo é guardado na sessão; mudanças de arquivo/idioma/configuração do modelo e
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
cache compartilhado ou envio a APIs. Limpar, remover/trocar arquivo, mudar a configuração do modelo ou o idioma,
revalidar ou falhar em uma nova inferência descarta resultados antigos. Isso não
promete apagamento seguro da memória do sistema nem de cópias feitas fora do app.

## Verificação

A suíte inclui testes determinísticos, Streamlit 1.32 AppTest e, quando
fornecida, a cópia oficial externa BIDMC. A amostra deve permanecer fora do projeto:

```powershell
$env:FIAP_BIDMC_SAMPLE = "C:\dados\bidmc_01_Numerics.csv"
python -m unittest discover -s tests -v
```

AppTest 1.32 não controla o upload nativo. Seus testes injetam o arquivo na fronteira
do widget e usam dublês declarados para as saídas positivas de ASR. Eles verificam
validação até 12 minutos, fala apenas no final, tempos absolutos, progresso,
orçamento cooperativo, metadados válidos/inválidos, estados, navegação e falhas;
também verificam a pasta padrão independente do diretório atual, override do servidor,
ausência do campo de caminho e reprodução disponível sem o motor preparado;
não demonstram qualidade do reconhecimento.

Os testes de MP3 cobrem limites de bytes/quadros/duração/amostras, descarte de
metadados, formatos renomeados, corrupção, prazo e encerramento do subprocesso,
ausência de FFmpeg e saída inválida. Dublês de decodificação e ASR verificam que
apenas PCM aprovado chega ao player/motor. Remover, substituir, revalidar ou limpar
um arquivo invalida a seleção, incluindo resultados/erros que terminem atrasados.
Esses testes não substituem execução nativa de FFmpeg ou reconhecimento real.

**Smoke real de MP3 separado:** a mesma voz sintética inglesa foi codificada
localmente em MP3 estéreo a 44.100 Hz e reconhecida pelo tiny offline. As 23
palavras coincidiram com a referência após normalizar caixa e pontuação, incluindo
as duas negações; a pontuação mudou. A duração normalizada foi 7,419875 s, contra
7,395 s no WAV de origem. Não é prova de acurácia geral. Um tom sintético de 3 s
não produziu segmentos; silêncio MP3 foi rejeitado antes da ASR por gerar PCM zero.
Um MP3 VBR sintético de 8 minutos também preservou as três falas e as duas negações,
com a última fala em 470,18–473,14 s. Um segmento ainda reuniu as duas primeiras
falas e suas pausas; tempos não são uma segmentação exata de atividade vocal.

MP3 corrompido, truncado, WAV apenas renomeado e origem de 721 s foram recusados.
Uma origem de exatamente 720 s foi recusada após decodificação por causa de
padding adicional, como previsto pelo limite estrito. Nenhum original foi alterado.
Os controles MP3 reais usaram FFmpeg 7.1.5 em Linux; não validam execução Windows,
todos os codificadores, fala portuguesa ou upload nativo no navegador.

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
nativo no navegador e Windows ainda não foram validados.** Pesos e arquivos gerados
nesses ensaios não são incluídos pelo recurso de áudio. O kit `Datasets/audio`
contém separadamente as amostras sintéticas e seus avisos de licença.

Fontes técnicas:
- [faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1)
- [Pesos oficiais na revisão fixa](https://huggingface.co/Systran/faster-whisper-tiny/tree/d90ca5fe260221311c53c58e660288d3deb8d356)
- [Streamlit 1.32: file_uploader](https://docs.streamlit.io/1.32.0/develop/api-reference/widgets/st.file_uploader)
- [FFmpeg: opções](https://ffmpeg.org/ffmpeg.html)
- [FFmpeg: protocolo pipe](https://ffmpeg.org/ffmpeg-protocols.html#pipe)
- [ID3v2.4: estrutura](https://id3.org/id3v2.4.0-structure)
