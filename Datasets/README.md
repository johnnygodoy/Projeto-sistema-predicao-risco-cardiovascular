# Mídias de teste do Tech Challenge

Kit didático de reprodução para a equipe. **Não contém prontuários, gravações de pacientes, pesos de modelos ou cópias do BIDMC.** Os CSVs e o texto falado são fictícios. Os vídeos são recortes autorizados de um dataset acadêmico de exercícios encenados, com pessoas reais e licença própria. Nada aqui valida diagnóstico, risco clínico, sentimento de uma pessoa ou qualidade de tratamento.

Compatibilidade e medições abaixo verificadas sobre o código `f1ba9f1004478eccd4ab7ebd5d206760ec1ddc94`. As mídias têm créditos/licenças próprios, independentes da licença do código do repositório.

## O que abrir

- **Áudio local:** [fala sintética inglesa](audio/synthetic_fictional_en.wav), 7,395 s, PCM16 mono, 16 kHz, 236.684 bytes. Selecione **Inglês**. O [texto conhecido](audio/intended_transcript_en.txt) serve para conferir o resultado, não para alimentar o reconhecedor.
- **Controle de áudio:** [silêncio digital](audio/silence_paired.wav), mesmo formato/duração. O aplicativo deve recusar a transcrição desse arquivo inteiramente zerado.
- **Monitoramento didático:** [controle sintético](csv/controle_sintetico_120.csv), 120 leituras e nenhum alerta estatístico; [variações sintéticas](csv/variacoes_sinteticas_120.csv), 120 leituras e quatro alertas. Os 25 primeiros resultados, cinco por sinal, têm histórico insuficiente. Não são alertas médicos.
- **Vídeo de movimento:** os cinco nomes e hashes exatos estão no [manifesto de vídeo](video/manifest.json). Eles podem ser obtidos do commit que contém os MP4 ou recriados localmente com a receita abaixo. Se a pasta só tiver este manifesto, os arquivos ainda não estão nesse checkout: não são baixados automaticamente pelo app.

Os exemplos automáticos já existentes em `app/monitoramento_exemplos` continuam no lugar. Os dois CSVs deste kit são diferentes: 24 instantes separados por dois minutos, cinco sinais por instante, em 1º de janeiro de 2025, data fictícia. Cabeçalho: `timestamp,signal,value,unit`. Não possuem identificadores, campos livres, pacientes ou vínculo com a aba Internação. `Datasets/BIDMC`, quando existir apenas na máquina de um colega, não é fonte destes exemplos e não é necessário para rodá-los.

## Vídeos: ordem e referência

Todos os arquivos são MP4/H.264, yuv420p, 1280×720, 30 fps constantes e sem áudio. Não houve recorte espacial, máscara de pessoas, alteração de brilho ou interpolação. Os intervalos abaixo são índices de frames decodificados do FFmpeg: início incluído, fim excluído.

| Arquivo | Fonte oficial | Frames | Duração |
|---|---|---|---|
| `TESTE-TEC4-01-REFERENCIA-OMBRO-DIREITO-PM012-C17-4p8s.mp4` | `Ex1/PM_012-Camera17-30fps.mp4`, repetição 5 | `[891,1035)` | 4,800 s |
| `TESTE-TEC4-02-OBSERVADO-OMBRO-DIREITO-PM012-C17-4p8s.mp4` | mesma gravação, repetição 9 | `[2071,2215)` | 4,800 s |
| `TESTE-TEC4-03-PM039-C17-REP02.mp4` | `Ex1/PM_039-Camera17-30fps.mp4`, repetição 2 | `[225,370)` | 4,833 s |
| `TESTE-TEC4-04-PM039-C17-REP08.mp4` | mesma gravação PM_039, repetição 8 | `[1515,1660)` | 4,833 s |
| `TESTE-TEC4-05-PM012-C17-REP08.mp4` | gravação PM_012, repetição 8 | `[1936,2070)` | 4,467 s |

A documentação dos autores não explicita se a numeração começa em zero ou um. Estes cortes usam o interior comum das duas interpretações do intervalo anotado. Todos os trechos selecionados têm `extra_person_in_cam17=0` no CSV oficial. PM_012 foi filmado com `lights_on=0`; PM_039 com `lights_on=1`. Isso não garante aprovação em qualquer detector: mantenha os controles de qualidade do aplicativo.

### Pré-requisitos locais

Para uma instalação nova, use os requisitos do projeto no ambiente principal e os requisitos de vídeo em **outro** ambiente. Execute estas etapas manualmente apenas se os ambientes ainda não existirem:

```powershell
py -m venv venv
& './venv/Scripts/python.exe' -m pip install -r requirements.txt
py -m venv .venv-video
& './.venv-video/Scripts/python.exe' -m pip install -r requirements-video-worker.txt
```

Obtenha manualmente o [Pose Landmarker Lite oficial, versão fixa float16/1](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task) e salve em `models/pose-landmarker-lite/pose_landmarker_lite.task`. O SHA-256 esperado pelo código é `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`. Confira com `Get-FileHash` antes de usar. [Documentação oficial do modelo](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker). Não inclua os pesos no Git. Para dependências e modelo de áudio, siga [Áudio local](../docs/AUDIO_LOCAL.md).

O preparador de clipes também exige FFmpeg e FFprobe previamente instalados e disponíveis no PATH. Nenhum script deste kit instala dependências ou baixa pesos automaticamente.

### Configurar a referência no Windows

Na raiz do projeto, no **mesmo PowerShell que iniciará o Streamlit**, configure o arquivo 01 e inicie o app. Isso não altera as configurações persistentes do Windows:

```powershell
$env:FIAP_VIDEO_PYTHON = (Resolve-Path './.venv-video/Scripts/python.exe').Path
$env:FIAP_VIDEO_REFERENCE_PATH = (Resolve-Path './Datasets/video/TESTE-TEC4-01-REFERENCIA-OMBRO-DIREITO-PM012-C17-4p8s.mp4').Path
& './venv/Scripts/python.exe' -m streamlit run './app/app.py' --server.address 127.0.0.1 --server.maxUploadSize 24 --browser.gatherUsageStats false
```

O modelo de pose local e `FIAP_VIDEO_PYTHON` precisam continuar configurados como no ambiente de vídeo do projeto. O kit não instala dependências nem baixa pesos. No menu **Vídeo**, envie um dos arquivos observados 02–05 e clique em **Analisar vídeo**. O arquivo 01, enviado também como observado, é apenas controle de entrada idêntica, não validação independente. Alternativamente, o app aceita a referência padrão local em `models/video-reference/referencia-ombro-direito.mp4`, diretório ignorado pelo Git.

### Recriar a partir dos originais já baixados

Fonte: [registro oficial REHAB24-6 no Zenodo](https://zenodo.org/records/13305826). O registro oferece `videos.zip` de aproximadamente 2,7 GB; não é necessário baixá-lo de novo quando os MP4 já estão na máquina. Os dois originais aceitos e seus SHA-256 estão no manifesto. A receita **não faz download**, não acessa BIDMC, não modifica os originais e recusa fontes diferentes pelo hash.

Com Python, FFmpeg e FFprobe já disponíveis no PATH:

```powershell
$saida = Join-Path $env:TEMP ('fiap-midia-' + [guid]::NewGuid().ToString('N'))
python scripts/prepare_rehab_test_videos.py --pm012 'C:\dados\Ex1\PM_012-Camera17-30fps.mp4' --pm039 'C:\dados\Ex1\PM_039-Camera17-30fps.mp4' --output-dir $saida
```

Substitua apenas os caminhos dos dois originais. A pasta de saída deve ser nova e fora do repositório. Se só houver PM_012, omita `--pm039`: serão preparados 01, 02 e 05. O script usa `-nostdin -n`, confere tamanho/hash dos originais, limita a quantidade de frames, valida o MP4 resultante e escreve um recibo com hashes. Não sobrescreve arquivos, não abre modal e não executa Git. Também imprime a configuração da referência usando a pasta criada.

Os mesmos cortes podem resultar em SHA-256 diferentes com versões/builds do FFmpeg, configurações de threads ou encoder diferentes. Os hashes do manifesto identificam os cinco arquivos convertidos no Windows; uma conversão nova deve produzir o próprio recibo. Não altere limiares para forçar um resultado esperado.

## O que foi medido, e o que isso não prova

- **Áudio, código f1ba9f1:** `transcribe_local` real, faster-whisper 1.2.1, CPU/int8, modelo Tiny local na revisão `d90ca5fe260221311c53c58e660288d3deb8d356`. A fala sintética retornou as 23 palavras esperadas após normalizar pontuação/maiúsculas, sem edições nesse único arquivo. As duas negações foram preservadas. Tempo observado nesta execução: 1,987 s. Silêncio digital recusado antes da ASR. Zero tentativas de conexão Python registradas durante essa execução offline; isso não é captura de pacotes do sistema.
- **Vídeo 02 no Windows:** um intervalo sinalizado, de 1,0 a 2,4 s, usando 01 como referência.
- **Vídeo 04 no Windows:** cobertura de 96%; intervalos 0,4–1,6 s e 2,8–3,4 s, máximos observados aproximadamente 39,5° e 22,9°.
- **Ensaio independente Linux, mesmos originais/cortes, bytes de encoder diferentes:** PM_039 rep2 gerou dois intervalos e máximo 28,45°; PM_012 rep8 não gerou intervalo sustentado, com cobertura de 95,83%.

**Esses números são medições específicas, não resultados garantidos nem classificação de exercício correto/incorreto.** O autor marca PM_039 rep2 como correto, mas houve intervalos de diferença. PM_012 rep8 é marcado incorreto, mas não houve intervalo sustentado. O programa compara ângulos projetados no tempo; diferenças de pessoa, ritmo e fase do movimento importam. Não há normalização temporal nem demonstração de equivalência clínica. Os cinco clipes vêm de somente **duas gravações**, não de cinco fontes independentes. Para resultados exatos de bytes, ambiente e fonte, consulte os manifestos e preserve os limites do app.

## Créditos e licença dos vídeos

**REHAB24-6: A multi-modal dataset of physical rehabilitation exercises**, versão 1, 2024. Autores: Andrej Černek, Jan Sedmidubsky, Petra Budikova, Miriama Jánošová, Lukáš Katzer e Michal Procházka. [DOI 10.5281/zenodo.13305826](https://doi.org/10.5281/zenodo.13305826).

O [metadado oficial](https://zenodo.org/api/records/13305826) indica [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/). A descrição do registro limita o uso a pesquisa acadêmica ou de organizações sem fins lucrativos, não comercial, e solicita citar o artigo:

Černek, A.; Sedmidubsky, J.; Budikova, P. **REHAB24-6: Physical Therapy Dataset for Analyzing Pose Estimation Methods.** 17th International Conference on Similarity Search and Applications (SISAP), Springer, 2024.

As mudanças deste kit são cortes temporais, redimensionamento proporcional para 720p, recodificação H.264/yuv420p, normalização de timestamps para 30 fps e remoção de áudio. A receita registrada usa `-map_metadata -1` para desativar a cópia automática de metadados globais; isso não comprova ausência de todas as tags, pois o encoder pode escrever informações técnicas. O manifesto identifica fonte e intervalo por arquivo. Preserve créditos, DOI, licença e indicação das alterações ao compartilhar estes recortes, mantenha uso acadêmico/não comercial e não sugira endosso dos autores. A licença de mídia não se transforma na licença do código deste repositório. O registro declara consentimento dos participantes para divulgação pública dos dados; isso não é permissão genérica para outros usos de suas imagens.

## Créditos de áudio e fonte opcional

A fala do kit foi sintetizada localmente a partir de texto fictício, usando o filtro **Flite** do FFmpeg e sua voz integrada inglesa **SLT**. Não é consulta real, gravação de paciente, nem imitação solicitada de uma pessoa específica. Flite e a definição SLT usam licença permissiva CMU: [licença oficial](https://github.com/festvox/flite/blob/master/COPYING), [definição/licença da voz](https://github.com/festvox/flite/blob/master/lang/cmu_us_slt/cmu_us_slt.c). O [aviso integral](licenses/FLITE-COPYING.txt) e o [aviso específico SLT](licenses/FLITE-SLT-NOTICE.txt) acompanham o kit; não há redistribuição de biblioteca, modelo de voz ou executável. Autores institucionais: Language Technologies Institute, Carnegie Mellon University; SLT/Flite: Alan W. Black e colaboradores. Não implica endosso. O controle silencioso contém apenas amostras PCM zeradas.

Para uma consulta encenada mais longa em inglês, o **PriMock57** disponibiliza médicos e atores representando casos fictícios: [fonte oficial](https://github.com/babylonhealth/primock57), [licença CC BY 4.0](https://github.com/babylonhealth/primock57/blob/main/LICENSE.md), [artigo dos autores](https://arxiv.org/abs/2204.00333). Exemplo: [day1_consultation01_patient.wav](https://media.githubusercontent.com/media/babylonhealth/primock57/main/audio/day1_consultation01_patient.wav), 14.651.564 bytes, aproximadamente 7 min 38 s, WAV PCM16 mono, 16 kHz; [transcrição com tempos](https://raw.githubusercontent.com/babylonhealth/primock57/main/transcripts/day1_consultation01_patient.TextGrid). É o canal do paciente, separado do médico. O arquivo não é incluído neste repositório para evitar duplicar desnecessariamente vozes de atores. Credite Papadopoulos Korfiatis, Moramarco, Sarac e Savkov (2022) se o usar e registre qualquer corte.

### Amostra limpa em português

Para verificar Português, grave uma encenação consentida de 10–20 s com um adulto, sem nome, prontuário ou dado real. Exemplo de texto fictício: “Este é um teste fictício. Ontem eu estava cansado. Não tive dor no peito nem falta de ar. Hoje estou melhor.” Guarde a transcrição esperada separadamente. Não publique a voz sem autorização específica de quem gravou. Converta uma cópia, sem sobrescrever:

```powershell
ffmpeg -nostdin -n -i 'minha-encenacao.m4a' -vn -ac 1 -ar 16000 -c:a pcm_s16le -map_metadata -1 'teste-pt-16k.wav'
```

Português não foi validado pela amostra inglesa deste kit. A fala sintética curta não mede a acurácia geral do reconhecedor, e “sentimento” lexical não é estado emocional ou clínico. Para preparação e limites do motor, veja [Áudio local](../docs/AUDIO_LOCAL.md).

## Reproduzir os CSVs e as verificações

```powershell
python scripts/generate_test_csvs.py --output-dir "$env:TEMP\fiap-csv-novo"
python -m unittest discover -s tests -p test_dataset_kit.py -v
```

Escolha uma pasta ainda inexistente para o gerador. Ele usa o esquema/parser/analisador do próprio app, verifica 120 leituras, 25 resultados de aquecimento e 0/4 alertas e recusa sobrescrita. Os testes determinísticos não executam ASR nem baixam mídia/modelos. O smoke real de áudio é uma evidência separada, registrada no [manifesto de áudio](audio/manifest.json).
