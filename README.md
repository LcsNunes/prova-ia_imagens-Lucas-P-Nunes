# Drone vehicle inspection

Prova pratica de visao computacional para detectar e contar veiculos em uma imagem aerea capturada por drone. A entrega prioriza a deteccao de veiculos; a malha viaria e um recurso visual secundario, apresentado por segmentacao semantica e avaliado de forma qualitativa.

## Visao geral

A aplicacao recebe uma imagem, valida seu conteudo, executa deteccao sincrona, segmenta a camada de vias e devolve as visualizacoes de veiculos, vias e composicao. O detector final foi escolhido por experimentos controlados, nao por suposicao:

| Configuracao final | Valor |
| --- | --- |
| Modelo | `yolo11n-obb.pt` treinado em DOTA v1 |
| Ontologia da aplicacao | `vehicle` |
| Confidence | `0.15` |
| Inferencia | SAHI, fatias de 512 px e overlap de 20% |
| Dispositivo observado | NVIDIA GeForce RTX 5060 8 GB, CUDA 12.8 |
| Resultado da configuracao final | F1 `0.895`, recall `0.855`, erro absoluto de contagem `5` |

O resultado acima e valido para a imagem da prova e para o ground truth revisado deste repositorio. Ele nao deve ser interpretado como medida de generalizacao para novas cidades, cameras ou altitudes.

## Problema e estrategia

A imagem original, extraida do PDF da prova, mede 2048 x 1534 px. Veiculos ocupam poucos pixels e podem estar orientados de formas variadas. Por isso foram comparados:

1. YOLO11 generalista pre-treinado em COCO;
2. YOLO11-OBB especializado em imagens aereas/DOTA;
3. inferencia normal e tiled inference com SAHI.

As classes COCO `car`, `motorcycle`, `bus` e `truck`, e as classes aereas `small vehicle` e `large vehicle`, sao normalizadas para o unico conceito de negocio `vehicle`. A avaliacao mede se o veiculo foi localizado, e nao se carro, onibus ou caminhao foram classificados com a classe fina correta.

## Arquitetura

```text
browser -> FastAPI -> validacao de imagem -> pipeline sincrono
                                             |-> detector YOLO / SAHI
                                             |-> Mask2Former para vias
                                             `-> overlays JPEG + metadados
```

```text
src/
├── api/            # FastAPI, contrato e validacao de upload
├── detection/      # adaptadores YOLO, SAHI, ontologia e deduplicacao
├── evaluation/     # ground truth, matching IoU, metricas e benchmark
├── roads/          # destaque deliberadamente simples com OpenCV
├── visualization/  # caixas e imagens RGB
├── web/            # HTML, CSS e JavaScript servidos pelo FastAPI
├── config.py
├── logging_config.py
└── pipeline.py
```

O notebook e o laboratorio de decisao; o codigo em `src/` representa a solucao escolhida.

## Instalacao e execucao local

Requer Python 3.10. Em Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_models.py --model aerial_yolo11n_obb
python scripts/download_models.py --road-model mask2former_satellite
python -m uvicorn src.api.app:app --reload
```

Abra `http://127.0.0.1:8000`. A pagina permite upload, mostra a imagem original, as deteccoes, a contagem, modelo, SAHI, confidence, tempo e a camada secundaria de via.

O padrao `DEVICE=auto` usa CUDA quando o PyTorch a encontra. Para forcar CPU, use `DEVICE=cpu`; para forcar a RTX, use `DEVICE=cuda:0`.

```powershell
$env:DEVICE = "cuda:0"
$env:LOG_LEVEL = "INFO"
python -m uvicorn src.api.app:app --reload
```

## Modelos

Pesos nao sao versionados. O manifesto [`configs/model_registry.yaml`](configs/model_registry.yaml) registra nome, release, URL e SHA-256; o script valida o digest quando o peso e obtido ou reutilizado.

```powershell
# Baseline e variantes usadas nos experimentos
python scripts/download_models.py --model coco_yolo11n
python scripts/download_models.py --model aerial_yolo11n_obb
python scripts/download_models.py --model aerial_yolo11s_obb
python scripts/download_models.py --model aerial_yolo11m_obb
```

## API

| Metodo e rota | Uso |
| --- | --- |
| `GET /api/health` | verifica se o pipeline foi carregado |
| `POST /api/analyses` | recebe um campo multipart `image` e devolve a analise |

`POST /api/analyses` aceita JPEG, PNG e WEBP apos validar os bytes reais do arquivo, com limite configuravel de tamanho e pixels. A resposta possui `vehicle_count`, `model_name`, `sahi_enabled`, `confidence`, `inference_ms`, `detections_image` e `roads_image`. As duas imagens sao JPEGs em data URL para a interface usar sem gravar uploads no disco.

Exemplo:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/analyses -F "image=@data/raw/drone_scene.jpg"
```

Erros de upload retornam `422` com um codigo e mensagem segura; indisponibilidade do modelo retorna `503`. A inferencia e propositalmente sincrona, sem filas, Redis ou workers.

Atualizacao da composicao visual: a resposta tambem inclui os tempos separados de veiculos e vias, alem da imagem combinada. A interface exibe caixas ciano para veiculos e camada laranja para vias; as caixas sao desenhadas por ultimo para continuarem legiveis.

## Ground truth e metricas

[`data/annotations/ground_truth.json`](data/annotations/ground_truth.json) contem 55 caixas, somente com a categoria `vehicle`, alem de dimensoes e SHA-256 da imagem. O fluxo no notebook permite criar ou revisar caixas manualmente com `RectangleSelector`.

As visualizacoes de benchmark usam renderizacao estatica por padrao, evitando dependencia do widget JavaScript do VS Code. A celula de anotacao manual permanece interativa e so deve ser executada quando for necessario editar o ground truth.

O arquivo atual foi iniciado por pre-anotacoes do modelo aereo e revisado manualmente, incluindo uma expansao para veiculos parcialmente ocluidos. Essa escolha acelera a prova, mas e uma fonte potencial de vies de selecao; por transparencia, ela consta no proprio JSON e deve ser substituida por anotacao independente em uma avaliacao de produto.

Predicoes e ground truth sao pareados de forma gulosa por score quando `IoU >= 0.50`. Alem de `Absolute Count Error`, o projeto calcula TP, FP, FN, precision, recall e F1, impedindo que falsos positivos e falsos negativos se cancelem apenas na contagem.

## Experimentos e resultados

Todos os resultados detalhados e metadados do ambiente estao em [`outputs/metrics/benchmark_results.json`](outputs/metrics/benchmark_results.json). As medidas usaram a mesma imagem, o mesmo ground truth, `imgsz=1024`, FP32, RTX 5060, um warm-up e tres repeticoes cronometradas; a tabela reporta a mediana. Arquivos gerados grandes continuam ignorados pelo Git.

### Nano x Small x Medium

Nesta etapa as demais variaveis foram mantidas fixas e SAHI ficou desligado.

| Modelo aereo | TP | FP | FN | Precision | Recall | F1 | Count error | Mediana | Pico GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n-OBB | 44 | 3 | 11 | **0.936** | 0.800 | **0.863** | 8 | **50.3 ms** | **88.3 MB** |
| YOLO11s-OBB | 44 | 4 | 11 | 0.917 | 0.800 | 0.854 | **7** | 52.3 ms | 161.5 MB |
| YOLO11m-OBB | 44 | 4 | 11 | 0.917 | 0.800 | 0.854 | **7** | 55.5 ms | 266.6 MB |

O Nano obteve o maior F1 e a menor latencia/memoria. Small e Medium tiveram erro de contagem uma unidade menor, mas F1 inferior e custo maior; por isso nao foram escolhidos sem evidencias em imagens independentes.

### Benchmark principal 2 x 2

| Dominio / inferencia | TP | FP | FN | Precision | Recall | F1 | Count error | Mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO normal | 0 | 1 | 55 | 0.000 | 0.000 | 0.000 | 54 | **41.2 ms** |
| COCO + SAHI 512 | 0 | 4 | 55 | 0.000 | 0.000 | 0.000 | 51 | 796.8 ms |
| Aerial/DOTA normal | 44 | 3 | 11 | 0.936 | 0.800 | 0.863 | 8 | 52.8 ms |
| Aerial/DOTA + SAHI 512 | **46** | **1** | **9** | **0.979** | **0.836** | **0.902** | 8 | 888.1 ms |

A especializacao aerea foi decisiva nesta imagem. SAHI aumentou F1 de 0.863 para 0.902 e reduziu FNs de 11 para 9, com custo material de latencia. Os nove FNs restantes correspondem aos veiculos parcialmente ocluidos ou cortados que passaram a fazer parte do ground truth revisado.

### Confidence e tamanho da fatia

Para o Nano aereo com SAHI:

| Confidence | TP | FP | FN | F1 | Count error |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.15** | **47** | 3 | **8** | 0.895 | **5** |
| 0.25 | 46 | **1** | 9 | **0.902** | 8 |
| 0.35 | 42 | **1** | 13 | 0.857 | 12 |
| 0.50 | 39 | 0 | 16 | 0.830 | 16 |

| Fatias | TP | FP | FN | F1 | Count error | Mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **512** | **46** | **1** | **9** | **0.902** | 8 | 884.2 ms |
| 640 | 44 | 3 | 11 | 0.863 | 8 | **652.5 ms** |

Escolhi `confidence=0.15` e fatia 512 para a configuracao final porque recupera um TP e reduz o erro de contagem de 8 para 5, a metrica mais proxima do objetivo de estimar o total de veiculos. O custo e dois FPs adicionais e queda pequena de F1 (0.902 para 0.895) frente a `0.25`; essa troca fica registrada explicitamente. Isso nao e um grid search nem uma regra universal: foi uma verificacao pequena para entender o comportamento da cena e devera ser revalidada em imagens independentes.

### Testes direcionados para oclusao

Mantendo modelo, confidence `0.25`, fatia, imagem e ground truth, dois testes alteraram apenas uma variavel por vez em relacao ao controle de `imgsz=1024` e overlap de 20%.

| Configuracao | TP | FP | FN | F1 | Count error | Mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controle (1024, 20%) | **46** | **1** | **9** | **0.902** | **8** | **839.1 ms** |
| `imgsz=1280`, 20% | 37 | 2 | 18 | 0.787 | 16 | 940.5 ms |
| 1024, overlap 30% | 44 | 3 | 11 | 0.863 | 8 | 1004.3 ms |

Nenhuma das duas hipoteses melhorou os veiculos ocluidos: ambas reduziram F1 e aumentaram a latencia. Portanto, `imgsz=1024` e overlap de 20% foram mantidos.

## Malha viaria

### Integracao final

A interface usa Mask2Former Satellite, obtido em uma revisao fixa e mantido no cache local do projeto. A classe de via foi auditada visualmente no notebook; como nao ha mascaras ground truth de estrada, este resultado e qualitativo e nao deve ser reportado como IoU, precision ou recall.

YOLO com SAHI e Mask2Former executam sequencialmente na mesma requisicao. Eles nao compartilham pesos: o primeiro localiza veiculos e o segundo produz a mascara de vias. A composicao aplica primeiro a camada laranja de vias e desenha as caixas ciano por cima. O modo HSV abaixo permanece como baseline e fallback configuravel.

[`src/roads/opencv_road.py`](src/roads/opencv_road.py) usa HSV, threshold, fechamento/abertura morfologica, filtros por componentes conectados e overlay semitransparente. Ela indica uma **regiao provavel de pavimento/via**, nao uma segmentacao semantica de estrada.

A limitacao e intencional: tempo e profundidade tecnica foram concentrados na deteccao de veiculos. Iluminacao, sombras, material da pista, telhados com cor semelhante, clima e cameras diferentes podem degradar o resultado. Uma proxima versao usaria segmentacao treinada e imagens variadas para avaliar IoU de via.

## Testes e qualidade

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Os testes cobrem ontologia, IoU e matching, metricas, deduplicacao, ground truth, downloads com checksum, OpenCV, pipeline, validacao de imagem e um smoke test da API. O `pyproject.toml` exige ao menos 80% de cobertura e o pipeline de CI executa lint, formatacao, pytest com coverage e Docker build.

## Docker

O projeto usa um unico container FastAPI. Primeiro baixe o peso final para o volume local e depois suba a aplicacao:

```powershell
docker compose build
docker compose run --rm app python scripts/download_models.py --model aerial_yolo11n_obb
docker compose run --rm app python scripts/download_models.py --road-model mask2former_satellite
docker compose up
```

Abra `http://127.0.0.1:8000`. Para expor a GPU para o Docker Desktop com NVIDIA Container Toolkit configurado:

```powershell
docker compose run --rm --gpus all --service-ports -e DEVICE=cuda:0 app
```

Sem GPU exposta ao container, `DEVICE=auto` faz fallback para CPU. O compose nao obriga GPU, o que preserva a demonstracao em maquinas sem NVIDIA.

## CI

O workflow [`ci.yml`](.github/workflows/ci.yml) e executado em push para `develop` e `main`, e em pull request para `main`. Ele faz checkout, instala Python 3.10 e dependencias, roda Ruff, pytest/coverage e build da imagem Docker.

## Limitacoes e proximos passos

- Ha apenas uma imagem de avaliacao; resultados nao representam uma distribuicao completa.
- O ground truth atual teve pre-anotacao do modelo aereo e expansao manual apos auditoria visual; ele e uma referencia de estudo de caso, nao um teste independente.
- OBB e comparado como bounding box alinhada aos eixos para manter uma unica metrica de matching; uma evolucao pode avaliar IoU orientado.
- A inferencia SAHI melhora recall, mas aumenta a latencia por imagem.
- A camada de vias usa um modelo pre-treinado, mas ainda nao possui ground truth independente; portanto, sua avaliacao e visual.
- Proximas evolucoes: ground truth independente para novas imagens, conjunto separado de validacao/teste, VisDrone ou fine-tuning leve se houver dados, avaliacao de OBB, e somente depois arquitetura AWS, filas e monitoramento.

## Uso de IA

Este repositorio registra uso de Codex/IA como apoio para scaffolding, sugestoes de implementacao, testes e documentacao. As decisoes tecnicas foram revisadas pelo desenvolvedor; os experimentos foram executados localmente na RTX 5060 e os resultados aqui listados foram registrados a partir dessas execucoes. O codigo e as anotacoes devem ser revisados manualmente pelo desenvolvedor antes da entrega final.

