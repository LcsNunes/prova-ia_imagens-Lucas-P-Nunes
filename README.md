# Detecção de Veículos com Drone

Prova prática de visão computacional para detectar e contar veículos em uma imagem aérea capturada por drone. A entrega prioriza a detecção de veículos; a malha viária é um recurso visual secundário, apresentado por segmentação semântica e avaliado de forma qualitativa.

## Visão geral

A aplicação recebe uma imagem, valida seu conteúdo, executa detecção síncrona, segmenta a camada de vias e devolve as visualizações de veículos, vias e composição. O detector final foi escolhido por experimentos controlados, não por suposição:

| Configuração final | Valor |
| --- | --- |
| Modelo | `yolo11n-obb.pt` treinado em DOTA v1 |
| Ontologia da aplicação | `vehicle` |
| Confidence | `0.15` |
| Inferência | SAHI, fatias de 512 px e overlap de 20% |
| Dispositivo observado | NVIDIA GeForce RTX 5060 8 GB, CUDA 12.8 |
| Resultado da configuração final | F1 `0.895`, recall `0.855`, erro absoluto de contagem `5` |

O resultado acima é válido para a imagem da prova e para o ground truth revisado deste repositório. Ele não deve ser interpretado como medida de generalização para novas cidades, câmeras ou altitudes.

## Problema e estratégia

A imagem original, extraída do PDF da prova, mede 2048 x 1534 px. Veículos ocupam poucos pixels e podem estar orientados de formas variadas. Por isso, foram comparados:

1. YOLO11 generalista pré-treinado em COCO;
2. YOLO11-OBB especializado em imagens aéreas/DOTA;
3. inferência normal e tiled inference com SAHI.

As classes COCO `car`, `motorcycle`, `bus` e `truck`, e as classes aéreas `small vehicle` e `large vehicle`, são normalizadas para o único conceito de negócio `vehicle`. A avaliação mede se o veículo foi localizado, e não se carro, ônibus ou caminhão foram classificados com a classe fina correta.

## Arquitetura

```text
browser -> FastAPI -> validação de imagem -> pipeline síncrono
                                             |-> detector YOLO / SAHI
                                             |-> Mask2Former para vias
                                             `-> overlays JPEG + metadados
```

```text
src/
├── api/            # FastAPI, contrato e validação de upload
├── detection/      # adaptadores YOLO, SAHI, ontologia e deduplicação
├── evaluation/     # ground truth, matching IoU, métricas e benchmark
├── roads/          # destaque de vias com OpenCV e Mask2Former
├── visualization/  # caixas e imagens RGB
├── web/            # HTML, CSS e JavaScript servidos pelo FastAPI
├── config.py
├── logging_config.py
└── pipeline.py
```

O notebook é o laboratório de decisão; o código em `src/` representa a solução escolhida.

## Instalação e execução local

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

Abra `http://127.0.0.1:8000`. A página permite upload, mostra a imagem original, as detecções, a contagem, o modelo, SAHI, confidence, tempo e a camada secundária de via.

O padrão `DEVICE=auto` usa CUDA quando o PyTorch a encontra. Para forçar CPU, use `DEVICE=cpu`; para forçar a RTX, use `DEVICE=cuda:0`.

```powershell
$env:DEVICE = "cuda:0"
$env:LOG_LEVEL = "INFO"
python -m uvicorn src.api.app:app --reload
```

## Modelos

Pesos não são versionados. O manifesto [`configs/model_registry.yaml`](configs/model_registry.yaml) registra nome, release, URL e SHA-256; o script valida o digest quando o peso é obtido ou reutilizado.

```powershell
# Baseline e variantes usadas nos experimentos
python scripts/download_models.py --model coco_yolo11n
python scripts/download_models.py --model aerial_yolo11n_obb
python scripts/download_models.py --model aerial_yolo11s_obb
python scripts/download_models.py --model aerial_yolo11m_obb
```

## API

| Método e rota | Uso |
| --- | --- |
| `GET /api/health` | verifica se o pipeline foi carregado |
| `POST /api/analyses` | recebe um campo multipart `image` e devolve a análise |

`POST /api/analyses` aceita JPEG, PNG e WEBP após validar os bytes reais do arquivo, com limite configurável de tamanho e pixels. A resposta possui `vehicle_count`, `model_name`, `sahi_enabled`, `confidence`, `inference_ms`, `detections_image` e `roads_image`. As duas imagens são JPEGs em data URL para a interface usar sem gravar uploads no disco.

Exemplo:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/analyses -F "image=@data/raw/drone_scene.jpg"
```

Erros de upload retornam `422` com um código e mensagem segura; indisponibilidade do modelo retorna `503`. A inferência é propositalmente síncrona, sem filas, Redis ou workers.

Atualização da composição visual: a resposta também inclui os tempos separados de veículos e vias, além da imagem combinada. A interface exibe caixas ciano para veículos e camada laranja para vias; as caixas são desenhadas por último para continuarem legíveis.

## Ground truth e métricas

[`data/annotations/ground_truth.json`](data/annotations/ground_truth.json) contém 55 caixas, somente com a categoria `vehicle`, além de dimensões e SHA-256 da imagem. O fluxo no notebook permite criar ou revisar caixas manualmente com `RectangleSelector`.

As visualizações de benchmark usam renderização estática por padrão, evitando dependência do widget JavaScript do VS Code. A célula de anotação manual permanece interativa e só deve ser executada quando for necessário editar o ground truth.

O arquivo atual foi iniciado por pré-anotações do modelo aéreo e revisado manualmente, incluindo uma expansão para veículos parcialmente ocluídos. Essa escolha acelera a prova, mas é uma fonte potencial de viés de seleção; por transparência, ela consta no próprio JSON e deve ser substituída por anotação independente em uma avaliação de produto.

Predições e ground truth são pareados de forma gulosa por score quando `IoU >= 0.50`. Além de `Absolute Count Error`, o projeto calcula TP, FP, FN, precision, recall e F1, impedindo que falsos positivos e falsos negativos se cancelem apenas na contagem.

## Experimentos e resultados

Todos os resultados detalhados e metadados do ambiente estão em [`outputs/metrics/benchmark_results.json`](outputs/metrics/benchmark_results.json). As medidas usaram a mesma imagem, o mesmo ground truth, `imgsz=1024`, FP32, RTX 5060, um warm-up e três repetições cronometradas; a tabela reporta a mediana. Arquivos gerados grandes continuam ignorados pelo Git.

### Nano x Small x Medium

Nesta etapa, as demais variáveis foram mantidas fixas e SAHI ficou desligado.

| Modelo aéreo | TP | FP | FN | Precision | Recall | F1 | Count error | Mediana | Pico GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n-OBB | 44 | 3 | 11 | **0.936** | 0.800 | **0.863** | 8 | **50.3 ms** | **88.3 MB** |
| YOLO11s-OBB | 44 | 4 | 11 | 0.917 | 0.800 | 0.854 | **7** | 52.3 ms | 161.5 MB |
| YOLO11m-OBB | 44 | 4 | 11 | 0.917 | 0.800 | 0.854 | **7** | 55.5 ms | 266.6 MB |

O Nano obteve o maior F1 e a menor latência/memória. Small e Medium tiveram erro de contagem uma unidade menor, mas F1 inferior e custo maior; por isso, não foram escolhidos, pois não apresentaram evidências de ganho em imagens independentes.

### Benchmark principal 2 x 2

| Domínio / inferência | TP | FP | FN | Precision | Recall | F1 | Count error | Mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO normal | 0 | 1 | 55 | 0.000 | 0.000 | 0.000 | 54 | **41.2 ms** |
| COCO + SAHI 512 | 0 | 4 | 55 | 0.000 | 0.000 | 0.000 | 51 | 796.8 ms |
| Aerial/DOTA normal | 44 | 3 | 11 | 0.936 | 0.800 | 0.863 | 8 | 52.8 ms |
| Aerial/DOTA + SAHI 512 | **46** | **1** | **9** | **0.979** | **0.836** | **0.902** | 8 | 888.1 ms |

A especialização aérea foi decisiva nesta imagem. SAHI aumentou F1 de 0.863 para 0.902 e reduziu FNs de 11 para 9, com custo material de latência. Os nove FNs restantes correspondem aos veículos parcialmente ocluídos ou cortados que passaram a fazer parte do ground truth revisado.

### Confidence e tamanho da fatia

Para o Nano aéreo com SAHI:

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

Escolhi `confidence=0.15` e fatia de 512 para a configuração final porque recupera um TP e reduz o erro de contagem de 8 para 5, a métrica mais próxima do objetivo de estimar o total de veículos. O custo é de dois FPs adicionais e uma pequena queda de F1 (de 0.902 para 0.895) frente a `0.25`; essa troca fica registrada explicitamente. Isso não é um grid search nem uma regra universal: foi uma verificação pequena para entender o comportamento da cena e deverá ser revalidada em imagens independentes.

### Testes direcionados para oclusão

Mantendo modelo, confidence `0.25`, fatia, imagem e ground truth, dois testes alteraram apenas uma variável por vez em relação ao controle de `imgsz=1024` e overlap de 20%.

| Configuração | TP | FP | FN | F1 | Count error | Mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Controle (1024, 20%) | **46** | **1** | **9** | **0.902** | **8** | **839.1 ms** |
| `imgsz=1280`, 20% | 37 | 2 | 18 | 0.787 | 16 | 940.5 ms |
| 1024, overlap 30% | 44 | 3 | 11 | 0.863 | 8 | 1004.3 ms |

Nenhuma das duas hipóteses melhorou os veículos ocluídos: ambas reduziram F1 e aumentaram a latência. Portanto, `imgsz=1024` e overlap de 20% foram mantidos.

### Avaliação externa de generalização

Três imagens diurnas externas foram usadas somente depois de congelar a configuração final. Elas não participaram de treinamento, fine-tuning ou escolha de modelo, confidence e SAHI. As imagens são capturas de tela redimensionadas do dataset [UAV-OBB](https://data.mendeley.com/datasets/6snrjwcpkh/4), disponibilizado sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); a procedência e as alterações feitas nos arquivos estão registradas em [`data/external/README.md`](data/external/README.md).

Como as capturas alteram a geometria das imagens originais, os rótulos OBB do dataset não foram reutilizados. Cada imagem recebeu pré-anotações e uma revisão humana antes da avaliação. A imagem noturna permaneceu como teste qualitativo de cenário adverso e não compõe as métricas agregadas.

| Imagem | Ground truth | Predições | TP | FP | FN | Precision | Recall | F1 | Count error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `testeviario1.jpg` | 20 | 19 | 19 | 0 | 1 | 1.000 | 0.950 | 0.974 | 1 |
| `testeviario2.jpg` | 25 | 25 | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 |
| `testeviario3.jpg` | 50 | 50 | 49 | 1 | 1 | 0.980 | 0.980 | 0.980 | 0 |

No agregado: 93 TP, 1 FP e 2 FN, com Precision micro `0.989`, Recall micro `0.979` e F1 micro `0.984`. O terceiro caso reforça por que erro de contagem não é suficiente: um FP e um FN se compensaram na contagem, embora a detecção não tenha sido perfeita.

## Malha viária

### Integração final

A interface usa Mask2Former Satellite, obtido em uma revisão fixa e mantido no cache local do projeto. A classe de via foi auditada visualmente no notebook; como não há máscaras de ground truth de estrada, este resultado é qualitativo e não deve ser reportado como IoU, precision ou recall.

YOLO com SAHI e Mask2Former executam sequencialmente na mesma requisição. Eles não compartilham pesos: o primeiro localiza veículos e o segundo produz a máscara de vias. A composição aplica primeiro a camada laranja de vias e desenha as caixas ciano por cima. O modo HSV abaixo permanece como baseline e fallback configurável.

[`src/roads/opencv_road.py`](src/roads/opencv_road.py) usa HSV, threshold, fechamento/abertura morfológica, filtros por componentes conectados e overlay semitransparente. Ela indica uma **região provável de pavimento/via**, não uma segmentação semântica de estrada.

A limitação é intencional: tempo e profundidade técnica foram concentrados na detecção de veículos. Iluminação, sombras, material da pista, telhados com cor semelhante, clima e câmeras diferentes podem degradar o resultado. Uma próxima versão usaria segmentação treinada e imagens variadas para avaliar IoU de via.

## Testes e qualidade

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Os testes cobrem ontologia, IoU e matching, métricas, deduplicação, ground truth, downloads com checksum, OpenCV, pipeline, validação de imagem e um smoke test da API. O `pyproject.toml` exige ao menos 80% de cobertura e o pipeline de CI executa lint, formatação, pytest com coverage e Docker build.

## Docker

O projeto usa um único container FastAPI. Primeiro, baixe o peso final para o volume local e depois suba a aplicação:

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

Sem GPU exposta ao container, `DEVICE=auto` faz fallback para CPU. O compose não obriga GPU, o que preserva a demonstração em máquinas sem NVIDIA.

## CI

O workflow [`ci.yml`](.github/workflows/ci.yml) é executado em push para `develop` e `main`, e em pull request para `main`. Ele faz checkout, instala Python 3.10 e dependências, roda Ruff, pytest/coverage e build da imagem Docker.

## Limitações e próximos passos

- A seleção inicial da configuração foi baseada na imagem da prova; as três imagens externas ampliam a verificação, mas ainda não representam uma distribuição completa de cidades, altitudes e câmeras.
- Os ground truths tiveram pré-anotação do modelo aéreo e revisão manual; eles são referências de estudo de caso, não um teste independente anotado do zero.
- OBB é comparado como bounding box alinhada aos eixos para manter uma única métrica de matching; uma evolução pode avaliar IoU orientado.
- A inferência SAHI melhora recall, mas aumenta a latência por imagem.
- A camada de vias usa um modelo pré-treinado, mas ainda não possui ground truth independente; portanto, sua avaliação é visual.
- Próximas evoluções: ground truth independente para novas imagens, conjunto separado de validação/teste, VisDrone ou fine-tuning leve se houver dados, avaliação de OBB e, somente depois, arquitetura AWS, filas e monitoramento.

## Uso de IA

Este repositório registra o uso de Codex/IA como apoio ao desenvolvimento do código, aos testes e à documentação. As decisões técnicas e as escolhas arquiteturais foram feitas pelo desenvolvedor; os experimentos foram executados localmente na RTX 5060, e os resultados aqui listados foram registrados a partir dessas execuções. O código e as anotações foram revisados e testados pelo desenvolvedor.
