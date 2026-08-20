# Amostras externas para avaliacao de generalizacao

As quatro imagens JPEG deste diretorio sao capturas de tela redimensionadas de imagens do dataset **UAV-OBB: An Aerial Urban Vehicle Dataset with Oriented Bounding Boxes for Remote Sensing Object Detection in Smart Cities**.

Fonte: [Mendeley Data, versao 4](https://data.mendeley.com/datasets/6snrjwcpkh/4) — DOI: [10.17632/6snrjwcpkh.4](https://doi.org/10.17632/6snrjwcpkh.4).

Autores do dataset: Israr Ahmad, Shang Fengjun, Kiran Bibi e Muhammad Slaman Pathan.

Licenca: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). As imagens deste diretorio foram modificadas em relacao aos arquivos originais: foram abertas individualmente, capturadas em tela e salvas em resolucao menor. Esta adaptacao nao implica endosso dos autores ao projeto.

As tres imagens diurnas sao usadas exclusivamente para uma avaliacao externa quantitativa de generalizacao. A imagem noturna permanece como exemplo qualitativo de cenario adverso e nao compoe as metricas agregadas. Nenhuma delas participa de treinamento, fine-tuning ou escolha de modelo, confidence threshold ou configuracao de SAHI. Como as capturas alteram a geometria dos arquivos originais, os rotulos OBB oficiais nao sao reutilizados; o projeto usa caixas de veiculos revisadas manualmente em `data/annotations/external/` antes de calcular metricas.
