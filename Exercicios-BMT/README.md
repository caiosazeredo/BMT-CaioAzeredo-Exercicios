Visão Geral
O presente sistema é uma ferramenta projetada para facilitar consultas ao banco de dados CysticFibrosis2. Embora seja voltado especificamente para essa base, o sistema possui módulos modificáveis, como o queryProcessor e o indexer, permitindo que seja ajustado para operar com diferentes conjuntos de dados. Aqui você encontrará instruções para configuração e uso do sistema.

Preparação
Instalando Requisitos
Primeiramente, precisamos instalar as bibliotecas necessárias para o funcionamento do sistema. Utilize Python 3.8.10 e siga as instruções abaixo:

bash
Copy code
$ python3 -m venv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
Preparando o Conjunto de Dados
Baixe o conjunto de dados do CysticFibrosis2 e salve em um local adequado.

Ajustes no Sistema
Para a configuração do sistema, dispomos de quatro arquivos principais localizados no diretório do projeto, cujos detalhes estão descritos a seguir.

PC.CFG: Define o caminho para os arquivos de consultas, resultados esperados e consulta pré-processada.

GLI.CFG: Define o caminho dos documentos para executar consultas e o local para salvar a lista invertida.

INDEX.CFG: Configura o local de leitura da lista invertida e onde armazenar o modelo criado.

BUSCA.CFG: Configura onde localizar o modelo e as consultas pré-processadas, além do local para armazenar os resultados das consultas.

AVALIA.CFG: Especifica quais arquivos de resultados utilizar para as medidas de avaliação, e onde essas avaliações serão armazenadas.

Utilização do Sistema
Execução
Para iniciar o sistema, execute o script main.py de acordo com o modo desejado:

Modo de consulta:

bash
Copy code
$ python3 main.py -m search
Modo de avaliação:

bash
Copy code
$ python3 main.py -m eval
Interpretando os Resultados
Os resultados da consulta serão exibidos em uma tabela, onde cada linha representa uma consulta realizada e as colunas indicam a consulta, a lista de documentos recuperados e a pontuação obtida.

No modo de avaliação, além da tabela de resultados, serão gerados gráficos de desempenho que permitem uma análise visual da eficácia das consultas realizadas. Esses gráficos incluem curvas de precisão-recall, ROC, entre outros, e são armazenados no diretório especificado no arquivo de configuração AVALIA.CFG.

Assegure-se de que os arquivos de configuração estão preenchidos corretamente antes de executar o sistema, pois qualquer erro pode levar a resultados imprecisos ou ao não funcionamento do sistema.