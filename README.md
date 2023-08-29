# programaria-sprint-dados

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge)

![Badge code size](https://img.shields.io/github/languages/code-size/fab-souza/programaria-sprint-dados)

| :placard: Vitrine.Dev |    |
| -------------  | --- |
| :sparkles: Nome        | **PrograMaria Sprint Dados: Ampliando fronteiras**
| :label: Tecnologias | python
| :rocket: URL         | Notebook da [Parte 1](https://www.kaggle.com/fabianadesouza/programaria-sprint-dados-parte-1), [Parte 2](https://www.kaggle.com/code/fabianadesouza/programaria-sprint-dados-parte-2) e Parte 3
| :fire: Desafio     | 

![capa](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/7ac8de8b-4d68-4966-ad83-96df8d9dc72c)

A PrograMaria é uma startup de impacto social que tem como missão empoderar mulheres e promover diversidade e inclusão no mundo da programação e da tecnologia, por meio de oficinas, eventos e cursos de formação técnica. Desde 2020, ano que comecei a pesquisar mais sobre dados, a PrograMaria foi uma das comunidades femininas que encontrei e me encantei. Tive a oportunidade de participar de alguns eventos online e foi onde encontrei uma linda rede de apoio, pois foi onde conheci muita gente maravilhosa, tanto palestrantes quanto outras participantes que estavam em uma situação parecida com a minha, que já estavam decididas a migrar de carreira ou que estavam no início da carreira em tecnologia e queriam apoiar outras mulheres a seguirem o mesmo caminho. 

# Sobre a Sprint 📚

A Sprint foi composta por 6 dias repletos de vídeos e artigos voltados à dados, em que cada dia era abordado um tópico diferente. Partindo de conteúdos sobre carreira e processos seletivos, no primeiro dia, e concluindo com uma live, com 3 especialistas da área. No 3º dia, o tema era Inteligência Artificial e, além dos artigos e palestras, tivemos um workshop sobre Deep Learning, com a head de dados na NeuralMed, [Jéssica dos Santos](https://www.linkedin.com/in/jessica-santos-oliveira).

# Workshop 👩🏻‍💻

Começamos o workshop com uma base de dados, disponibilizada no [Kaggle](https://www.kaggle.com), sobre [exames de câncer de mama](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) com a finalidade de  classificar os tumores em malignos (cancerígenos) ou benignos (não cancerosos), utilizando Redes Neurais.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/e11b63cf-2c7d-4d02-8fdf-bb5774575297)

Não foi preciso fazer nenhum tratamento nos dados, pois a maioria deles já estava no formato numérico e não havia registros nulos ou equivocados. A única alteração que fizemos foi criar uma nova coluna para receber o diagnóstico dos exames na forma de números, já que eles estavam classificados, na coluna “**diagnosis**”,  como “M”, para tumores malignos, e “B”, para benignos. 

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/bc319f78-c3ed-4c00-bb80-f11c02155ac2)

A nova coluna foi denominada como “**diagnosis_maligno**”, em que todos os registros que estavam classificados como “B”, na coluna “**diagnosis**”, foram preenchidos com “0”, enquanto os exames que resultaram em tumores malignos, foram preenchidos com “1”.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/27a7e803-3892-4926-b08b-7a86989f8db3)

O próximo passo foi separar a base de dados entre **Treino**, **Teste** e **Validação**. Até o momento, eu não me recordo de ter feito nenhum projeto que teve a criação destes três conjuntos e achei super-interessante a finalidade do terceiro. Pelo o que entendi, a “Validação” é um grupo de registros que vai dizer se os resultados obtidos nos testes estão indo na direção correta, ou seja, é como se a Rede Neural estivesse fazendo um segundo teste e quando o resultado estiver incorreto, haverá uma mudança nos pesos para que haja um acerto na próxima vez.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/46c81360-57fd-488e-84f9-1bbe1519dbd0)

Antes de dar início a criação do modelo, também fizemos a normalização dos dados, como uma forma de garantir que eles estejam em uma escala comum e tenham características estatísticas semelhantes, por exemplo a média e a variância. A Jéssica citou duas formas de se fazer a normalização: 

- min-max, em que todos os registros passam a ter um intervalo de 0 à 1;
- e padronização, que dimensiona os dados para ter uma média igual a 0.

Usando a segunda opção, fizemos:

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/93efcbf9-0db4-444e-a45c-46d87b3fe259)













# Conclusão 🏁



## Ferramentas utilizadas 🧰
<p>
  <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
  <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://numpy.org/images/logo.svg" alt="numpy" width="40" height="40"/> </a>
  <a href="https://matplotlib.org/" target="_blank" rel="noreferrer"> <img src="https://matplotlib.org/_static/images/documentation.svg" alt="matplotlib" width="40" height="40"/> </a>
  <a href="https://www.tensorflow.org/?hl=pt-br" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png?20170429160244" alt="tensorflow" width="40" height="40"/> </a>
  <a href="https://keras.io/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/1200px-Keras_logo.svg.png" alt="keras" width="40" height="40"/> </a>
     </p>
