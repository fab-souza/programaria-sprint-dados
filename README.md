# programaria-sprint-dados

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=FINALIZADO&color=GREEN&style=for-the-badge)

![Badge code size](https://img.shields.io/github/languages/code-size/fab-souza/programaria-sprint-dados)

| :placard: Vitrine.Dev |    |
| -------------  | --- |
| :sparkles: Nome        | **PrograMaria Sprint Dados: Ampliando fronteiras**
| :label: Tecnologias | python
| :rocket: URL         | Notebook da [Parte 1](https://www.kaggle.com/fabianadesouza/programaria-sprint-dados-parte-1), [Parte 2](https://www.kaggle.com/code/fabianadesouza/programaria-sprint-dados-parte-2) e [Parte 3](https://www.kaggle.com/code/fabianadesouza/programaria-sprint-dados-parte-3)
| :fire: Desafio     | 

![](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/7ac8de8b-4d68-4966-ad83-96df8d9dc72c#vitrinedev)

A [PrograMaria](https://www.programaria.org/) é uma startup de impacto social que tem como missão empoderar mulheres e promover diversidade e inclusão no mundo da programação e da tecnologia, por meio de oficinas, eventos e cursos de formação técnica. 

Desde 2020, ano que comecei a pesquisar mais sobre dados, a PrograMaria foi uma das comunidades femininas que encontrei e me encantei. Tive a oportunidade de participar de alguns eventos online e foi onde encontrei uma linda rede de apoio, pois foi onde conheci muita gente maravilhosa, tanto palestrantes quanto outras participantes que estavam em uma situação parecida com a minha, que já estavam decididas a migrar de carreira ou que estavam no início da carreira em tecnologia e queriam apoiar outras mulheres a seguirem o mesmo caminho. 

<div id='sobre'></div>

# Sobre a Sprint 📚

A Sprint foi composta por 6 dias repletos de vídeos e artigos voltados à dados, em que cada dia era abordado um tópico diferente. Partindo de conteúdos sobre carreira e processos seletivos, no primeiro dia, e concluindo com uma live, com 3 especialistas da área. 

No 3º dia, o tema era Inteligência Artificial e, além dos artigos e palestras, tivemos um workshop sobre Deep Learning, com a head de dados na NeuralMed, [Jéssica dos Santos](https://www.linkedin.com/in/jessica-santos-oliveira).

## Índice:

- [Parte 1](https://github.com/fab-souza/programaria-sprint-dados#parte-1)
- [Parte 2](https://github.com/fab-souza/programaria-sprint-dados#parte-2)
- [Parte 3](https://github.com/fab-souza/programaria-sprint-dados#parte-3)
  
# Workshop 👩🏻‍💻

## Parte 1

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

Usando a segunda opção, obtivemos:

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/93efcbf9-0db4-444e-a45c-46d87b3fe259)

Para criar o modelo, primeiro, tivemos que importar o **Tensorflow** e **Keras**, que são bibliotecas usadas na criação de Machine Learning e Redes Neurais, respectivamente, e que eu nunca trabalhei anteriormente.

Em seguida, fizemos a função para criar o modelo.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/7e143df5-fcb0-4ab3-9063-76039ce4c01f)

Na primeira linha, definimos que as camadas serão criadas em sequência de forma manual.

Na linha seguinte, definimos a primeira camada da Rede Neural, estabelecendo parâmetros para os dados de entrada:

‘*Dense*’ significa que os neurônios da Rede estarão interligados, ‘*input_dim*’ diz quantas variáveis de entrada a Rede vai receber, ‘*units*’ refere-se a quantidade de neurônios que estarão conectados aos dados de entrada e no ‘*activation*’ informamos a função de ativação dos neurônios. No caso, ‘*relu*’ significa que os neurônios só serão ativados se os valores forem positivos, caso contrário, a informação não será passada adiante.

Depois criamos uma camada oculta, no primeiro parâmetro de ‘*Dense*’, estamos definindo o número de neurônios desta camada, seguida por sua ativação.

Finalizamos com o “1” que significa a quantidade de neurônios na saída, que neste caso é só para dizer se o tumor é maligno ou benigno. Sua ativação será diferente, “*sigmoid*”, e irá retornar a probabilidade do resultado ser do tipo maligno.

Com o modelo criado, fizemos um *summary* para visualizar sua arquitetura. Ele tem 431 parâmetros no total, mas de onde saiu este valor?

Na primeira linha, temos 310 parâmetros que referem-se às variáveis de entrada, a quantidade de neurônios e mais 1 viés (bias) para cada um. Em seguida, temos 110 parâmetros das camadas ocultas e seus vieses. Finalizando com 11 parâmetros, na camada ‘dense_2’, que são os neurônios da camada anterior com a variável de saída.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/fc8a12db-a3ac-4119-bbde-e6d86769303a)

Para fazer a compilação, referente a forma que a rede aprende, definimos três parâmetros:
- *optimizer* = responsável por ajustar os atributos da rede neural, como os pesos e taxas de aprendizado. Neste caso, a Jéssica escolheu o ‘Adam’, pois ele adapta a taxa de aprendizado de cada parâmetro com base em seus gradientes históricos e momento, acelerando o treinamento e melhorando o desempenho da rede.
- *loss* = determina o quão errado estão as previsões do modelo e podemos derivar os gradientes que são usados para atualizar os pesos.
- *learning rate* = é um hiperparâmetro que controla o quanto o modelo deve mudar em resposta ao erro estimado cada vez que os pesos do modelo são atualizados.

```
adam = Adam(lr = 0.01)                      
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
```

Seguindo para o treinamento, fizemos:

```
model.fit(x = X_train, y = y_train, validation_data = (X_val, y_val), batch_size = 16, epochs = 10)
```

Definimos no ‘*fit*’: o conjunto de treino, tanto X quanto y, os conjuntos de validação, o ‘*batch_size*’, que é o parâmetro que determina o número de exemplos de treinamento que serão propagados pela rede de cada vez. Neste caso, o ‘*fit*’ pega os 16 primeiros registros, faz o treinamento deles e treina a rede. Depois ele pega os próximos 16 registros, faz o treinamento, treina a rede e continua até passar por todos os registros. Finalizando com o ‘*epoch*’ (número de épocas), que é a quantidade de vezes que a rede vai percorrer os dados e aprender. 
No workshop, a Jéssica fez o ‘*fit*’ com 10 épocas. Os resultados obtidos pelo modelo da Jéssica foram:
- acurácia = 0.9808
- loss = 0.0971
  
Excelente.

O meu modelo ficou com:
- acurácia = 0.9783
- loss = 0.0981
  
Valores próximos. 😀 👍

Finalizando com a avaliação do modelo, usamos o ‘*.predict()*’ para fazer previsões com os dados do conjunto ‘X_test’ e a forma escolhida para verificar os resultados foi a matriz de confusão. E como ela funciona?

Ela mostra o número de previsões corretas e incorretas feitas pelo modelo, divididas por classe. Isso permite identificar quantas previsões o modelo acertou e quantas errou. Neste caso, estamos fazendo uma classificação binária (com duas classes), a matriz de confusão terá duas linhas e duas colunas. Cada célula da matriz representa uma combinação de classe verdadeira (real) e classe prevista (prevista pelo modelo). 

Mas antes de fazer a matriz, é preciso lembrar que o conjunto ‘*y_pred*’ está na forma de probabilidade, ou seja, ao invés de termos resultados entre 0 e 1, na verdade, temos valores de probabilidade do exame resultar em um tumor maligno ou não.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/cbe04e46-79ea-474c-a5c3-fbe73c0ddc2f)

Para apresentar a matriz, foi determinado que as probabilidades maiores do que 0,5 fossem consideradas como 1:

```
cm = confusion_matrix(y_test, y_pred > 0.5)
```

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/f1f62b3c-6400-4208-858b-5001b42b95d3)

Mas, o que os valores significam?
A célula na primeira linha e primeira coluna representa verdadeiros positivos (VP), ou seja, exemplos da primeira classe que foram corretamente classificados como pertencentes à ela. 

A célula na primeira linha e segunda coluna representa falsos negativos (FN), ou seja, exemplos da segunda classe que foram incorretamente classificados como pertencentes à primeira. 

A célula na segunda linha e primeira coluna representa falsos positivos (FP), os exemplos da primeira classe que foram incorretamente classificados como pertencentes à segunda. 

E por fim, a célula na segunda linha e segunda coluna representa verdadeiros negativos (VN), exemplos da segunda classe que foram corretamente classificados como pertencentes à ela.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/83e188e0-d92c-4dec-bf71-a67e584a770c)

O modelo do workshop resultou em:
- 37 resultados no Verdadeiro Positivo (tumores benignos que foram classificados corretamente)
- 1 resultado no Falso Negativo (tumor benigno classificado como maligno)
- 1 resultado no Falso Positivo (tumor maligno classificado como benigno)
- 18 resultados no Verdadeiro Negativo (tumores malignos classificados corretamente)

![parte1](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/8b58400b-7c5f-4c6a-9499-2f95b40448a2)

Enquanto no meu modelo, tive:

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/a1f26952-32c5-4f3b-8bf2-3b72d52fad89)

- 34 resultados no Verdadeiro Positivo
- nenhum no Falso Negativo
- 2 resultados no Falso Positivo
- e 21 resultados no Verdadeiro Negativo

Acredito que não cheguei ao mesmo resultado, (1º) porque não trabalhamos com os mesmos registros, ou seja, não utilizamos um ‘seed’ na hora de fazer a divisão entre *Treino*, *Teste* e *Validação*, que fez com que tivéssemos exames diferentes nos conjuntos. (2º) porque o modelo do workshop teve um desempenho melhor do que o meu, tanto na acurácia quanto no ‘*loss*’. 

<a href='#sobre'>🔼 Voltar ao Índice</a>

## Parte 2

Na segunda parte do workshop, fizemos um modelo para distinguir e classificar imagens de exames médicos, desta vez utilizando Redes Convolucionais. A base de dados também é do Kaggle, o [Medical MNIST](https://www.kaggle.com/datasets/andrewmvd/medical-mnist). 

Após fazer o carregamento dos arquivos, o caminho de cada imagem tornou-se uma variável e para saber seu tipo, criamos uma nova coluna para armazenar esta informação.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/01774eb7-fb06-4914-be5a-2a3670807b79)

A separação de **Treino** e **Teste** foi parecida com a primeira parte do workshop, ou seja, importação de biblioteca, criação de *X_train*, *X_test*, *y_train* e *y_test* e determinando o tamanho do teste. No caso da leitura de imagens, não foi preciso criar um conjunto de *Validação*, porque usamos uma classe que já faz isso, o **ImageDataGenerator**.

Para defini-la, primeiro, criamos uma variável que recebeu alguns parâmetros de processamento de imagens: *rescale* e *validation_split*. Uma para multiplicar os dados por um valor fornecido antes de qualquer outra transformação, como uma forma de normalização, e para reservar uma fração dos dados de treinamento para validação, respectivamente.

Em seguida, criamos o gerador de treino (*train_generator*) depois o gerador de validação (*valid_generator*). Definimos em ambos, o *dataframe*, a coluna ‘x’, a coluna ‘y’, o tamanho das imagens, as cores das imagens (no caso, preto e branco = escalas em cinza), a quantidade de arquivos que serão carregados ao mesmo tempo e o que aquele gerador é (treino ou validação).

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/69abe1a5-7f62-4b3d-99c7-76b3f42821bd)

Seguimos para a criação da função que irá construir o modelo:

```
def build_model():

    model = Sequential()

    model.add(layers.Conv2D(filters = 32, kernel_size = 2, activation = "relu", input_shape = (64, 64, 1)))
    model.add(layers.MaxPooling2D(pool_size = 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(28, activation = 'relu'))
    model.add(layers.Dense(6, activation = 'softmax'))

    return model
```

Na primeira linha, definimos que as camadas serão criadas em sequência, igual foi feito no primeiro modelo.

Depois, criamos a primeira camada, uma camada convolucional com sua quantidade de filtros, tamanho das matrizes, tipo de ativação e o formato das imagens, em escala de cinza.

Na próxima camada, *MaxPooling2D*, ela reduz a amostra de entrada ao longo de suas dimensões espaciais (altura e largura), pegando o valor máximo sobre uma janela de entrada (de tamanho definido por *pool_size*) para cada canal da entrada e calcula sua média.

Na camada de *Dropout*, tentamos reduzir a ocorrência de *overfit* no modelo. Neste caso, estamos desabilitando 30% dos neurônios da rede.

Com o *Flatten*, fazemos com que a matriz de entrada seja ‘achatada’, na forma de um array, antes de passá-la para a camada densa. 

Depois criamos uma camada oculta densa, definindo seu número de neurônios, seguida por sua ativação (*relu*).

Finalizando com a camada de saída com 6 neurônios, referentes às categorias de imagens, e sua ativação, que neste caso é “*softmax*”, que retorna a probabilidade da imagem pertencer às categorias e a soma delas resulta em 1. Por exemplo:
- AbdomenCT = 0,2   
- Hand = 0,1        
- CXR = 0,25         
- HeadCT = 0,2      
- ChestCT = 0,25     
- BreastMRI = 0,27

Para criar o modelo, fizemos um *summary* para visualizar sua arquitetura e vemos que o modelo tem mais do que 850 mil parâmetros.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/bf06d4b3-9004-45c0-96fd-8584e15f4466)

Passando para a compilação do modelo, também definimos três parâmetros (*optimizer*, *loss* e *learning rate*) como foi feito no primeiro modelo. A única diferença é que no *loss*, mudamos de *binary_crossentropy* para *categorical_crossentropy*, pois as saídas são exames de diferentes tipos.

Antes de iniciar o treinamento, definimos mais algumas funções de otimização, também chamados de *callbacks*: *ModelCheckpoint* e *EarlyStopping*. O primeiro, salva o melhor modelo enquanto os treinamentos ocorrem. Nele, definimos como os modelos serão chamados, a métrica usada para avaliar o desempenho do modelo (*val_loss*), o modo de avaliação (*min* = queremos obter o menor valor treinado), no *verbose* determinamos se isso será feito em todas as épocas e confirmamos o salvamento do melhor modelo.

No segundo, interrompemos o treinamento mais cedo, quando a métrica monitorada para de melhorar. Definimos a métrica que deverá ser observada (*val_loss*), o treinamento será interrompido quando não houver uma melhoria absoluta maior que 0,001 (*min_delta*) por 5 épocas consecutivas (*patience*) e o modo é definido como *min*, o que significa que o treinamento será interrompido quando a quantidade monitorada parar de diminuir.

```
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
checkpoint = ModelCheckpoint('medical_image_model.hdf5', monitor = 'val_loss', verbose = 1, mode = 'min', save_best_only = True)
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 5, mode = 'min', verbose = 1)
```

Para o treinamento do modelo, também usamos o *.fit()*, igual foi feito na primeira parte do workshop. Também informamos que o modelo deve usar *train_generator* como dados de treinamento, o uso dos callbacks que criamos anteriormente, o número de etapas por época (*train_generator.samples//BATCH_SIZE*), o uso de *valid_generator* como dados de validação, o número de etapas de validação (*valid_generator.samples//BATCH_SIZE*) e o modelo será treinado por, no máximo, 25 épocas.

```
model.fit(train_generator, callbacks = [checkpoint, early_stop], steps_per_epoch = train_generator.samples//BATCH_SIZE, 
          validation_data = valid_generator, validation_steps = valid_generator.samples//BATCH_SIZE, epochs = 25)
```

No meu caso, parou de ser treinado na época 12 com uma acurácia na validação de 0,9991 e uma perda na validação de 0,0028. Ambos foram um pouco melhor do que o modelo apresentado pela Jéssica, que obteve *val_loss* = 0,0041 e *val_accuracy* = 0,9986.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/92d08ca3-b4c1-4731-a41f-d7306fc4de05)

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/68764596-0ee9-4892-b35e-cc598cdb5ca9)

Para aplicar o modelo treinado no conjunto de teste, não foi criado mais um *ImageGenerator* para eles, usamos duas funções do **Keras** que fazem o carregamento e a leitura das imagens, o *load_img* e *img_to_array*, pois a quantidade de arquivos era bem menor do que o conjunto de treino. Usamos o *.predict* no conjunto de teste e as previsões foram armazenadas na variável *y_pred*. Mas diferente do primeiro caso, que a saída era binária, não é possível avaliar o resultado fazendo *y_pred > 0.5*. Foi preciso usar um *.argmax(axis = 1)* para retornar o índice com o maior valor ao longo do eixo especificado, ou seja, o índice com o valor máximo ao longo do eixo 1 foi salvo na nova variável. Assim, obtivemos a classe prevista para cada imagem de teste.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/8273aecf-7a9d-4eb6-be73-8dbc303f6073)

As classes dos exames ainda estavam como *string* ('AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT') e utilizamos um *LabelEncoder* para transformá-las em uma sequência de números, igual as previsões. 

E finalizamos a avaliação com uma matriz de confusão, nela observamos que não obtive bons resultados. Com exceção dos exames do tipo 2 e 4, pois eles tiveram os maiores valores na linha diagonal da matriz, os demais exames foram mais classificados em outras categorias.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/0d14b5ee-693f-40d7-8a3a-b0574985f985)

Durante o workshop, a Jéssica reparou que o *LabelEncoder* mudaria as classes dos exames seguindo uma ordem alfabética, enquanto o *y_pred* seguiu a ordem que as classes foram aparecendo. Eu tentei corrigir este erro ao fazer um *LabelEncoder.fit*, mas acho que não obtive sucesso. Fica de lição para o próximo projeto.

<a href='#sobre'>🔼 Voltar ao Índice</a>

## Parte 3:

Para finalizar o workshop, a Jéssica ensinou como fazer uma rede sem precisar definir sua arquitetura, usar uma que já aprendeu a identificar outras categorias de imagens e adaptá-la ao nosso projeto, ou seja fazer um **transfer learning**. A maior diferença entre este modelo e o anterior, é que desta vez usamos imagens coloridas, pois a arquitetura pronta foi treinada desta forma. Então, após fazer a importação das bibliotecas e arquivos, separar *Treino* e *Teste*, criamos o *train_generator* e *valid_generator* para imagens coloridas.

```
train_generator = data_generator.flow_from_dataframe(dataframe = df_train, x_col = 'path', y_col = 'class', 
                                                     class_mode = 'categorical', batch_size = BATCH_SIZE, 
                                                     target_size = (64,64), subset = 'training', color_mode = 'rgb')
valid_generator = data_generator.flow_from_dataframe(dataframe = df_train, x_col = 'path', y_col = 'class', 
                                                     class_mode = 'categorical', batch_size = BATCH_SIZE, 
                                                     target_size = (64,64), subset = 'validation', color_mode = 'rgb')
```

Na criação do *Transfer Learning*, usamos a rede *MobileNetV2*, por ela ser menor e conseguir treinar o modelo mais rápido. Também definimos usar todos os pesos das milhões de imagens que passaram por ela, no *include_top* definimos que não queremos as 1000 classes de saída, e sim, as 6 categorias de exames, e no *input_shape*, o tamanho das imagens. Neste modelo, não usamos o *MaxPooling2D* e *Dropout*, mas fizemos o congelamento de algumas camadas no *for*. Transformamos a saída da camada anterior em uma camada densa ao fazer *x = base_model.output*, seguido por *x = layers.GlobalAveragePooling2D()(x)* que, diferente do *Flatten*, traz a média geral de matriz. Concluindo com o uso da camada densa na construção do modelo, *model = Model(base_model.input, predictions)*.

```
def build_model2(shape):

    base_model = MobileNetV2(weights = "imagenet", include_top = False, input_shape = shape)
    # congelando camadas que não iremos treinar.
    # para congelar alguns layers específicos basta passar o indice: for layer in mobile.layers[:5]:
    for layer in base_model.layers[:3]:
        layer.Trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(6, activation = 'softmax')(x)

    model = Model(base_model.input, predictions)

    return model
```

Construímos o modelo e ao fazer o resumo, vemos que ele possui mais do que 2 milhões de parâmetros. 

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/821dcfc9-3be7-4e3d-8d18-080184894318)

Fizemos a criação dos callbacks *ModelCheckpoint* e *EarlyStopping*, seguido pela compilação do modelo, que chegou a um bom resultado um pouco mais rápido, em 10 épocas. Aplicamos o modelo no conjunto de teste e obtive um resultado pior, porque os exames foram classificados apenas como tipo 4 e 5.

![image](https://github.com/fab-souza/programaria-sprint-dados/assets/67301805/f260f192-03fd-457b-ab51-0cc4df5d286a)

<a href='#sobre'>🔼 Voltar ao Índice</a>

# Conclusão 🏁

Antes deste workshop, eu nunca tinha trabalhado com Deep Learning, muito menos com Rede Neural e achei interessante o fato de poder atribuir “pesos” às variáveis, sem precisar balancear os dados, algo que fiz nos meus projetos anteriores. Mesmo não obtendo bons resultados na classificação de imagens, eu gostei de ter aprendido uma nova ferramenta, de ter este primeiro contato com este tipo de modelo de Machine Learning e até consigo imaginar alguns projetos pessoais em que posso replicar este conhecimento.

Sei que preciso corrigir a questão do *labelEncoder*, mas adquiri um novo interesse e pretendo melhorar a forma que utilizo esta ferramenta. 

---

Muito obrigada por chegar até aqui e até a próxima 🤗

## Ferramentas utilizadas 🧰
<p>
  <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
  <a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://numpy.org/images/logo.svg" alt="numpy" width="40" height="40"/> </a>
  <a href="https://matplotlib.org/" target="_blank" rel="noreferrer"> <img src="https://matplotlib.org/_static/images/documentation.svg" alt="matplotlib" width="40" height="40"/> </a>
  <a href="https://www.tensorflow.org/?hl=pt-br" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png?20170429160244" alt="tensorflow" width="40" height="40"/> </a>
  <a href="https://keras.io/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/1200px-Keras_logo.svg.png" alt="keras" width="40" height="40"/> </a>
     </p>
