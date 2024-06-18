# Trabalho_Pai
# Leia requirements e Refences

#Tutorial
---- Tutorial para Execução dos Programa ----

Siga esses passos para criar, treinar e testar um modelo de reconhecimento de gestos de mão usando aprendizado de 
máquina em Python.

Passo 1: Preparação do Ambiente
    .Instale as dependências necessárias:
        pip install opencv-python mediapipe numpy scikit-learn

Passo 2: Coletar Dados:
    .Execute coleta.py ou coleta_com_augmentation.py para capturar imagens dos gestos e salvá-los em diretórios correspondentes.

Passo 3:Criar Modelo:
    .Execute criacao_modelo.py  para criar o modelo.

Passo 4: Treinar o Modelo:
    .Execute treinamento_modelo.py para processar as imagens, extrair características e treinar um modelo SVM. O modelo 
    treinado será salvo como model.p.

Passo 5: Testar o Modelo:
    .Execute teste_modelo.py para carregar o modelo treinado e usá-lo para reconhecer gestos em tempo real usando a  webcam. ou teste_modelo_video.py para video.