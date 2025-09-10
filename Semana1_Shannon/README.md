# Experimento de Shannon — README

## O que este notebook faz
- Carrega um corpus (BookSum), limpa o texto para ficar só com letras e espaços.
- Gera texto de forma **aleatória**, em etapas:
  - **Uniforme**: todas as letras (e espaço) têm a mesma probabilidade
  - **n‑grams (caracteres)**: a próxima letra é sorteada olhando a ** letra anterior** (tamanho *n‑1*).  
    Com **n=1** o gerador é simples e realiza apenas uma contagem de cada letra para gerar sua probabilidade. 


## Por que isso importa
Shannon mostrou que só **prever o próximo símbolo** já dá uma "cara" de linguagem. Os modelos atuais (LLMs) ainda jogam o jogo da previsão, mas com **muito mais dados e contexto e de maneira neural**, mas ainda seguem um princípio muito próximo do proposto por Shannon no século passado.
