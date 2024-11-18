# UPDATES

## Update 1

Em `_change_direction` verifica se a direção escolhida pelo agente levaria a uma colisão com a parede. Se sim, substitui a direção por uma que seja válida.

### Resultado 1

Demora para encontrar a fruta.

## Update 2

Adicionei uma "reward recompensa maior para movimentos que aproximem a cobra da fruta.
Antes de cada movimento, calcula-se a distância entre a cabeça da cobra e a fruta.
Aproxime movimentos que diminuem essa distância com uma recompensa adicional.
Movimentos que aumentam a distância recebem uma penalização leve.

### Resultado 2

Demora pra encontrar a fruta.

## Update 3

Penaliza a cobra após exceder um número máximo de passos (max_steps) sem comer uma fruta.
Valor de max_steps ajustado para ser proporcional ao tamanho da grade (grid_size * grid_size).

Ajustes nas punições:

Colisão com a parede: -1 (reduzido de -2).
Colisão com o corpo: -0.5 (reduzido de -1).
Penalidade por passo: -0.01 (muito leve para cada movimento).
Penalidade por afastamento da fruta: -0.02.

Recompensas:

Recompensa de 0.1 por se aproximar da fruta.
Recompensa de +0.05 para cada movimento que não leva a colisões.
Recompensa de +0.1 por alcançar as bordas de forma segura.
Recompensa bônus de +0.5 ao comer uma fruta rapidamente.

Punições:
Penalidade de -0.5 se o número de passos exceder o limite.
Penalidade reduzida de -0.02 ao se afastar da fruta.

Removida a Condição de self.done por Demora:

Penalidades por demora continuam (-0.1), mas o jogo não termina.
Encerramento Apenas em Caso de Colisão:

O jogo só é encerrado quando ocorre uma colisão (parede ou corpo).
Penalidade Leve por Movimento Ineficiente:

### Hotfix

Removida a Condição de self.done por Demora.

Penalidades por demora continuam (-0.1), mas o jogo não termina.
Encerramento Apenas em Caso de Colisão.

O jogo só é encerrado quando ocorre uma colisão (parede ou corpo).
Penalidade Leve por Movimento Ineficiente:

Penalidade por demora foi ajustada para ser de -0.1.
