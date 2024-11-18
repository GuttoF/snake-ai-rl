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

Demora pra encontrar a fruta