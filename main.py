
from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from itertools import count
from typing import Dict, List, Optional, Tuple


# ============================================================
# 
# ============================================================

@dataclass(frozen=True)
class Estado:
    """
      - mascara: clientes já atendidos (clientes 1..)
      - posicao: posição atual do caminhão (0 = depósito)
      - capacidade_restante: quanto ainda cabe no caminhão (0..C)
    """
    mascara: int
    posicao: int
    capacidade_restante: int


@dataclass
class InfoPai:

    anterior: Optional[Estado]
    acao: Tuple


# ============================================================
# Distâncias
# ============================================================

def distancia_euclidiana(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def construir_matriz_distancias(coordenadas: List[Tuple[float, float]]) -> List[List[float]]:
    total = len(coordenadas)
    dist = [[0.0] * total for _ in range(total)]
    for i in range(total):
        for j in range(total):
            dist[i][j] = distancia_euclidiana(coordenadas[i], coordenadas[j])
    return dist


# ============================================================
# Solução (PD no espaço de estados + Dijkstra)
# ============================================================

def resolver_cvrp_um_caminhao_pd(
    coordenadas: List[Tuple[float, float]],
    demandas: List[int],
    capacidade_maxima: int,
    voltar_ao_deposito_no_final: bool = True
) -> Tuple[float, List[List[int]]]:
    
    numero_clientes = len(coordenadas) - 1

    # Validações
    if len(demandas) != numero_clientes + 1:
        raise ValueError("A lista 'demandas' deve ter tamanho n+1 e demandas[0] = 0 (depósito).")

    for j in range(1, numero_clientes + 1):
        if demandas[j] <= 0:
            raise ValueError(f"Demanda do cliente {j} deve ser positiva.")
        if demandas[j] > capacidade_maxima:
            raise ValueError(
                f"Problema inviável: cliente {j} tem demanda {demandas[j]} maior que a capacidade {capacidade_maxima}."
            )

    distancias = construir_matriz_distancias(coordenadas)
    mascara_completa = (1 << numero_clientes) - 1

    def bit_cliente(cliente: int) -> int:
        return 1 << (cliente - 1)

    # Estado inicial: nenhum atendido, no depósito, com capacidade cheia
    estado_inicial = Estado(mascara=0, posicao=0, capacidade_restante=capacidade_maxima)

    # dp: menor custo conhecido para cada estado
    dp: Dict[Estado, float] = {estado_inicial: 0.0}

    # pai: para reconstruir
    pai: Dict[Estado, InfoPai] = {estado_inicial: InfoPai(anterior=None, acao=("inicio",))}

    # Fila de prioridade (Dijkstra)
    contador = count()
    fila: List[Tuple[float, int, Estado]] = [(0.0, next(contador), estado_inicial)]

    while fila:
        custo_atual, _, estado = heapq.heappop(fila)

        #  algo melhor para este estado, ignora
        if custo_atual != dp.get(estado, float("inf")):
            continue

        #  Tentar visitar um novo cliente j
        for cliente in range(1, numero_clientes + 1):
            b = bit_cliente(cliente)
            if estado.mascara & b:
                continue  # já atendido
            if demandas[cliente] > estado.capacidade_restante:
                continue  # não cabe

            novo_estado = Estado(
                mascara=estado.mascara | b,
                posicao=cliente,
                capacidade_restante=estado.capacidade_restante - demandas[cliente]
            )
            novo_custo = custo_atual + distancias[estado.posicao][cliente]

            if novo_custo < dp.get(novo_estado, float("inf")):
                dp[novo_estado] = novo_custo
                pai[novo_estado] = InfoPai(anterior=estado, acao=("visitar", cliente))
                heapq.heappush(fila, (novo_custo, next(contador), novo_estado))

        # Voltar ao depósito para recarregar e iniciar nova viagem
        if estado.posicao != 0:
            novo_estado = Estado(
                mascara=estado.mascara,
                posicao=0,
                capacidade_restante=capacidade_maxima
            )
            novo_custo = custo_atual + distancias[estado.posicao][0]

            if novo_custo < dp.get(novo_estado, float("inf")):
                dp[novo_estado] = novo_custo
                pai[novo_estado] = InfoPai(anterior=estado, acao=("recarregar",))
                heapq.heappush(fila, (novo_custo, next(contador), novo_estado))

    # Escolher melhor estado final 
    melhor_estado_final: Optional[Estado] = None
    melhor_custo_total = float("inf")

    for estado, custo in dp.items():
        if estado.mascara != mascara_completa:
            continue

        custo_final = custo
        if voltar_ao_deposito_no_final and estado.posicao != 0:
            custo_final += distancias[estado.posicao][0]

        if custo_final < melhor_custo_total:
            melhor_custo_total = custo_final
            melhor_estado_final = estado

    if melhor_estado_final is None:
        raise RuntimeError("Nenhuma solução encontrada (verifique se o problema é viável).")

    # ============================================================
    # Reconstrução das ações
    # ============================================================

    acoes: List[Tuple] = []
    atual = melhor_estado_final
    while True:
        info = pai[atual]
        if info.anterior is None:
            break
        acoes.append(info.acao)
        atual = info.anterior
    acoes.reverse()

    if voltar_ao_deposito_no_final and melhor_estado_final.posicao != 0:
        acoes.append(("retorno_final",))

    # ============================================================
    # rotas
    # ============================================================

    viagens: List[List[int]] = []
    viagem_atual: List[int] = [0]  # sempre começa no depósito

    for acao in acoes:
        if acao[0] == "visitar":
            cliente = acao[1]
            viagem_atual.append(cliente)

        elif acao[0] == "recarregar":
            # fecha a viagem atual voltando ao depósito
            if viagem_atual[-1] != 0:
                viagem_atual.append(0)
            viagens.append(viagem_atual)

            # inicia nova viagem
            viagem_atual = [0]

        elif acao[0] == "retorno_final":
            if viagem_atual[-1] != 0:
                viagem_atual.append(0)
            viagens.append(viagem_atual)
            viagem_atual = [0]

    # Remove viagens vazias 
    viagens = [v for v in viagens if len(v) > 1]

    return melhor_custo_total, viagens


# ============================================================
# Testes
# ============================================================

def main() -> None:
    # Depósito + clientes 
    coordenadas = [
        (0.0, 0.0),  # 0 = depósito
        (2.0, 1.0),  # 1 = cliente 1
        (2.0, 4.0),  # 2 = cliente 2
        (5.0, 3.0),  # 3 = cliente 3
        (6.0, 1.0),  # 4 = cliente 4
    ]

    # demandas[0] = 0 depósito não tem demanda
    demandas = [0, 2, 3, 4, 2]

    capacidade_maxima = 5

    menor_distancia, viagens = resolver_cvrp_um_caminhao_pd(
        coordenadas=coordenadas,
        demandas=demandas,
        capacidade_maxima=capacidade_maxima,
        voltar_ao_deposito_no_final=True
    )

    print(f"Menor distância total: {menor_distancia:.3f}")
    for indice, viagem in enumerate(viagens, start=1):
        print(f"Viagem {indice}: " + " -> ".join(map(str, viagem)))


if __name__ == "__main__":
    main()
