#Alunos: Davi Alves Lima & Matheus


from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple, Optional


# ============================================================
# Materiais e Pedidos
# ============================================================

@dataclass(frozen=True)
class Material:
    nome: str
    peso_kg: int         # peso unitário em kg 
    volume_l: int        # volume unitário em litros 


@dataclass(frozen=True)
class Pedido:
    cliente_id: int
    quantidades: Dict[str, int] = field(default_factory=dict)

    def peso_total_kg(self, materiais: Dict[str, Material]) -> int:
        return sum(materiais[n].peso_kg * q for n, q in self.quantidades.items())

    def volume_total_l(self, materiais: Dict[str, Material]) -> int:
        return sum(materiais[n].volume_l * q for n, q in self.quantidades.items())


# ============================================================
# Distâncias (matriz)
# ============================================================

def distancia_euclidiana(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def construir_matriz_distancias(coordenadas: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(coordenadas)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = distancia_euclidiana(coordenadas[i], coordenadas[j])
    return dist


# ============================================================
# Geração de Materiais, Coordenadas e Pedidos 
# ============================================================

def gerar_materiais() -> Dict[str, Material]:
    """
    Materiais em kg e litros 
    1 m³ = 1000 litros.
    """
    return {
        "areia_saco_50kg":   Material("areia_saco_50kg",   peso_kg=50, volume_l=35),
        "cimento_saco_50kg": Material("cimento_saco_50kg", peso_kg=50, volume_l=32),
        "tijolo":            Material("tijolo",            peso_kg=3,  volume_l=2),
        "madeira_peca":      Material("madeira_peca",      peso_kg=12, volume_l=20),
        "ferro_barra":       Material("ferro_barra",       peso_kg=20, volume_l=6),
    }


def gerar_coordenadas_aleatorias(numero_clientes: int, seed: int = 7) -> List[Tuple[float, float]]:
    random.seed(seed)
    coords = [(0.0, 0.0)]  # depósito
    for _ in range(numero_clientes):
        coords.append((random.uniform(0, 25), random.uniform(0, 25)))  # “km” no plano
    return coords


def gerar_pedidos_aleatorios(
    numero_clientes: int,
    materiais: Dict[str, Material],
    capacidade_peso_kg: int,
    capacidade_volume_l: int,
    seed: int = 99
) -> Dict[int, Pedido]:
  
    random.seed(seed)

    limite_peso_cliente = capacidade_peso_kg // 3
    limite_volume_cliente = capacidade_volume_l // 3

    nomes = list(materiais.keys())
    pedidos: Dict[int, Pedido] = {}

    for cid in range(1, numero_clientes + 1):
        quantidades: Dict[str, int] = {}
        peso_atual = 0
        volume_atual = 0

        alvo_peso = random.randint(250, min(600, limite_peso_cliente))
        alvo_volume = random.randint(200, min(1500, limite_volume_cliente))

        tipos = random.randint(2, min(4, len(nomes)))
        selecionados = random.sample(nomes, tipos)

        for nome in selecionados:
            mat = materiais[nome]

            # quanto ainda dá para colocar sem estourar limites do cliente
            sobra_peso = limite_peso_cliente - peso_atual
            sobra_volume = limite_volume_cliente - volume_atual

            max_por_peso = sobra_peso // mat.peso_kg
            max_por_volume = sobra_volume // mat.volume_l
            max_qtd = min(max_por_peso, max_por_volume)

            if max_qtd <= 0:
                continue

            # tenta aproximar do alvo, mas sem exagero
            qtd = random.randint(1, max_qtd)

            novo_peso = peso_atual + mat.peso_kg * qtd
            novo_volume = volume_atual + mat.volume_l * qtd

            if novo_peso <= limite_peso_cliente and novo_volume <= limite_volume_cliente:
                quantidades[nome] = qtd
                peso_atual = novo_peso
                volume_atual = novo_volume

            # para quando já chegou próximo do alvo
            if peso_atual >= alvo_peso and volume_atual >= alvo_volume:
                break

        if not quantidades:
            # garante pedido mínimo
            nome_mais_leve = min(nomes, key=lambda x: materiais[x].peso_kg)
            quantidades[nome_mais_leve] = 1

        pedido = Pedido(cliente_id=cid, quantidades=quantidades)

        # valida regra 1/3 (segurança)
        if pedido.peso_total_kg(materiais) > limite_peso_cliente or pedido.volume_total_l(materiais) > limite_volume_cliente:
            # reduz para ficar válido
            nome_mais_leve = min(nomes, key=lambda x: materiais[x].peso_kg)
            pedido = Pedido(cliente_id=cid, quantidades={nome_mais_leve: 1})

        pedidos[cid] = pedido

    return pedidos


# ============================================================
# TSP exato (Held–Karp) para um subconjunto de clientes
# ============================================================

def tsp_held_karp( clientes: List[int], dist: List[List[float]] ) -> Tuple[float, List[int]]:
    """
    Retorna:
      - custo do melhor tour 0 -> ...clientes... -> 0
      - rota [0, ..., 0]
    """
    k = len(clientes)
    if k == 0:
        return 0.0, [0, 0]
    if k == 1:
        c = clientes[0]
        return dist[0][c] + dist[c][0], [0, c, 0]

    # dp[(mask, i)] = menor custo para sair do depósito e terminar em clientes[i] visitando mask
    dp: Dict[Tuple[int, int], float] = {}
    pai: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for i in range(k):
        m = 1 << i
        dp[(m, i)] = dist[0][clientes[i]]
        pai[(m, i)] = (-1, -1)

    for mask in range(1, 1 << k):
        for i in range(k):
            if not (mask & (1 << i)):
                continue
            chave = (mask, i)
            if chave not in dp:
                continue
            custo_atual = dp[chave]
            for j in range(k):
                if mask & (1 << j):
                    continue
                nmask = mask | (1 << j)
                novo = custo_atual + dist[clientes[i]][clientes[j]]
                chave2 = (nmask, j)
                if chave2 not in dp or novo < dp[chave2]:
                    dp[chave2] = novo
                    pai[chave2] = (mask, i)

    full = (1 << k) - 1
    melhor = float("inf")
    melhor_i = -1
    for i in range(k):
        chave = (full, i)
        if chave in dp:
            custo = dp[chave] + dist[clientes[i]][0]
            if custo < melhor:
                melhor = custo
                melhor_i = i

    # reconstrução
    ordem: List[int] = []
    mask = full
    i = melhor_i
    while i != -1:
        ordem.append(clientes[i])
        pm, pi = pai[(mask, i)]
        mask, i = pm, pi
    ordem.reverse()

    return melhor, [0] + ordem + [0]


# ============================================================
# PD exata para CVRP multi-viagens (partição de subconjuntos)
# ============================================================

def resolver_cvrp_pd_exata_particao(
    dist: List[List[float]],
    pedidos: Dict[int, Pedido],
    materiais: Dict[str, Material],
    capacidade_peso_kg: int,
    capacidade_volume_l: int,
    minimo_clientes_por_viagem: int = 2
) -> Tuple[float, List[List[int]]]:
   
    n = len(pedidos)
    full_mask = (1 << n) - 1

    # pesos/volumes por cliente (1..n)
    peso = [0] * (n + 1)
    vol = [0] * (n + 1)
    for c in range(1, n + 1):
        peso[c] = pedidos[c].peso_total_kg(materiais)
        vol[c] = pedidos[c].volume_total_l(materiais)

    # regra: cada cliente <= 1/3 da capacidade
    limite_peso_cliente = capacidade_peso_kg / 3.0
    limite_vol_cliente = capacidade_volume_l / 3.0
    for c in range(1, n + 1):
        if peso[c] > limite_peso_cliente or vol[c] > limite_vol_cliente:
            raise ValueError(
                f"Cliente {c} viola a regra 1/3: "
                f"{peso[c]}kg (lim {limite_peso_cliente:.1f}), {vol[c]}L (lim {limite_vol_cliente:.1f})"
            )

    # Como cada cliente <= 1/3, no máximo 3 clientes “grandes” cabem. Mesmo assim,
    #  limite superior realista pelo menor peso/volume.
    menor_peso = min(peso[1:])
    menor_vol = min(vol[1:])
    max_k_por_peso = max(2, min(6, capacidade_peso_kg // max(1, menor_peso)))
    max_k_por_vol = max(2, min(6, capacidade_volume_l // max(1, menor_vol)))
    max_k = min(max_k_por_peso, max_k_por_vol, 6)

    # Gera subconjuntos viáveis (máscaras) e calcula custo TSP exato
    custo_subset: Dict[int, float] = {}
    rota_subset: Dict[int, List[int]] = {}

    clientes_ids = list(range(1, n + 1))

    for k in range(minimo_clientes_por_viagem, max_k + 1):
        for comb in combinations(clientes_ids, k):
            peso_total = sum(peso[c] for c in comb)
            vol_total = sum(vol[c] for c in comb)
            if peso_total <= capacidade_peso_kg and vol_total <= capacidade_volume_l:
                # monta máscara
                mask = 0
                for c in comb:
                    mask |= 1 << (c - 1)
                # TSP exato para este subset
                custo, rota = tsp_held_karp(list(comb), dist)
                custo_subset[mask] = custo
                rota_subset[mask] = rota

    if not custo_subset:
        raise RuntimeError("Nenhum subconjunto viável com >=2 clientes. Ajuste volume/capacidade/pedidos.")

    # DP de cobertura
    INF = float("inf")
    F = [INF] * (1 << n)
    pai: List[Optional[Tuple[int, int]]] = [None] * (1 << n)  # (mask_anterior, subset)
    F[0] = 0.0

    # lista de subsets para iterar mais rápido
    subsets = list(custo_subset.keys())

    for mask in range(1 << n):
        if F[mask] >= INF:
            continue
        for sub in subsets:
            if mask & sub:
                continue  # já tem cliente repetido
            novo = mask | sub
            custo_novo = F[mask] + custo_subset[sub]
            if custo_novo < F[novo]:
                F[novo] = custo_novo
                pai[novo] = (mask, sub)

    if F[full_mask] >= INF:
        raise RuntimeError(
            "Não deu para cobrir todos os clientes com viagens de >=2. "
            "Aumente a capacidade, reduza pedidos ou permita viagem com 1."
        )

    # Reconstrução das viagens
    viagem_masks: List[int] = []
    cur = full_mask
    while cur != 0:
        p = pai[cur]
        if p is None:
            raise RuntimeError("Falha na reconstrução.")
        prev, sub = p
        viagem_masks.append(sub)
        cur = prev
    viagem_masks.reverse()

    rotas = [rota_subset[m] for m in viagem_masks]
    return F[full_mask], rotas


# ============================================================
# Relatório (distância, tempo, entregas por cliente)
# ============================================================

def calcular_distancia_rota(rota: List[int], dist: List[List[float]]) -> float:
    return sum(dist[a][b] for a, b in zip(rota, rota[1:]))


def formatar_tempo(tempo_h: float) -> str:
    minutos = int(round(tempo_h * 60))
    h = minutos // 60
    m = minutos % 60
    if h == 0:
        return f"{m} min"
    return f"{h} h {m:02d} min"


def imprimir_relatorio(
    rotas: List[List[int]],
    dist: List[List[float]],
    pedidos: Dict[int, Pedido],
    materiais: Dict[str, Material],
    capacidade_peso_kg: int,
    capacidade_volume_l: int,
    velocidade_km_h: float,
    custo_total: float
) -> None:
    print("\n" + "=" * 90)
    print("RELATÓRIO — CVRP (1 caminhão, múltiplas viagens)")
    print("=" * 90)
    print(f"Capacidade do caminhão: {capacidade_peso_kg} kg (2 toneladas) | {capacidade_volume_l} L")
    print(f"Velocidade média (para tempo): {velocidade_km_h:.1f} km/h")
    print("-" * 90)

    total_dist = 0.0
    total_tempo = 0.0
    total_peso = 0
    total_vol = 0
    total_entregas = 0

    for i, rota in enumerate(rotas, start=1):
        clientes = [x for x in rota if x != 0]
        dist_viagem = calcular_distancia_rota(rota, dist)
        tempo_h = dist_viagem / velocidade_km_h if velocidade_km_h > 0 else 0.0

        peso_viagem = sum(pedidos[c].peso_total_kg(materiais) for c in clientes)
        vol_viagem = sum(pedidos[c].volume_total_l(materiais) for c in clientes)

        total_dist += dist_viagem
        total_tempo += tempo_h
        total_peso += peso_viagem
        total_vol += vol_viagem
        total_entregas += len(clientes)

        print(f"\nVIAGEM {i}")
        print(f"Rota: {' -> '.join(map(str, rota))}")
        print(f"Clientes atendidos: {len(clientes)}")
        print(f"Distância: {dist_viagem:.3f} km | Tempo: {formatar_tempo(tempo_h)}")
        print(f"Carga entregue: {peso_viagem} kg / {capacidade_peso_kg} kg  |  Volume: {vol_viagem} L / {capacidade_volume_l} L")

        print("Entregas por cliente:")
        for c in clientes:
            p = pedidos[c].peso_total_kg(materiais)
            v = pedidos[c].volume_total_l(materiais)
            print(f"  Cliente {c:02d}: {p} kg | {v} L")
            for nome, qtd in pedidos[c].quantidades.items():
                mat = materiais[nome]
                print(f"    - {qtd}x {nome} ({qtd*mat.peso_kg} kg, {qtd*mat.volume_l} L)")

    print("\n" + "-" * 90)
    print("RESUMO GERAL")
    print("-" * 90)
    print(f"Total de viagens: {len(rotas)}")
    print(f"Total de entregas (clientes atendidos): {total_entregas}")
    print(f"Distância total: {total_dist:.3f} km")
    print(f"Distância total (PD): {custo_total:.3f} km")
    print(f"Tempo total estimado: {formatar_tempo(total_tempo)}")
    print(f"Peso total entregue: {total_peso} kg")
    print(f"Volume total entregue: {total_vol} L")
    print("=" * 90 + "\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    numero_clientes = 15

    # Capacidade pedida: 2 toneladas
    capacidade_peso_kg = 2000
    capacidade_volume_l = 12000   # 12 m³ = 12000 L (ajuste se quiser)
    velocidade_km_h = 40.0

    materiais = gerar_materiais()
    coordenadas = gerar_coordenadas_aleatorias(numero_clientes, seed=7)
    dist = construir_matriz_distancias(coordenadas)

    pedidos = gerar_pedidos_aleatorios(
        numero_clientes=numero_clientes,
        materiais=materiais,
        capacidade_peso_kg=capacidade_peso_kg,
        capacidade_volume_l=capacidade_volume_l,
        seed=99
    )

    # Mostrar pedidos (resumo)
    limite_peso = capacidade_peso_kg / 3.0
    limite_vol = capacidade_volume_l / 3.0

    print("\n" + "=" * 90)
    print("DADOS DO PROBLEMA")
    print("=" * 90)
    print(f"Clientes: {numero_clientes}")
    print(f"Capacidade caminhão: {capacidade_peso_kg} kg | {capacidade_volume_l} L")
    print(f"Regra: cada cliente <= 1/3 => {limite_peso:.1f} kg | {limite_vol:.1f} L")
    print("-" * 90)
    print("Resumo dos pedidos:")
    for c in range(1, numero_clientes + 1):
        p = pedidos[c].peso_total_kg(materiais)
        v = pedidos[c].volume_total_l(materiais)
        print(f"  Cliente {c:02d}: {p:4d} kg | {v:4d} L | itens={len(pedidos[c].quantidades)}")

    # Resolver (PD exata) garantindo >=2 clientes por viagem
    custo_total, rotas = resolver_cvrp_pd_exata_particao(
        dist=dist,
        pedidos=pedidos,
        materiais=materiais,
        capacidade_peso_kg=capacidade_peso_kg,
        capacidade_volume_l=capacidade_volume_l,
        minimo_clientes_por_viagem=2
    )

    imprimir_relatorio(
        rotas=rotas,
        dist=dist,
        pedidos=pedidos,
        materiais=materiais,
        capacidade_peso_kg=capacidade_peso_kg,
        capacidade_volume_l=capacidade_volume_l,
        velocidade_km_h=velocidade_km_h,
        custo_total=custo_total
    )


if __name__ == "__main__":
    main()