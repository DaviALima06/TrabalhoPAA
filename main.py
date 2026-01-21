
from __future__ import annotations

import math
import heapq
import random
from dataclasses import dataclass, field
from itertools import count
from typing import Dict, List, Optional, Tuple


# ============================================================
# Definição de Materiais e Pedidos
# ============================================================

@dataclass(frozen=True)
class Material:
    nome: str
    peso: float
    volume: float


@dataclass(frozen=True)
class Pedido:
    """
    Representa a quantidade de cada material que um cliente pediu
    - cliente_id: id do cliente
    - quantidades: dicionário {material_nome: quantidade}
    """
    cliente_id: int
    quantidades: Dict[str, int] = field(default_factory=dict)
    
    def peso_total(self, materiais: Dict[str, Material]) -> float:
        total = 0.0
        for nome_material, quantidade in self.quantidades.items():
            if nome_material in materiais:
                total += materiais[nome_material].peso * quantidade
        return total
    
    def volume_total(self, materiais: Dict[str, Material]) -> float:
        total = 0.0
        for nome_material, quantidade in self.quantidades.items():
            if nome_material in materiais:
                total += materiais[nome_material].volume * quantidade
        return total


# ============================================================
# Estado do Problema
# ============================================================

@dataclass(frozen=True)
class Estado:
    """
    - mascara: clientes já atendidos (clientes 1..)
    - posicao: posição atual do caminhão (0 = depósito)
    - peso_restante: quanto ainda cabe em peso no caminhão
    - volume_restante: quanto ainda cabe em volume no caminhão
    """
    mascara: int
    posicao: int
    peso_restante: float
    volume_restante: float


@dataclass
class InfoPai:
    anterior: Optional[Estado]
    acao: Tuple

# Geração de Pedidos Aleatórios

def gerar_materiais() -> Dict[str, Material]:
    #Cria 5 tipos de materiais de construção com peso e volume
    return {
        "areia": Material("areia", peso=1.5, volume=0.8),
        "cimento": Material("cimento", peso=2.0, volume=0.6),
        "tijolos": Material("tijolos", peso=3.0, volume=0.4),
        "madeira": Material("madeira", peso=0.8, volume=1.2),
        "ferro": Material("ferro", peso=4.0, volume=0.2),
    }

def gerar_pedidos_aleatorios(
    numero_clientes: int,
    materiais: Dict[str, Material],
    peso_maximo_caminhao: float,
    volume_maximo_caminhao: float,
    seed: Optional[int] = None
) -> Dict[int, Pedido]:
    """
    Gera pedidos aleatórios para cada cliente.
    Os limites por cliente são calculados baseado na capacidade do caminhão.
    """
    if seed is not None:
        random.seed(seed)
    
    pedidos: Dict[int, Pedido] = {}
    nomes_materiais = list(materiais.keys())
    
    # Calcula limites por cliente baseado no caminhão
    peso_maximo_cliente = peso_maximo_caminhao * 0.85
    volume_maximo_cliente = volume_maximo_caminhao * 0.85
    
    for cliente_id in range(1, numero_clientes + 1):
        quantidades: Dict[str, int] = {}
        peso_atual = 0.0
        volume_atual = 0.0
        
        # Seleciona aleatoriamente 1-5 tipos de materiais diferentes para este cliente
        num_tipos_materiais = random.randint(1, 5)
        materiais_selecionados = random.sample(nomes_materiais, min(num_tipos_materiais, len(nomes_materiais)))
        
        # Para cada material selecionado, decide uma quantidade aleatória
        for material_nome in materiais_selecionados:
            material = materiais[material_nome]
            
            # Calcula a quantidade máxima que cabe baseado em peso e volume
            max_por_peso = int((peso_maximo_cliente - peso_atual) / material.peso)
            max_por_volume = int((volume_maximo_cliente - volume_atual) / material.volume)
            max_quantidade = min(max_por_peso, max_por_volume)
            
            if max_quantidade > 0:
                # Define uma quantidade aleatória entre 1 e o máximo permitido
                quantidade_selecionada = random.randint(1, max(1, max_quantidade))
                
                novo_peso = peso_atual + material.peso * quantidade_selecionada
                novo_volume = volume_atual + material.volume * quantidade_selecionada
                
                # Só adiciona se couber dentro dos limites
                if novo_peso <= peso_maximo_cliente and novo_volume <= volume_maximo_cliente:
                    quantidades[material_nome] = quantidade_selecionada
                    peso_atual = novo_peso
                    volume_atual = novo_volume
        
        # Garante que o cliente tenha pelo menos algo para pedir
        if not quantidades:
            material_mais_leve = min(nomes_materiais, key=lambda x: materiais[x].peso)
            quantidades[material_mais_leve] = 1
        
        pedidos[cliente_id] = Pedido(cliente_id=cliente_id, quantidades=quantidades)
    
    return pedidos


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
    pedidos: Dict[int, Pedido],
    materiais: Dict[str, Material],
    peso_maximo_caminhao: float,
    volume_maximo_caminhao: float,
    voltar_ao_deposito_no_final: bool = True
) -> Tuple[float, List[List[int]]]:
    
    numero_clientes = len(pedidos)

    # Validações
    if len(coordenadas) != numero_clientes + 1:
        raise ValueError(f"A lista 'coordenadas' deve ter tamanho {numero_clientes + 1} (depósito + clientes).")

    for cliente_id, pedido in pedidos.items():
        peso = pedido.peso_total(materiais)
        volume = pedido.volume_total(materiais)
        
        if peso > peso_maximo_caminhao:
            raise ValueError(
                f"Problema inviável: cliente {cliente_id} pediu {peso:.2f}kg "
                f"(máximo do caminhão: {peso_maximo_caminhao:.2f}kg)"
            )
        if volume > volume_maximo_caminhao:
            raise ValueError(
                f"Problema inviável: cliente {cliente_id} pediu {volume:.4f}m³ "
                f"(máximo do caminhão: {volume_maximo_caminhao:.4f}m³)"
            )

    distancias = construir_matriz_distancias(coordenadas)
    mascara_completa = (1 << numero_clientes) - 1

    def bit_cliente(cliente: int) -> int:
        return 1 << (cliente - 1)

    # Estado inicial: nenhum atendido, no depósito, com capacidade cheia
    estado_inicial = Estado(
        mascara=0, 
        posicao=0, 
        peso_restante=peso_maximo_caminhao,
        volume_restante=volume_maximo_caminhao
    )

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
            
            pedido_cliente = pedidos[cliente]
            peso_pedido = pedido_cliente.peso_total(materiais)
            volume_pedido = pedido_cliente.volume_total(materiais)
            
            # Verifica se cabe em peso e volume
            if peso_pedido > estado.peso_restante or volume_pedido > estado.volume_restante:
                continue

            novo_estado = Estado(
                mascara=estado.mascara | b,
                posicao=cliente,
                peso_restante=estado.peso_restante - peso_pedido,
                volume_restante=estado.volume_restante - volume_pedido
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
                peso_restante=peso_maximo_caminhao,
                volume_restante=volume_maximo_caminhao
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
    # Materiais disponíveis
    materiais = gerar_materiais()
    
    # Depósito + clientes
    coordenadas = [
        (0.0, 0.0),   # 0 = depósito
        (2.0, 1.0),   # 1 = cliente 1
        (2.0, 4.0),   # 2 = cliente 2
        (5.0, 3.0),   # 3 = cliente 3
        (6.0, 1.0),   # 4 = cliente 4
    ]
    
    numero_clientes = len(coordenadas) - 1
    
    # Capacidades do caminhão
    peso_maximo_caminhao = 20 # kg
    volume_maximo_caminhao = 5 # m³
    
    pedidos = gerar_pedidos_aleatorios(
        numero_clientes=numero_clientes,
        materiais=materiais,
        peso_maximo_caminhao=peso_maximo_caminhao,
        volume_maximo_caminhao=volume_maximo_caminhao,
        seed=None  # sem seed para ter variabilidade
    )
    
    print("PEDIDOS DOS CLIENTES")
    for cliente_id, pedido in pedidos.items():
        peso = pedido.peso_total(materiais)
        volume = pedido.volume_total(materiais)
        print(f"\nCliente {cliente_id}:")
        for material_nome, quantidade in pedido.quantidades.items():
            mat = materiais[material_nome]
            print(f"{quantidade}x {material_nome:} {quantidade * mat.peso:.2f}kg | {quantidade * mat.volume:.2f}m³")

        print(f"Total: {peso:.2f}kg | {volume:.2f}m³")
    
    # Resolver o problema
    try:
        menor_distancia, viagens = resolver_cvrp_um_caminhao_pd(
            coordenadas=coordenadas,
            pedidos=pedidos,
            materiais=materiais,
            peso_maximo_caminhao=peso_maximo_caminhao,
            volume_maximo_caminhao=volume_maximo_caminhao,
            voltar_ao_deposito_no_final=True
        )
        
        print(f"Distância total: {menor_distancia:.3f}")
        print(f"Número de viagens: {len(viagens)}")
        
        for indice, viagem in enumerate(viagens, start=1):
            print(f"Viagem {indice}: {' -> '.join(map(str, viagem))}")
            
            # Calcula peso e volume da viagem
            peso_viagem = 0.0
            volume_viagem = 0.0
            for cliente_id in viagem:
                if cliente_id != 0:  # ignorar depósito
                    peso_viagem += pedidos[cliente_id].peso_total(materiais)
                    volume_viagem += pedidos[cliente_id].volume_total(materiais)
            
            print(f"Carga: {peso_viagem:.2f}kg / {peso_maximo_caminhao:.2f}kg")
            print(f"Volume: {volume_viagem:.2f}m³ / {volume_maximo_caminhao:.2f}m³")
        
    except RuntimeError as e:
        print(f"ERRO: {e}")


if __name__ == "__main__":
    main()
