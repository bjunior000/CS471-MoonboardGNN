# Presentation Script

## English Script

Hello, our project is about difficulty prediction of MoonBoard climbing problems using an Edge-Aware Graph Attention Network.

A MoonBoard problem is defined by a set of selected holds on a fixed climbing board. Its difficulty is not determined only by which holds are used, but also by their positions, their roles, and the movement relations between them. So in this project, we represent each MoonBoard problem as a graph and learn a graph embedding for grade prediction.

The task is graph classification. The input is one MoonBoard problem with start, middle, and end holds, plus problem-level metadata such as setter ID, benchmark flag, and repeat count. Each hold has a board position and role information. The output is a discrete difficulty grade, such as 6A+ or 7B. In the graph, nodes are holds, edges are spatially reachable hold-to-hold relations, and edge attributes describe relative position, distance, direction, and relation statistics.

There are two related existing approaches. MoonBoardRNN first infers a move sequence using BetaMove, and then uses an LSTM-based GradeNet to predict difficulty from move-level features. Another approach, MoonGen or TextGCN-style GNN, builds one global graph from the entire dataset, where problems and holds are nodes, and edges represent occurrence or co-occurrence relations.

Our approach is different because we construct a separate physical graph for each MoonBoard problem. Holds are nodes, possible movement relations are edges, and edge attributes describe how two holds are spatially related. The key idea is that the model should not only know that two holds are connected, but also what kind of movement relation that edge represents.

The model has two branches. In the graph branch, the hold graph is passed into Edge-Aware GAT layers. These layers update hold embeddings using both node features and edge attributes. Then mean and max graph pooling produce a graph-level representation, and a graph head MLP transforms it into the graph embedding `gh`.

In the dense branch, problem-level metadata and statistics are standardized and passed through another MLP, producing a dense embedding `dh`. Then `gh` and `dh` are concatenated and passed to a final MLP classifier. The output is grade logits, and the predicted grade is the class with the highest logit.

For training, we used cross entropy loss for grade classification, plus an ordinal loss that penalizes larger grade-index errors. This is useful because difficulty grades are ordered. For evaluation, we used exact accuracy, within-1 accuracy, and macro-F1.

The result shows that Edge-Aware GAT improves over vanilla GAT and also improves exact accuracy compared with the reported MoonBoardRNN result. The Edge-Aware GAT achieved about 0.508 exact accuracy, 0.851 within-1 accuracy, and 0.304 macro-F1. The macro-F1 improvement suggests that edge-aware movement information helps the model classify grades more effectively. However, high-grade classes still have fewer samples and remain difficult to predict, so overcoming class imbalance in the provided dataset can be a future direction.

## Korean Script

안녕하세요. 저희 프로젝트는 **Edge-Aware Graph Attention Network를 이용해서 MoonBoard 문제의 난이도를 예측하는 것**입니다.

MoonBoard 문제는 고정된 클라이밍 보드 위에서 선택된 홀드들의 집합으로 정의됩니다. 그런데 난이도는 단순히 어떤 홀드가 쓰였는지만으로 결정되지 않고, 홀드의 위치, start/middle/end 같은 역할, 그리고 홀드 사이의 이동 관계에도 영향을 받습니다. 그래서 저희는 각 MoonBoard 문제를 그래프로 표현하고, 그 graph embedding을 이용해서 난이도 grade를 예측했습니다.

문제는 graph classification으로 정의했습니다. 입력은 start, middle, end hold를 가진 하나의 MoonBoard problem이고, 여기에 setter ID, benchmark 여부, repeat count 같은 problem-level metadata도 포함됩니다. 출력은 6A+, 7B 같은 discrete difficulty grade입니다. 그래프에서 node는 각 hold이고, edge는 공간적으로 도달 가능한 hold-to-hold 관계입니다. edge attribute에는 상대 위치, 거리, 방향, relation statistics 같은 정보가 들어갑니다.

관련된 기존 접근은 두 가지가 있습니다. 첫 번째는 MoonBoardRNN입니다. 이 방법은 BetaMove로 move sequence를 먼저 추정한 뒤, LSTM 기반 GradeNet으로 move-level feature에서 난이도를 예측합니다. 두 번째는 MoonGen, 또는 TextGCN-style GNN입니다. 이 방법은 전체 dataset을 하나의 global graph로 만들고, problem node와 hold node 사이의 occurrence, co-occurrence 관계를 이용합니다.

저희 접근의 차이는 각 문제마다 별도의 physical graph를 만든다는 점입니다. 즉 전체 데이터셋 graph가 아니라, 하나의 MoonBoard problem 자체를 graph classification sample로 봅니다. 그리고 edge가 단순히 연결 여부만 나타내는 것이 아니라, 두 홀드 사이가 어떤 이동 관계를 가지는지도 표현하도록 했습니다.

모델은 두 branch로 구성됩니다. graph branch에서는 problem graph가 Edge-Aware GAT layer를 통과합니다. 이 layer는 node feature뿐 아니라 edge attribute도 사용해서 hold embedding을 업데이트합니다. 이후 mean pooling과 max pooling으로 graph-level representation을 만들고, graph head MLP를 통해 graph embedding `gh`를 만듭니다.

dense branch에서는 problem-level metadata와 statistics를 standardized feature로 만든 뒤, dense head MLP를 통과시켜 dense embedding `dh`를 만듭니다. 마지막으로 `gh`와 `dh`를 concat하고, final MLP classifier가 difficulty grade logits를 출력합니다. 예측 grade는 가장 logit이 큰 class입니다.

학습에는 grade classification을 위한 cross entropy loss와, grade 순서성을 반영하기 위한 ordinal loss를 함께 사용했습니다. 평가는 exact accuracy, within-1 accuracy, macro-F1으로 진행했습니다.

결과적으로 Edge-Aware GAT는 vanilla GAT보다 성능이 좋았고, MoonBoardRNN의 reported exact accuracy보다도 높은 exact accuracy를 보였습니다. Edge-Aware GAT는 exact accuracy 약 0.508, within-1 accuracy 약 0.851, macro-F1 약 0.304를 기록했습니다. 특히 macro-F1이 개선된 것은 edge-aware movement relation이 grade prediction에 도움이 된다는 근거로 볼 수 있습니다. 다만 높은 난이도 class는 데이터 수가 적어서 여전히 예측이 어렵고, 제공된 dataset의 class imbalance를 극복하는 것이 추후 과제가 될 수 있겠습니다.

## Additional Explanations / Expected Questions

### 1. What does Edge-Aware exactly mean?

**English**

In a standard GAT, attention is usually computed mainly from node embeddings. In our Edge-Aware GAT, edge attributes are also used when computing the attention score and the message passed between two holds. This means the model does not only ask, "which neighboring hold is important?", but also "what kind of relation connects these two holds?" For MoonBoard problems, this matters because a movement can be easy or difficult depending on relative distance, direction, and hold roles.

**Korean**

일반적인 GAT에서는 attention score가 주로 node embedding을 기준으로 계산됩니다. 하지만 저희 Edge-Aware GAT에서는 edge attribute도 attention score와 message passing에 함께 사용합니다. 즉 모델이 단순히 "어떤 이웃 hold가 중요한가"만 보는 것이 아니라, "두 hold가 어떤 이동 관계로 연결되어 있는가"도 함께 봅니다. MoonBoard에서는 두 hold 사이의 거리, 방향, 역할에 따라 이동 난이도가 달라질 수 있기 때문에 이 정보가 중요합니다.

### 2. What are setter ID, benchmark flag, and repeat count, and why do they matter?

**English**

Setter ID identifies the person or account that created the problem. It may matter because different setters can have different grading tendencies or route-setting styles. The benchmark flag indicates whether a problem is officially recognized as a benchmark problem, which can be related to more reliable or more widely accepted grading. Repeat count represents how many climbers have repeated the problem. A high repeat count may indicate that the grade is more stable, while low-repeat problems may have noisier labels. These features are not graph structure, but they provide problem-level context that can help grade prediction.

**Korean**

Setter ID는 문제를 만든 사람 또는 계정을 나타냅니다. 서로 다른 setter는 난이도를 매기는 기준이나 문제 스타일이 다를 수 있기 때문에 의미가 있을 수 있습니다. Benchmark flag는 해당 문제가 공식 benchmark 문제인지 나타내며, benchmark 문제는 grade가 더 안정적이거나 널리 받아들여졌을 가능성이 있습니다. Repeat count는 그 문제를 반복해서 완등한 사람 수를 의미합니다. 반복 수가 많으면 grade가 더 안정적일 가능성이 있고, 반복 수가 적은 문제는 label noise가 클 수 있습니다. 이 정보들은 graph structure 자체는 아니지만, problem-level context로서 grade prediction에 도움을 줄 수 있습니다.

### 3. What does relation statistics mean in edge attributes?

**English**

Relation statistics are train-only statistics computed from hold-pair patterns. For example, if similar hold-to-hold relations frequently appear in harder problems in the training set, that relation receives a higher difficulty-related encoding. We compute these statistics only from the training split to avoid test leakage. They summarize how certain spatial or typed hold-pair relations are associated with difficulty in the training data.

**Korean**

Relation statistics는 training set에서만 계산한 hold-pair 패턴 통계입니다. 예를 들어 어떤 hold-to-hold 관계가 training data에서 어려운 문제에 자주 등장한다면, 그 관계는 difficulty와 관련된 높은 encoding 값을 가질 수 있습니다. 이 통계는 test leakage를 피하기 위해 train split에서만 계산합니다. 즉 특정 spatial relation이나 hold type 조합이 학습 데이터에서 어떤 난이도와 자주 연결되는지를 요약한 값입니다.

### 4. Can you explain BetaMove and GradeNet in more detail?

**English**

MoonBoardRNN uses a two-stage approach. First, BetaMove converts a hold configuration into a possible climbing move sequence. It uses hand assignment, hold difficulty, relative distances, and movement-related features to estimate plausible moves. Then GradeNet takes the generated sequence as input and predicts the difficulty grade using an LSTM. So MoonBoardRNN is strongly sequence-oriented: it first infers how a climber might move, and then classifies the difficulty from that move sequence.

**Korean**

MoonBoardRNN은 두 단계 구조를 사용합니다. 먼저 BetaMove가 hold configuration을 가능한 climbing move sequence로 변환합니다. 이 과정에서 손 배정, hold difficulty, 상대 거리, 이동 관련 feature 등을 이용해서 그럴듯한 move를 추정합니다. 그 다음 GradeNet은 이렇게 생성된 sequence를 입력으로 받아 LSTM을 통해 difficulty grade를 예측합니다. 따라서 MoonBoardRNN은 sequence-oriented한 접근입니다. 먼저 climber가 어떻게 움직일지를 추정하고, 그 move sequence를 기반으로 난이도를 분류합니다.

### 5. What exactly is included in the edge feature?

**English**

The edge feature describes the movement relation from one hold to another. It includes normalized relative displacement such as `dx` and `dy`, absolute horizontal and vertical distance, Euclidean distance, upward or downward movement indicators, whether the horizontal move is large, source and target hold role indicators, train-only relation encoding, relation confidence, geometric hardness, and a class-distribution summary for similar hold-pair relations. In short, the edge feature tells the model not only that two holds are connected, but also how they are connected.

**Korean**

Edge feature는 한 hold에서 다른 hold로 이동할 때의 관계를 설명합니다. 여기에는 normalized `dx`, `dy`, 절대 가로/세로 거리, Euclidean distance, 위쪽 또는 아래쪽 이동 여부, 가로 이동이 큰지 여부, source/target hold의 role 정보, train-only relation encoding, relation confidence, geometric hardness, 그리고 비슷한 hold-pair relation의 class distribution summary가 포함됩니다. 요약하면 edge feature는 두 hold가 연결되어 있다는 사실뿐 아니라, 어떤 방식으로 연결되어 있는지를 모델에 알려줍니다.

### 6. Why use macro-F1? What does it show better than accuracy?

**English**

Accuracy can be dominated by frequent grade classes. In the MoonBoard dataset, lower and middle grades have many more samples than rare high grades. A model can achieve reasonable accuracy by performing well on common classes while still doing poorly on rare classes. Macro-F1 computes F1 for each class separately and then averages them equally, so rare classes are not ignored. This makes macro-F1 useful for checking whether the model performs across the full grade distribution, not only on the majority classes.

**Korean**

Accuracy는 sample 수가 많은 class의 영향을 크게 받습니다. MoonBoard dataset에서는 낮은 난이도와 중간 난이도 class가 많고, 높은 난이도 class는 상대적으로 적습니다. 그래서 모델이 흔한 class만 잘 맞춰도 accuracy는 괜찮게 나올 수 있지만, rare class에서는 성능이 낮을 수 있습니다. Macro-F1은 각 class별 F1을 따로 계산한 뒤 동일한 비중으로 평균내기 때문에 rare class가 무시되지 않습니다. 따라서 macro-F1은 모델이 전체 grade distribution에서 얼마나 균형 있게 작동하는지를 확인하는 데 유용합니다.
