# 曲面 Fairing：阶数、约束边界与失败语义的评估

分析对象：`src/pmp/algorithms/fairing.cpp`（`fair` / `minimize_area` / `minimize_curvature`）、`src/pmp/algorithms/laplace.cpp`（cotan Laplacian 与质量矩阵）、`src/pmp/algorithms/numerics.cpp`（`cholesky_solve`）、`src/pmp/algorithms/hole_filling.cpp`（内部调用者）、`tests/fairing_test.cpp`。

结论均标注源码位置或实测数据。实测方式：在仓库外（`/tmp/fairing-probe`）单独编译探针程序，仅链接本仓库 `Release` 构建出的 `libpmp`，用 `icosphere` / `plane` / `cone` 等生成器构造网格，直接读取 `laplace_matrix` / `mass_matrix` 的数值、约化矩阵特征值与求解结果。**默认构建下 `PMP_SCALAR_TYPE` 未设为 64，故 `Scalar = float`，顶点坐标是 float32，而矩阵与求解全程是 double**（`src/pmp/types.h:17-24`、`CMakeLists.txt:169-172`）——写回时截断为 float。

---

## 0. 一句话结论

`fair()` 是一个**以当前几何线性化的、只服从 Dirichlet 约束的单步 k 阶调和方程求解器**：它不保持体积、不保持特征、不迭代、不检查输入的合法性（垃圾索引、零面积面、闭合分量）。"求解成功"只意味着 `SimplicialLDLT` 没有遇到**恰好为零**的主元，与结果的几何有效性无关。

---

## 1. 算法与约束构造

### 1.1 方程

```
A X = 0 ,   A = L (M⁻¹ L)^(k-1)                     fairing.cpp:110-118
X(v) = 当前坐标,  v ∈ locked  (Dirichlet 约束)         numerics.cpp:72-76
```

- `L` 是 cotan Laplacian（`laplace_matrix`，`clamp` 默认 `false`，`laplace.h:48`）；`M` 是 lumped 混合 Voronoi / 多边形质量矩阵（`mass_matrix`）。`B` 恒为精确零（`fairing.cpp:102-104`，`fairing.cpp:118` 的 `B = M*B` 是数学上的 no-op）。
- 因此右端项只在消去约束时出现：`BB -= A(i,j) * C(j)`（`numerics.cpp:74`）。**解完全由"约束如何被算子耦合进来"决定**，没有任何目标函数里的体积/面积项。
- `A` 的符号随 k 交替：`L` 是负半定（文档 `laplace.h:39` 亦是此意），故 k=1、3、5 的 `A` 是**负**半定，k=2、4、6 是正半定。实测（holed icosphere，`nv=521`）：

| k | 约化矩阵最小/最大特征值 | 定性 | 条件数 |
|---|---|---|---|
| 1 | −5.27 / −1.81e-2 | NSD | 2.9e2 |
| 2 | +0.100 / +1.89e3 | PSD | 1.9e4 |
| 3 | −7.00e5 / −1.272 | NSD | 5.5e5 |
| 4 | +18.2 / +2.60e8 | PSD | 1.4e7 |
| 5 | −9.64e10 / −360.8 | NSD | 2.7e8 |
| 6 | +9.39e3 / +3.58e13 | PSD | 3.8e9 |

`numerics.h:23-24` 要求 "sparse, symmetric, positive definite"；对称性成立（实测非对称量级 ~1e-17·max|A|），**正定性对奇数 k 不成立**。`SimplicialLDLT` 只对零主元报错，对负定不报错，所以奇数 k 能跑通，但"最小化能量"的解释要取反号。

`A(k=2) = L M⁻¹ L` 即薄板/双调和算子（曲率极小），`A(k=3)` 对应曲率变化极小。阶数越高，`A` 的带宽与元素动态范围越大（实测 max|A|：k=2 约 1.9e3，k=4 约 2.6e8，k=6 约 3.6e13），这是 `L (M⁻¹L)^(k-1)` 的内在性质，不是 bug。

### 1.2 单步、非幂等

矩阵在**输入几何**上构建一次，解出后直接写回（`fairing.cpp:122-125`），不解算、不迭代。实测（8×8 交替边界高度的三角网格）：k=1 时 `max|A_input·x_solved|` 在自由集上为 3.8e-7（线性系统确实被精确解掉），但对**同一网格再调用一次**，顶点仍会移动 0.023。也就是说：

- `fair(m,2)` 对闭合球+选区反复施加 5 次：体积 4.153 → 4.036（收敛），面积 12.507 → 12.287，无翻面。**收敛但非一次到位**；等价于"用当前几何的算子做一步隐式平滑"。

---

## 2. cotangent 权重

### 2.1 定义与符号

`triangle_laplace_matrix`（`laplace.cpp:175-244`）用海伦公式求面积后取 `cot[i] = 0.25·(…)/area`，装配成 `L(i,i)=Σcot, L(i,j)=−cot`（`laplace.cpp:204-209`）。符号约定正确：`L·1 ≈ 0`（实测 `max|L·1| = 1e-15`，常数精度成立），而 `L·x ≠ 0`（实测 `max|L·x| ≈ 0.04`）正是离散平均曲率法向，符合预期。

### 2.2 钝角与负权：`clamp` 在 fairing 路径上未启用

- `laplace_matrix(mesh, L)` 在 `fair` 中被以默认 `clamp=false` 调用（`fairing.cpp:111`）。`explicit_smoothing` 显式传 `clamp=true`（`smoothing.cpp:22`），`implicit_smoothing` 同样用默认的 `false`（`smoothing.cpp:73`）。所以准确的说法是：**fairing 与显式平滑在这一点上不同，与隐式平滑则相同**；fairing 自己保留负的 cotan 非对角元，且因为求解 `A = L(M⁻¹L)^(k−1)`，负权会被反复放大，影响比 `implicit_smoothing` 的 `M − tL` 更大。
- 实测单个三角形：锐角三角形非对角元全正（+0.311/+0.278）；把一个顶点压到几乎共线（极钝角）后出现 **2 个负非对角元**（−125），矩阵仍半定但已从 M 矩阵向病态方向退化。`L·x ≠ 0` 的偏差也随之放大。
- 结论：**fairing 的解在钝角/狭长三角形上是"无约束的能量极小"，不是"几何上合理的平滑"**。若要面积/曲率意义上的稳定行为，应传 `clamp=true`（或调用前先 remesh）。

### 2.3 退化三角形与退化多边形：两条路径行为完全不同

- **退化三角形（n=3）**：`triangle_laplace_matrix` 在 `area <= min` 时**完全跳过赋值**（`laplace.cpp:197-210` 只有 `if`，无 `else`），但调用方 `polygon_laplace_matrix` 已先 `Lpoly = Zero(n,n)`（`laplace.cpp:249`），所以结果是干净的全零局部矩阵 → 该面对 `L` 无贡献。
- **退化多边形（n>3）**：`polygon_laplace_matrix` 里 `DenseMatrix Ltri(3,3);` 声明在**循环外**（`laplace.cpp:266`），每个扇区三角形调用 `triangle_laplace_matrix`。当某扇区退化时该函数不写 `Ltri`，于是循环**读到未初始化或上一个扇区的残留值**，并被累加进 `Lfan` 与全局 `L`（`laplace.cpp:272-284`）。

实测证据（单个共线四点四边形，`(10,0,0) (11,0,0) (12,0,0) (13,0,0)`）：

```
默认堆内容      : sum|L| = 52      , max|行和| = 13
ASAN malloc 填 0 : sum|L| = 2.8e-309, max|行和| = 6.9e-310
```

正确的 Laplacian 在共线点上必须**恒为 0**。两者都不为 0，且随堆内容变化 → 确认是**未初始化内存读取**。同理 `polygon_mass_matrix` 走同一个 `compute_virtual_vertex` 路径，四个权重解得 (0.25,0.25,0.25,0.25)、虚拟顶点落在直线上，质量矩阵该面为 0（`triangle_mass_matrix` 有 `Mtri.setZero(3)`，`laplace.cpp:91-95`，质量侧是安全的）。

**后果**：`L` 有非零元素但 `M` 为 0 → `M⁻¹` 的对应位置为 `+inf` → `L·M⁻¹·A` 中出现 `0×inf = NaN`。实测把 `plane(4)` 压成一条线后：k=2 的 `A` 有 **361/361 个元素为 NaN**，但 `SimplicialLDLT` 仍返回 `Success`（见 §5.2）。

---

## 3. 边界与锁定区域

### 3.1 锁定规则

`fairing.cpp:40-80`：

| k | 锁定集合 |
|---|---|
| 1 | 边界顶点 |
| 2 | 边界 + 1-ring |
| ≥3 | 边界 + 1-ring + 2-ring |

**代码没有 `k > 3` 的分支**，所以 k=3、4、5、6… 的约束集合完全相同。实测（带一个洞的 icosphere，`nv=521`）：

```
k=1 locked= 48 free=473
k=2 locked= 90 free=431
k=3 locked=134 free=387
k=4 locked=134 free=387
k=5 locked=134 free=387
k=6 locked=134 free=387
```

### 3.2 阶数与锁定环数不匹配 → k≥4 的结果不可信

算子阶数为 k，而约束带宽固定在 3 环。k 增大时 `A` 的支撑向自由区内部延伸，约束却不再加强。实测（同上网格，原始面积 9.698，原始体积 3.220）：

| k | 面积 | 面积变化 | 内部最大位移 |
|---|---|---|---|
| 1 | 2.351 | −75.8% | 1.52 |
| 2 | 6.017 | −38.0% | 0.731 |
| 3 | 9.107 | −6.1% | 0.140 |
| 4 | 6.849 | −29.4% | 1.94 |
| 5 | 22.87 | +135.8% | 2.33 |
| 6 | 1138 | +11632% | 43.1 |

k=3 最优，k≥4 单调恶化。**对照实验**（同一网格，只改锁定环数为 k 环，其余代码照抄 `fair`）：

```
 k=4  3 环锁定: maxdisp=1.943  area= 6.849      k 环锁定: maxdisp=1.07e-2  area= 9.663
 k=5  3 环锁定: maxdisp=2.333  area=22.871      k 环锁定: maxdisp=1.72e-3  area= 9.703
 k=6  3 环锁定: maxdisp=43.06  area=1137.8      k 环锁定: maxdisp=1.12e0   area= 8.511
```

→ **k≥4 的爆炸可直接归因于约束带宽被截断在 3 环**，把环数改成 k 后 k=4、5 立刻恢复（k=6 仍不稳，说明 6 环也未必够）。这也解释了为什么 `examples/fairing.cpp:75-87` 的 "Minimize Curvature Variation"（k=3）是安全的，而文档里没有 k>3 的推荐用法。

### 3.3 边界是**逐位精确固定**的

约束行 `C` 直接来自输入坐标，解出后原样写回（`numerics.cpp:101-103`）。实测 holed icosphere 在 k=1/2/3 下**边界顶点位移恒为 0**，内部位移 1.52 / 0.731 / 0.140。所以"边界只有边界值被尊重，边界上的法向/切向形状完全不被保留"——边界环上的折角、尖锐转折会被内部的调和延拓"记住"，但内部的尖锐特征（曲率突变、硬边）**没有任何保持机制**：`fair` 不读 `v:feature`/`e:feature`（全仓库只有 decimation/remeshing 读，`decimation.cpp:538-539`、`remeshing.cpp:326-327`）。

### 3.4 两类静默 no-op

- **粗网格上 k=3 锁满全网格**：`open_cone` + 一次 Loop 细分（`nv=25`）时 k=3 锁定 25/25，`fair` 正常返回、网格**一位不差**（`maxdisp=0`）。"成功"在这里等于"什么都没做"。
- **k=2 只剩 1 个自由顶点**：同一网格 k=2 锁定 24/25，结果退化为单点更新（`maxdisp=0.265`）。这与 `tests/fairing_test.cpp:13-21` 的网格相同——该测试用 `EXPECT_LT(bbs, bbz)`（最大 z 变小）作为成功判据，而这个"变小"恰恰是尖端被拉到基平面（k=1 时 max z 从 1.019 → **0**，面积 4.276 → 2.621）。

---

## 4. 孤立分量

这是最危险的区域。`fair()` 用**一个全局矩阵 + 一个全局锁定掩码**，从不检查连通性。

### 4.1 闭合分量（无边界）→ 静默塌缩到原点

闭合分量没有任何边界顶点，因而不产生任何锁定。而 `fairing.cpp:82-97` 的 `something_locked` 只要求**整个网格中存在一个**锁定顶点——所以只要网格别处有开边界，闭合分量的奇异性就完全不会被发现。

实测：`plane(4)`（提供约束）+ 平移后的 `icosphere(2)`（闭合分量），`fair(m,2)`：

```
sphere bbox before: [9,11]^3
sphere bbox after : [0,0]^3
sphere volume: 4.04768 -> 0
```

球体所有顶点坐标**精确变为 0**（实测 `max|coord| == 0.0`，非零坐标 0/486）。机制清楚：该块的右端项恒为 0（`B` 为零，且约束行不作用于该块），`LDLT` 对 `0/微小主元` 的前代/回代在浮点上保持精确零 → 得到该块的**零解**。而 0 是零空间（常数 + 线性函数）中*一个*解，于是几何被彻底擦除。

同一现象用 `tetrahedron` 复现（k=1/2/3 一致，均塌缩到 `[0,0]`）。注意此时：

- **不抛异常**；
- 拓扑仍然合法（`nv`、`nf` 不变）；
- `analyze()` 只报告 `degenerate_faces`，没有"分量被压扁"的信号；
- 塌缩后 `volume()` 抛 `InvalidInputException`（`differential_geometry.cpp:129-135` 要求三角网格，而合并后的网格含四边形面），这是**下游**才发现的后果。

### 4.2 孤立顶点 = 假约束

`fairing.cpp:76-79` 把孤立顶点锁为约束。这使得**闭合网格只因多了一个孤立顶点就通过了 `something_locked` 检查**：

```
icosphere(2) + 1 个孤立顶点 (0,0,10), nv=163
fair(m,2) 返回 OK
volume 4.04768 -> 0        area 12.33 -> 0
321/321 个面退化为零面积
```

孤立顶点自身位置不变（`(0,0,10)`，因为它被锁定），但其`locked`标志**不能**为任何分量提供有效的边界约束。这是 §4.1 的一个更容易误触发的入口：读入带游离点的模型、或预处理后残留一个顶点，就会让整个闭合壳层被抹掉。

`DTO` 层面同样存在问题：`is_isolated(v)`（`surface_mesh.h:1266`）只判断 `!halfedge(v).is_valid()`，不区分"孤立点"与"退化片"。真正需要的是**每个连通分量至少有一个非退化约束**。

### 4.3 分量之间的交叉污染

`fairing.cpp:68-75`：只要 `v:selected` 存在且**至少一个**顶点被选中（`no_selection=false`），所有未选中的顶点都会被锁定——**包括其它分量上的顶点**。实测：两个不相连的闭合球，只选球 A 的上半部 → 球 B 全部被锁，保持不动，而球 A 正常 fairing。这在"选择即约束"的语义下是自洽的，但它意味着**选区语义与分量语义纠缠**：无法用 `v:selected` 表达"只 fair 这个分量、其它分量也别动边界"。

---

## 5. 奇异系统

### 5.1 求解器只认一种失败：主元恰好为 0

`cholesky_solve`（`numerics.cpp:83-98`）只在 `solver.info() != Success` 时抛 `SolverException`，而 Eigen 的 `SimplicialLDLT` 只在

```cpp
m_diag[k] = d;
if (d == RealScalar(0)) { ok = false; break; }      // SimplicialCholesky_impl.h:365-370
```

时置 `NumericalIssue`。注意：

- 没检查 `std::isfinite(d)`；
- 没检查对称性、正定性、条件数、残差；
- `!= Success` 时 `_solve_impl` 直接 `return`，`dest` 保持未变（`SimplicialCholesky.h:155`），所以**"失败"和"未求解"不可区分**。

### 5.2 NaN 让失败判定失效 → 静默写入 NaN

`NaN == 0` 为 false，所以含 NaN 的分解一律报 `Success`。实测把 `plane(4)` 压成一条线（16 个面全部退化，走 §2.3 的多边形路径）：

```
k=1: 返回 OK，NaN 顶点 0 个
k=2: 返回 OK，NaN 顶点 1 个（v12），surface_area 变为 NaN
k=3: 返回 OK，NaN 顶点 0 个（全被锁定，no-op）
```

再调用一次 `fair(k=1)` 后 NaN 顶点从 1 扩散到 9。**几何被不可逆污染，且没有任何异常**。这与 §5.1 的"`dest` 未被修改"形成尖锐对比：真正的奇异会安全地保留网格，而"伪成功"会写坏它。

### 5.3 零自由度与全锁定

- 全网格锁定（粗网格 k≥3、单三角形、边界上的所有顶点）：`numerics.cpp:34-40` 的 `n_constraints` 统计到全部顶点，`n=0`，`AA` 为 0×0，`compute`/`solve` 对空矩阵返回 `Success`，`X` 全部取自 `C`。**结果正确（恒等映射），但白花一次 O(n³·带宽) 的构建 + 分解**。
- `A` 的构建与 `L`/`M` 的 O(nnz) 在锁满时依然执行（`fairing.cpp:110-118`），没有短路。

### 5.4 无法被约束修复的块结构

`cholesky_solve(A,B,is_constrained,C)` 的约化只管"删掉约束行列、把约束项挪到右端"。若某个自由块与所有约束**完全不连通**，其约化矩阵必然奇异（常数或线性函数在零空间），而 `d == 0` 的检查在浮点上**几乎不会恰好命中**（实测参见 §4.1：塌缩时未报错）。也就是说，**"奇异系统"在本实现里通常不表现为异常，而表现为零解/巨解**。

---

## 6. 失败后的网格状态

按失败类型分开说明，全部有实测支撑。

### 6.1 `InvalidInputException`（缺少约束）：干净

`fairing.cpp:92-97` 先 `cleanup()` 再 `throw`。实测：闭合 icosphere 无选区 → 抛 `fair: Missing boundary constraints.`，`maxdisp = 0`，`fairing:locked` 属性已移除。可重复调用，状态一致。（`v:selected` 存在但全为 false 时同样走这条路径——见 `fairing.cpp:27-38` 的 `no_selection` 判定。）

### 6.2 `SolverException`（分解失败）：几何干净，**但属性泄漏**

`cleanup` 只在 `fairing.cpp:94`（缺约束）和 `fairing.cpp:128`（成功）调用。第 122 行的 `cholesky_solve` 抛出的异常**没有 catch**，于是 `cleanup()` 被跳过，`fairing:locked` 留在网格上。实测（共线扇面，k=1）：

```
第 1 次调用: THREW SolverException: cholesky_solve: Failed to factorize linear system.
             位置不变 maxdisp=0 ; has fairing:locked = 1   ← 泄漏
第 2 次调用: THREW InvalidInputException: [PropertyContainer] A property with name
             "fairing:locked" already exists.              ← 语义完全错乱的报错
第 3 次调用: 同上
手动 remove_vertex_property 后: fair(m,2) 正常返回
```

**这是最恶劣的失败状态**：第一次失败是合理的（网格确实奇异），但之后每一次调用都会因为**上一次失败的残留属性**而失败，报错信息还是"属性已存在"这种与几何无关的消息。网格对 `fair()` **永久不可用**，直到调用方手工按名字删除 `fairing:locked`。`cholesky_solve` 的另一个重载（`numerics.cpp:8-27`）没有这个问题，泄漏只发生在带约束的路径上。

### 6.3 成功但写入 NaN：不可逆

见 §5.2。网格拓扑不变、`nv`/`nf` 不变、`analyze()` 仍能正常遍历，但顶点坐标是 NaN，`surface_area`/`bounds`/后续 `fair` 全部被污染且**逐次扩大**。没有任何备份，也没有回滚路径。

### 6.4 成功但几何被摧毁

- 闭合分量塌缩到原点（§4.1）：坐标精确为 0，拓扑保留，实体消失。
- k=1 的 holed icosphere：带符号体积从 +3.220 变为 **−0.341**，即**整体内外翻转**，并有 8 对相邻面法向相反（折叠）；面积缩到 24%。这是"求解成功"的又一种形态。
- k≥4：面积放大到 117×–10000×，`bounds` 从 `[−1,1]` 涨到 `[−6,6]`。

### 6.5 未消费的垃圾（deleted 但未 `garbage_collection`）：崩溃

`fair()` 内部一律用 `v.idx()` 写 `n_vertices()` 大小的容器：

- `coordinates_to_matrix`：`X.resize(mesh.n_vertices(), 3)` 然后 `X.row(v.idx()) = ...`（`numerics.cpp:167-172`）；
- `laplace_matrix`：`L.resize(nv, nv)` 然后按 `vertices[k].idx()` 生成三元组（`laplace.cpp:519-527`）。

`n_vertices() = vertices_size() − deleted_vertices_`（`surface_mesh.h:1193`），所以有已删除顶点时 `v.idx()` 可以 ≥ `n_vertices()`。实测删除一个顶点（不回收）后调用 `fair(m,1)`：

```
nv=641 size=642
Debug 构建: Assertion failed: it->row() < mat.rows() , SparseMatrix.h:1148
Release(NDEBUG) 构建: exit=138 (SIGBUS)，3/3 次可复现
```

正如 `docs/tutorial.md:188` 所述，`garbage_collection()` 是删除元素后的**强前置条件**；但 `fair()` 既不检查也不在文档中声明这一点（`fairing.h:22-28` 只列了两种异常），违反者的代价是**进程崩溃而不是异常**。注意只有**删除顶点**会触发；只删面不回收是安全的（实测 `fair` 正常返回）。

### 6.6 状态总结表

| 触发条件 | 异常 | 顶点坐标 | 属性 | 可恢复性 |
|---|---|---|---|---|
| 无任何约束 | `InvalidInputException` | 不变 | 干净 | 完全可恢复 |
| 分解失败 | `SolverException` | 不变 | **泄漏 `fairing:locked`** | 需手动删属性 |
| 退化多边形 / 零质量 | **无** | **NaN** | 干净 | 不可恢复（须外部备份） |
| 闭合分量缺约束 | **无** | 塌缩到原点 | 干净 | 不可恢复 |
| 钝角/高阶病态 | **无** | 可用但失真 | 干净 | — |
| 有未回收的删除顶点 | **崩溃/断言** | — | — | 进程级 |

---

## 7. 平滑成功 ≠ 体积或特征保持

### 7.1 目标函数里没有体积项

`fair` 解 `A X = 0`，`A` 只由 k 阶算子与几何决定；`B ≡ 0`。所以任何"体积"、"面积"、"特征"都**不是**残差的一部分，也无法通过收敛来保证。k 阶能量的极小化在数学上本就**允许**（甚至倾向于）收缩：

- k=1（`minimize_area`）：调和/极小曲面能量，固定边界下面积会严格减小 → 体积必然减小。
- k=2（`minimize_curvature`）：薄板能量 `∫|Δx|²`，惩罚曲率而不惩罚尺度，但仍然只受边界约束，内部没有任何"回到原曲面"的项。
- k=3：曲率变化能量，进一步"软化"，收缩更少（实测 −6.1% 而 k=2 是 −38.0%）。

### 7.2 实测：全部收缩

闭合 icosphere(3) + 选区（上半球选为 free），原始面积 12.506 / 体积 4.153：

| k | 面积变化 | 体积变化 |
|---|---|---|
| 1 | −6.28% | −15.58% |
| 2 | −1.41% | −2.22% |
| 3 | −0.11% | −0.17% |

对 holed icosphere（自由边界）：

| k | 面积变化 | 体积变化 | 朝向 |
|---|---|---|---|
| 1 | −75.8% | −89.4% | **翻转**（+3.22 → −0.34），8 对折叠面 |
| 2 | −38.0% | −56.6% | 一致 |
| 3 | −6.1% | −9.2% | 一致 |
| 4 | −29.4% | −93.5% | 一致但不可信 |
| 5 | +135.8% | +108.1% | 一致但不可信 |
| 6 | +11632% | +72553% | 一致但不可信 |

所以：

- **平滑"成功"（不抛异常、系统被解掉）与体积保持是两件独立的事**，二者之间没有蕴含关系。本实现的 fairing 在正常参数下**总是**收缩，且收缩量随 k 增大而**不单调**（k=3 最好，因为约束环与阶数在这里最"匹配"，纯属巧合）。
- 想让体积不变，本库没有任何开关。`implicit_smoothing(..., rescale=true)`（`smoothing.cpp:43-51, 96-110`）提供的是**面积 + 质心**恢复，不是体积；而且它只对平滑有效，对 fairing 无效，因为 `fair` 里没有对应的后处理。
- **特征保持完全没有**：不读 `v:feature`/`e:feature`，不给边界加切向约束，不保留硬边。尖锐特征只有在"恰好位于锁定环内"时才被保留；落在自由区的特征会被平滑掉。
- **退化/折叠没有防护**：k=1 会把曲面内外翻转（带符号体积变号），`fair` 既不检测也不报告。下游 `volume()` 返回绝对值（`differential_geometry.cpp:147` 的 `std::abs`），所以这种翻转对 `volume()` **完全不可见**——这是"看起来体积正常"的一个陷阱。

### 7.3 现有测试把"收缩"当作"成功"

`tests/fairing_test.cpp:13-21` 断言 `bounds(mesh).max()[2]` 变小；`23-39` 断言包围盒对角线变小。在 `open_cone` 上，这个"变小"来自尖端被拉平到基平面（max z 1.019 → 0）。**测试锁定的是收缩行为本身，而不是几何质量**——任何进一步的实现改动只要还能收缩就能通过，反之亦然。

---

## 8. 结论清单

**阶数**

1. k 决定方程：k=1 调和（极小曲面/面积），k=2 双调和（薄板/曲率），k≥3 高阶。`A = L(M⁻¹L)^(k−1)` 在输入几何上只线性化一次，**不是迭代投影、不幂等**。
2. 奇数 k 的 `A` 是负半定，与 `numerics.h:23-24` 的正定前置条件不符；`LDLT` 不报错，但"极小能量"的符号要取反。
3. 锁定环数被硬编码为最多 3 环（`fairing.cpp:40-66`），而阶数 k 无上界。**k≥4 时约束带宽与算子阶数失配，结果从"可用"跳变为"垃圾"**；对照实验证明改为 k 环可修复 k=4、5。

**cotangent 权重**

4. 权重的常数精度正确，但 `clamp=false`（与 `smoothing.cpp` 相反），钝角/狭长三角形会产生负非对角元，使 fairing 失稳。
5. 退化三角形安全（局部矩阵已预清零）；**退化多边形（n>3）不安全——`Ltri` 在循环外声明且退化时不赋值，读未初始化内存**（实测 `sum|L|` 随堆填充在 52 与 2.8e−309 之间变化）。这是本报告中唯一确认的**库级内存 bug**。

**边界**

6. 边界被逐位固定；但边界之外的锁定带宽固定为 2 环，与 k 无关。
7. 粗网格上 k=3 可能锁满全网格 → 静默 no-op 且返回成功；粗网格 k=2 可能只剩 1 个自由顶点。

**孤立分量**

8. 闭合分量没有任何约束 → 约化矩阵奇异 → 右端为零使求解器给出**精确零解**，整个分量塌缩到原点，不抛异常、拓扑保留。
9. 一个孤立顶点就足以让 `something_locked` 通过，从而触发上述塌缩。
10. `v:selected` 的"未选中即锁定"跨越分量边界，无法表达"分量级"约束。

**奇异系统**

11. 只检测 `主元 == 0`；NaN 使检测失效（`NaN == 0` 为 false）→ **静默写入 NaN 并逐次扩散**。
12. 与约束不连通的自由块几乎从不产生恰好为零的主元，因此"奇异"在本实现里通常表现为零解或巨解，而非异常。

**失败后的网格状态**

13. 缺约束：干净抛异常，几何与属性均无损。
14. 求解失败：几何无损（坐标在求解成功后才写回），但 **`fairing:locked` 泄漏，使该网格对 `fair()` 永久不可用**，后续报错为 `property already exists`。
15. 退化输入：几何被 NaN 污染，不可恢复。
16. 闭合分量：几何被塌缩，不可恢复。
17. 未回收的删除顶点：**崩溃**（Debug 断言 / Release SIGBUS），不是异常。

**平滑成功 vs 体积/特征保持**

18. 二者无蕴含关系。`fair` 的目标里没有体积或面积项，正常参数下**总是收缩**（k=1 可达 −76% 面积、−89% 体积，并可整体翻转）；k=3 的"低收缩"是约束与阶数偶然匹配，不是保持机制。
19. 特征保持完全缺失（不读 `v:feature`/`e:feature`，无切向约束）。
20. 体积翻转对 `volume()` 不可见（其内部取 `std::abs`），容易被误判为"体积保持良好"。

---

## 9. 可操作建议（按优先级）

1. **调用前做守卫**：`mesh.garbage_collection()`；对每个连通分量检查是否存在有效约束；拒绝包含退化面（`analyze().n_degenerate_faces > 0`）的输入。本库不做这些。
2. **不要对含闭合分量的网格直接调用** `fair`（即使别处有开边界）；先拆分分量，只对目标分量调用。
3. **不要在奇数 k 之外还指望"最小化"语义**：k=1 会内外翻转，先验证带符号体积。
4. **k 取 2 或 3**；若必须用 k≥4，先自行锁定 k 环（把 `fairing.cpp:40-66` 的 3 环上限改为 k 环，实测能修复 k=4、5）。
5. **对结果做后验检查**：有限性（NaN/Inf）、带符号体积符号、相邻面法向一致性（折叠对）、最小面积；`fair` 自身一项都不做。
6. **失败处理必须手工清理 `fairing:locked`**：
   ```cpp
   try { pmp::fair(mesh, k); }
   catch (const pmp::SolverException&) {
       if (auto p = mesh.get_vertex_property<bool>("fairing:locked"))
           mesh.remove_vertex_property(p);
   }
   ```
7. **需要体积/特征保持**时，`fair` 不合适：用带约束的优化（显式加入体积/特征项）或 `implicit_smoothing(..., rescale=true)`（仅面积+质心，非体积），并在特征边上加锁定。
8. **不要依赖 `fair` 的返回值/异常来判定有效性**——它只在两种情况下报错，且其中一种会被上一次的残留属性误触发。

---

## 附录：复现要点

所有数据由仓库外独立探针产生（`Release` 构建，仅链接 `libpmp`；顶点为 float32，矩阵为 double）。核心复现片段：

```cpp
// §4.1/§4.2 闭合分量塌缩
auto m = pmp::icosphere(2);
m.add_vertex(pmp::Point(0,0,10));      // 一个孤立顶点即可通过约束检查
pmp::fair(m, 2);                        // 返回 OK
// m 的所有坐标 => 0 ; volume/area => 0

// §5.2 静默 NaN
auto m2 = pmp::plane(4);
for (auto v : m2.vertices()) m2.position(v)[1] = 0;   // 把四边形全压成一条线
pmp::fair(m2, 2);                       // 返回 OK，但 v12 为 NaN

// §2.3 未初始化读取（退化多边形）
//   collinear quad (10,0,0)..(13,0,0)：真实 L 应为 0
//   默认堆: sum|L| = 52 ; ASAN malloc_fill_byte=0: sum|L| ≈ 2.8e-309

// §4/§7.2 带符号体积检测翻转
//   holed icosphere, k=1:  signed_volume +3.220 -> -0.341
```

`ASAN_OPTIONS=max_malloc_fill_size=1000000:malloc_fill_byte=0` 用于暴露未初始化读取；`NDEBUG` 构建用于复现垃圾索引导致的 `SIGBUS`（Debug 下则是 Eigen 的 `set_from_triplets` 断言）。
