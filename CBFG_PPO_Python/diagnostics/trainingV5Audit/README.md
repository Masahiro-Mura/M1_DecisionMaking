# PPO確率計算とエピソード更新の修正（v5）

依頼された2点を実装しました。150000 stepの本学習は実行していません。旧モデル・旧結果は保持しています。

## 確率計算

Actorからtanh前のサンプルzも取得し、ActionResult、RolloutBuffer、minibatchまで保持します。PPO更新では保存したzで旧・新policyの確率を評価します。float32でtanh(z)が±1に丸められても、atanhで復元せず、元のzを使います。

log-Jacobianは数値的に安定な

```text
log(1 - tanh(z)^2) = 2 * (log(2) - z - softplus(-2*z))
```

で計算します。未更新policyで確率比が1になること、変更後policyの比が潜在Gaussianの比と一致すること、飽和時も値と勾配が有限になることを検証しました。旧2戻り値のActor.sample APIは保持していますが、実際の学習経路は必ず元サンプルを保存します。元サンプルのない飽和行動をbufferへ入れるとエラーにします。

## 更新タイミングと容量

- 主論文Algorithm 1に合わせ、各エピソードを集めてから40 epoch更新し、bufferをclearします。3072件に達するまで待つ処理は除去しました。
- Table IIIの3072は最大容量と解釈します。dt=0.2 s、40/20 sのシナリオでは通常最大200/100件なので容量に収まります。
- Algorithm 1と容量の関係は本文に明示されていないため、この解釈は記録した実装判断です。
- 学習予算で途中終了する最後の部分エピソードも一度更新します。これは有限step予算の扱いとして明示した例外です。その部分エピソードを完了した失敗として成功率へ計上しません。
- minibatch=128は上限です。エピソードが128 step未満なら小さいbatchで更新します。
- buffer容量を超える設定に変更した場合、途中で勝手に更新方式を切り替えずエラーにします。

更新回数が増えます。現行のbest policy選択は各更新後に固定10シナリオを評価するため、選択の評価回数も増え、旧3072-step方式より実行時間が長くなる可能性があります。この選択方式は今回の2点以外の既存仮定として維持しています。

## モデルの互換性

モデル識別はcbfg-ppo-episode-latent-v5、schema=5です。元サンプル使用、更新方式、容量解釈をmetadataへ保存し、main/mainLearning.pyのhashも記録します。旧v4をv5の学習結果として扱ったり、旧モデルからresumeしたりすることは互換性検査で拒否します。旧モデルは別途履歴として残っています。旧診断用source-updateオプションも、今回のアルゴリズム変更を許可しません。

## 実行方法

PowerShellでプロジェクトへ移動し、次を実行してください。通常のPython環境に必要な依存が入っていることを前提にしています。

```powershell
Set-Location -LiteralPath 'C:\Users\mura\Simulation\Research activities\M1_DecisionMaking\CBFG_PPO_Python'
python -m main.mainLearning --speed 5 --total-steps 150000 --seed 0 --torch-threads 1 --output-dir results/runs/training_v5_speed5_seed0
```

学習が成功して終了したら、対応する新モデルでシミュレーションします。

```powershell
python -m main.mainSimulation --model results/runs/training_v5_speed5_seed0/models/bestPolicy.pt --speed 5 --episodes 50 --seed 10000 --torch-threads 1 --output-dir results/runs/training_v5_speed5_seed0
```

10 m/sも新規学習する場合：

```powershell
python -m main.mainLearning --speed 10 --total-steps 150000 --seed 0 --torch-threads 1 --output-dir results/runs/training_v5_speed10_seed0
python -m main.mainSimulation --model results/runs/training_v5_speed10_seed0/models/bestPolicy.pt --speed 10 --episodes 50 --seed 10000 --torch-threads 1 --output-dir results/runs/training_v5_speed10_seed0
```

各学習の終了成功を確認してから次のシミュレーションを実行してください。--resumeや--diagnostic-source-updateは使用しません。

両速度の学習・評価完了後、主論文形式の図を新結果から作成：

```powershell
python -m main.mainPaperFigures --manifest results/figures_paper/input_v5.json
```

manifestなしの描画は以前のv4 runを使用するので、v5では上記manifestを指定してください。figures_paperの比較図は再生成時にv5結果で更新されます。

## 検証

72項目が全件成功しました。新規6項目は、飽和時の確率比・勾配、独立したGaussian比との照合、bufferの元サンプル保存、欠損元サンプルの拒否、sample API互換性、複数エピソードと最終部分エピソードの更新順序です。新モデルを保存し、厳密なsource互換性検査付きでロードすることも確認しました。実環境8-stepの既存smoke testも成功しています。性能の改善や論文結果の再現を確認したテストではありません。

```powershell
python -m unittest diagnostics.testTrainingV5 diagnostics.testControlReferences diagnostics.testAdditionalReferences diagnostics.testReferencePapers diagnostics.testSingleLanePaper diagnostics.testImplementationFixes -q
```

変更前はdiagnostics/trainingV5Audit/before、検証ログはdiagnostics/trainingV5Audit/verification.txtに保存しました。変更差分と実行手順のコピーはこの出力フォルダにも保存しています。
