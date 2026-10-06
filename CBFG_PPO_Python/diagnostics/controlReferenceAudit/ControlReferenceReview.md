# [39]・[38]・[34]の照合と修正

追加3論文を読み、主論文と現在のコードを照合しました。CBFの診断値の返却漏れと、数値解法の終了状態・可行性の区別を修正しました。66項目のテストは全件成功しました。制御則・学習設定は変更していないため再学習は行わず、既存best policyを各速度50シナリオで診断評価しました。成功率は5 m/sで72%、10 m/sで28%。修正前の50件の評価CSV、および代表エピソードの従来の全軌跡・制御列は完全一致しました。

これにより論文性能を再現できた、または未記載パラメータが確定した、という結果ではありません。

## 各資料から確認できたこと

|資料|該当箇所|主論文実装への扱い|
|---|---|---|
|[39] Zeng, Zhang, Sreenath|PDF3ページ Eq.(6)、(9)、(10)、PDF4～5ページ Sec.III-C/D/E|離散CBFは h_next >= (1−γ)h、0<γ≤1。安全集合の議論は現在状態のh>=0等の前提が必要。一般に非凸NLPなのでsolver失敗は解不存在の証明ではない。joint MPC-CBFのhorizon制約・terminal条件は主論文の独立したMPC＋1-step補正へ移植しない|
|[38] Zeng, Li, Sreenath|PDF3～4ページ Eq.(12e)、Sec.III-A、PDF5ページ Sec.IV、PDF8ページ結論|可行性改善は h_next >= ω(1−γ)h、ω>=0という減衰率の最適化と追加costを伴う。主論文Eq.(25)にはωがないため導入しない。単純な加法slackでh_next>=0を緩める方法と同一視しない。recursive feasibilityも論文末尾で今後の理論検討として残っている|
|[34] Lubars et al.|PDF4～5ページ Sec.IV、Algorithm1、PDF7ページ実験設定|SUMO・DDPG・ST空間の計画と、予測rolloutの安全性・効率による切替。主論文のPPO教師変換やBC重みを補完する方法ではない。5 s/150 mの計画horizon、v*=30 m/sなども別実験の設定なので転用しない|

主論文はこれらを先行研究として引用しています。引用されていることは、先行研究の数値やアルゴリズムを主論文へそのまま継承すべき根拠にはなりません。

## 実際に修正したコード

対象プロジェクト：C:/Users/mura/Simulation/Research activities/M1_DecisionMaking/CBFG_PPO_Python

1. `cbf/safeController.py`：名目制御が可行ならearly returnで現在barrierと安全集合判定が返っていましたが、実際に最適化・補正した経路のreturnでは両フィールドを省略しており、NaNとFalseになっていました。失敗経路も含めて計算済みの値を返すよう修正しました。
2. 同ファイル：候補制御の制約検査によるsuccessと、optimizer_reported_success、used_feasibility_recoveryを分離しました。候補の状態はverified_feasible_with_tolerance / no_verified_feasible_candidateとして記録します。optimizerが成功を返しても制約違反があれば適用しません。optimizerが失敗でも検証済みの可行候補があれば既存どおり扱います。
3. `environment/mergeEnvironment.py`：上記をinfoへ出力。既存cbf_solver_successは従来どおり可行候補の判定のaliasとして保持し、optimizerのフラグとの違いを明記しました。cbf_infeasibility_provenはFalseです。探索は数学的な解不存在証明を実施していません。
4. `main/mainSimulation.py`：候補の制約充足、optimizer状態、再探索、現在barrier、安全集合判定をsimulationLog.csvへ保存します。

initial_safe_set_satisfiedは「そのCBF solveの開始状態」の判定です。エピソード開始時だけの判定でも、全軌跡の安全保証でもありません。初期集合条件とsolver残差を満たしても、採用CBF・道路解釈・予測モデルの妥当性、時刻間の接触まで保証したことにはなりません。

旧termination_reason='cbf_infeasible'は後方互換のため保持しました。その意味は「この数値探索で適用できる候補を得られなかった」です。「可行集合が数学的に空と証明した」という意味ではありません。過去CSVも書き換えていません。

コスト、CBF減衰率、制約、入力制限、最低速度、教師変換、reward、PPO更新、制御の適用判定は維持しています。

## 検証

66項目：既存55項目と追加11項目。補正経路の診断値、実道路境界付近での操舵補正、負の現在barrier、optimizer成功と可行性の独立、再探索の記録、名目可行時のoptimizer未実行、環境info、履歴policyのsource互換性を検証しました。

```powershell
python -m unittest diagnostics.testControlReferences diagnostics.testAdditionalReferences diagnostics.testReferencePapers diagnostics.testSingleLanePaper diagnostics.testImplementationFixes -q
```

既存v4 best policy、seed10000～10049、追加速度変動±1 m/sを各50回評価し、旧CSVと全項目を照合しました。

|最低速度|成功|候補制御を得られない失敗|衝突|修正前との一致|
|---|---:|---:|---:|---|
|5 m/s|36/50（72%）|13|1|全50件の評価値・代表軌跡の従来列が完全一致|
|10 m/s|14/50（28%）|35|1|全50件の評価値・代表軌跡の従来列が完全一致|

代表軌跡そのものにはCBF補正がないため、返却漏れの再現・回帰検証には実道路境界付近で補正を必要とする追加テストも使いました。

## 学習済みモデルとsource hash

保存済みモデルは旧v4の学習結果であり、書き換えていません。通常評価の厳密なsource hash検査は維持したため、診断修正前のモデルを現在のsourceと完全一致する学習済みモデルとしては受け付けません。

履歴モデルの検証用に `--diagnostic-source-update` を追加しました。今回の2ファイルの変更前・変更後hashが記録と完全一致するときだけ診断評価を許可し、actor等への未監査の変更があれば拒否します。専用output-dirを必須とし、manifestでdiagnostic_only=true、今回受け入れた変更を明示します。単純にすべてのsource不一致を無視するオプションではありません。

```powershell
python -m main.mainSimulation --model results/runs/reference_v4_speed5_seed0/models/bestPolicy.pt --speed 5 --episodes 50 --seed 10000 --torch-threads 1 --diagnostic-source-update --output-dir results/runs/control_reference_diagnostics_speed5_seed0
```

10 m/sはmodel・speed・output-dirを対応する10のものへ変更してください。新しく学習するpolicyは現在のsource hashを保存し、通常の厳密評価を使えます。

## 追加文献を読んでも残る仮定

- CBF補正から参照速度・headingへの教師変換、その補正の正規化座標と介入閾値δ。
- 主論文実験のMPC予測長・Q/R・wheelbase・solver設定。
- 主論文の楕円寸法、道路式を連結道路へ適用する解釈、数値探索手順。
- 行動・観測の物理範囲、欠測処理、SV数・配置・速度乱数の範囲。
- 数値的な成功判定、探索初期分散、有界policyの詳細、平均的なpolicyの選択手順。

CBFの1−γについては[37]に加えて[38][39]が根拠になりました。ただし主論文Eq.(25)の1+γとの不整合自体は残っています。文献照合で支持を増やした解釈であり、著者へ確認して確定したものではありません。

## 保存先

- 変更前：diagnostics/controlReferenceAudit/before
- テスト・stdout・hash：diagnostics/controlReferenceAudit
- 各速度の診断評価：results/runs/control_reference_diagnostics_speed5_seed0 と speed10_seed0
- 差分：changes.patch、検証：verification_manifest.json
- results/figures_paperは既存v4の学習データに基づく比較用図を保持。今回学習挙動は変わらず、描画元CSVと評価結果も一致したため、学習曲線の描き直しは不要です。診断版の図・動画は上記専用runへ保存しました。

資料PDFの実体とSHA256はverification_manifest.jsonに記録しました。
