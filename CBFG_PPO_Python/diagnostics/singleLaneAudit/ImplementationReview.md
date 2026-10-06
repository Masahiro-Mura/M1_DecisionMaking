# 論文実装の確認と本線1車線化（2026-10-04）

本線1車線への修正と、主論文だけで確定できる成功率集計の修正を実施しました。26項目の検証は全件成功しています。論文全体の完全再現、1車線での15万step再学習、学習済みpolicyによる新しい評価グラフ・動画の生成は、必要な引用論文の確認後に継続する作業です。短時間の動作確認を本学習の代わりとは扱っていません。

## 1. 今回修正した箇所

|対象|修正内容|根拠|
|---|---|---|
|parameters/environmentParameters.py|本線中心を `(4.0,)`、本線IDを `(0,)` に変更。ランプ中心8 mは維持|Fig.5、IV-B：本線 y=2～6 m、ランプ y=6～10 m|
|scenario/mergeRoad.py|各本線区間を1車線に変更。並行ランプのindexは2から1へ。本線の両外側は連続線、合流側だけ破線|Fig.5、IV-B。HighwayEnv標準の2車線をそのまま使うと対象シナリオと異なる|
|scenario/mergeScenario.py|ランプIDについての説明を更新。既存の定数参照により周辺車両の配置先は本線1車線に限定される|道路変更との整合|
|parameters/rlParameters.py|成功率の集計範囲3000 stepを明示|IV-A|
|main/mainLearning.py|直近3000 stepで終了したepisodeの成功割合を計算し、その系列を幅10で平滑化。学習予算で途中終了したepisodeは成功率・episode reward図から除外|IV-Aは「直近10 episodeの成功割合」とは記述していない|
|diagnostics/testImplementationFixes.py|道路下側境界を y=-2 から y=2 へ変更したことに伴う期待値を更新|1車線の物理道路範囲|
|diagnostics/testSingleLanePaper.py|道路構造、20 seedのspawn、車線変更可能性、削除車線の道路逸脱、3000-step集計、旧モデル不適合の検証を追加|統合の回帰防止|

成功率の集計時点は、終了した各episodeの終端stepとしました。主論文は集計系列を何stepごとに採取するかまでは指定していないため、この採取間隔は実装上の選択です。平滑化だけで10 seedの平均や標準偏差を再現できるわけではありません。

修正前の対象ファイルは、プロジェクト内の `diagnostics/singleLaneAudit/before/` に保存しました。今回の変更を確認するための差分は同梱の `implementation_changes.patch` にあります。

## 2. 現在の results/figures と results/videos の判定

現在の結果を論文通りと判定することはできません。

- 道路は2車線でした。動画の先頭・中間・最後のframeと、y座標図の周辺車両 y=0 / y=4 の両系列を確認しました。
- trainingLogは2130行ですが、最後の1行は予算終了で未完了です。完了した2129 episodeのうち成功は14回。2089回が cbf_infeasible、25回が衝突、1回が道路逸脱です。
- evaluationLogは50 episode、成功3回（6%）、cbf_infeasible 46回、衝突1回です。主論文Table IVのCBFG-PPOは5 m/s・10 m/sとも成功率100%です。seedや未記載条件があるため同一数値を保証できませんが、現在の性能差は明確です。
- bestPolicyは未学習ではありません。metadataによると更新1回、3072 step時点のpolicyです。学習全体は150000 stepですが、独立評価の順位で最初の更新時点のpolicyが選ばれています。過去の「未学習best」という別runの診断を今回に当てはめていません。
- training図は5 m/s、seed=0の単一runです。Fig.7の2速度条件・10 seed平均・標準偏差、およびPPO/PPO-R/DDPGとの比較を満たしていません。
- successRate.pngは従来の「直近10 episode」集計でした。今後の生成コードは修正しました。添付の historical_success_corrected.png は既存2車線runを新しい集計方法で再描画した参考図であり、1車線の再学習結果ではありません。
- simulationYPosition.pngではEVが本線中心4 mを通過後、さらに y≈0.6 mまで進んでいます。1車線化だけで既存policyが本線中心に留まる保証はありません。

既存のモデル、checkpoint、CSV、results/figures、results/videosは上書きしていません。道路変更後はcheckpointのsource hashが一致しないため、既存の互換性チェックにより旧policyの通常評価・再開が拒否されることを検証しました。旧policyをそのまま1車線環境で使って「論文再現」と表示することはできません。

## 3. 主論文と対応する点、未確定の点

|項目|確認結果|
|---|---|
|観測|Table Iに対応する27次元と周辺4車両の情報ブロックを確認。近い順で選ぶ処理、欠損値、正規化範囲は主論文未記載の仮定|
|行動|Eq.(1)の速度・headingの2次元参照に対応。物理速度0～20 m/s、heading ±π/4への変換範囲は未記載|
|報酬|Eq.(2)～(7)の5項を実装。front/rear不在時の代替距離、vmax、初回heading差の扱いは未記載|
|学習設定|現在はTable IIIの150000 step、40 epoch、3072 buffer、batch128、lr0.0006、discount0.99、CBF定数0.1と対応|
|MPC|速度・headingの追従と運動モデルを実装。horizon、Q/R、wheelbase、solver数値は主論文未記載|
|成功判定|合流後の継続安定性と指定target到達を確認する方針はIV-A/Bに対応。2秒、target x=450 m、車線中心±1 mは実装上の仮定|
|CBF教師行動|CBF補正対象は加速度・steering、RL出力は速度・heading。現在は安全補正後の1step先状態を参照行動に変換する。次元は整合するが、この変換方法は主論文には書かれていない。完全再現とは断定できない|
|道路CBF|現在は接続道路の外側境界を常時制約する。印刷されたEq.(23)～(25)のlocal式・車線変更flagによる有効化とは異なる解釈で、既存の安全性修正である。今回機械的には戻していない|
|車両CBFの符号|Eq.(20)からは h_next ≥ (1−gamma)h_current。印刷されたEq.(25)の車両h0は (1+gamma) であり、本文内で不整合。現在はEq.(20)を使用。道路h1/h2の印刷は (1−gamma)|
|最良policy選択|現在は固定held-out scenarioの成功率・returnでcheckpointを選ぶ。IV-Bは平均的training returnの学習strategyを使う記述があり、選択規則は一致していない|
|試験scenario|現在はtraining/testingが同じ速度・台数samplingを使用。IV-Bは試験時に速度変動を追加してtrainingと区別するが、追加変動の数値は未記載|
|学習step数|Table IIIの150000とIV-Aの600000/500000という収束step記述に不整合がある。本文の数値を推測で統一していない|

上表の未記載部分を一律に「実装ミス」とは扱っていません。主論文が指定していない部分については、引用文献で根拠が得られるか、得られない場合は再現上の仮定として明記する必要があります。

## 4. 送っていただきたい引用論文

1. [47] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, “Proximal Policy Optimization Algorithms,” 2017, arXiv:1707.06347。
   - PPO clip幅、Advantage推定、更新の詳細を確認します。主論文にはclip幅0.2、GAE lambda0.95の数値指定はありません。原典の代表値を使うことと、今回の主論文がその値を使ったことは区別します。
2. [46] M. Treiber, A. Hennecke, D. Helbing, “Congested Traffic States in Empirical Observations and Microscopic Simulations,” Physical Review E 62(2), 1805–1824, 2000。
   - IDMの相対速度の符号、車間距離が中心間距離か車体間のnet gapか、desired gapの下限処理を確認します。現在はHighwayEnvのcenter-to-center lane_distance_toを使用しています。

PDFを添付するか、パスをこのチャットに送ってください。CBF教師行動の参照変換は主論文が明示的に特定の原典へ委ねているとは確認できておらず、引用論文を読めば必ず解決できるとは述べていません。必要に応じて主論文がLfD/LfIで引用する[43]/[44]を次に確認します。

## 5. 論文自身が認める限界

IV-CはCBFの解が常に存在するとは限らないこと、動的環境での設計の難しさ、追加hyperparameterの調整負担を挙げています。この限界の完全な解消を今回の実装修正の成功条件にはしません。ただし、cbf_infeasibleが多いことだけで「論文通り」とも断定できません。モデル差、道路制約の解釈、教師行動、初期交通条件の影響を検証する必要があります。

## 6. 検証と次の作業

実行した検証：`python -m unittest diagnostics.testSingleLanePaper diagnostics.testImplementationFixes -v`。

26項目全件成功。20 seedの初期化、ランプから本線への変更可能性、削除した y=0 車線の道路逸脱扱い、CBF境界、MPC・GAE・checkpoint整合、短い実環境の学習更新、診断用feedbackでの合流を検証しました。短時間学習とfeedback診断は、学習policyの論文性能再現を証明するものではありません。

引用論文の受領後、PPO/IDMの照合と残る差分を処理し、1車線の新規学習を実施します。その後50 episode評価で図・動画を再生成し、最終的には5/10 m/sの各10 seedの集計と比較条件を揃えます。性能の数値を論文に似せるために結果データや曲線を書き換えることはしません。
