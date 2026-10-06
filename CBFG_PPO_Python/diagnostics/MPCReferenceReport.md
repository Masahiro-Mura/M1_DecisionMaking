今回のsteering下限への継続的な張り付きは、学習済みpolicyが負のheading referenceを出し続けることで説明できます。MPCの符号、index、単位、actionAdapterの符号反転、warm startによる固定化を示す証拠は得られませんでした。production codeおよびbestPolicy.ptは変更していません。

1. 原因候補と判定

|候補|確認結果|判定|
|---|---|---|
|Actor出力|Aのnormalized heading actionは初期−0.583928、最後−0.998725。Gaussian meanは−0.668402→−3.678498|今回の負referenceの直接の発生源|
|actionAdapter|v_ref=10(a0+1) m/s、theta_ref=(π/4)a1 rad。0 heading action→0 rad。±1→±π/4|現在の変換式に符号反転・index異常なし。ただし範囲自体は論文未記載の既存仮定|
|MPC heading tracking|v/headingはstate[2:4]。error=reference−current。二次形式なのでerror全体の符号を反転してもcostは同じ|取り違えの証拠なし|
|MPC optimizer/warm start|2条件のAでsolver失敗0。warm/cold steering最大差0.000172/0.000209 rad|継続的な飽和の説明にならない|
|古いpolicyの環境不整合|checkpointに環境版情報なし。保存時刻はtermination/observation/roadCBF修正前。旧学習ログには極端に短い成功episodeが存在|強い疑い。ただし学習時のソースを復元できず断定不可|

2. 確認したファイル・関数とデータフロー

`environment/observation.py:build_observation`が27次元のnormalized stateを生成します。先頭7要素はego vx, vy, x, y, lateral deviation, lcf, rcfです。

`rl/actor.py:forward`は未squashのGaussian meanを返します。`distribution`はmeanとexp(log_std)からNormalを構成し、`sample`はtanh(Gaussian sample)、`deterministic_action`はtanh(mean)を返します。meanそのものは[-1,1]に制約されません。tanh後のactionは数学的には(-1,1)、浮動小数点では端点に丸められる場合があります。deterministic actionはtanh(mean)であり、tanh後の確率分布の期待値ではありません。

`rl/ppoAgent.py:select_action(deterministic=True)` → `actor.deterministic_action` → normalized `[v_ref, theta_ref]`（heading index=1）。`environment/actionAdapter.py:action_to_reference`は[-1,1]へclipして単調増加のlinear mapを適用します。reference headingはlane相対値ではなくglobal headingとしてMPCへ渡されます。

`environment/mergeEnvironment.py:step` → `mpc/mpcController.py:solve` → `_objective` → `mpc/kinematicBicycleModel.py:rollout/discrete_step/continuous_dynamics` → `mpc/mpcCost.py:trajectory_cost` → SLSQP → 最初のnominal `[a,delta]` → `cbf/safeController.py:correct` → nominal+correction → `Vehicle.act` → `road.step(0.2)`。

MPCのstateは`[x,y,v,heading]`、outputは`[v,heading]`。angle wrappingはありませんが、今回のheading/referenceは±πの不連続点に近づいていません。Q=diag(1,1)は速度とheadingの追従誤差、R=diag(0.1,0.1)はaccelerationとsteeringの入力penaltyです。単位はそれぞれm/s、rad、m/s²、radです。

steering boundsは±π/4 rad、acceleration boundsは±5 m/s²。初回はzero sequence、以降は前回sequenceを1stepずらして末尾を繰り返します。episode reset時にwarm startはclearされます。solver失敗時もresult.xを使用する実装ですが、今回MPCでは失敗がありませんでした。

正steering→beta>0→heading増加。heading=0ではyも増加します。HighwayEnvでは画像上の下方向、lane ID増加側です。インストール済みVehicle.stepのheading更新分母LENGTH/2=2.5 mは、このプロジェクトのWHEELBASE=2.5 mと一致します。実車stepとMPC discrete_stepを±reference・速度5/10で数値比較し、一致を確認しました。

3. 実行したdiagnostic

実行コマンド：

```powershell
python diagnostics/testMPCReference.py
python diagnostics/testMPCReference.py --seed 10001 --speed 10 --output diagnostics/mpcReference_speed10_seed10001
```

各条件A〜Dについて1 episodeをtermination/truncationまで実行します。Aはdeterministic learned policy、B/C/Dはtheta_ref=0/+0.1/−0.1 rad、v_refは各stepの現在速度。manual actionは既存adapterの逆変換で生成し、物理referenceの一致をassertしています。Actorはmanual caseでも観測に対するmeanとpolicy actionを記録しますが、環境へ渡すactionとは別列です。

診断はenv instanceのsolve/correct呼び出しをwrapperで観測するだけで、計算結果や適用入力を差し替えません。cold startとzero referenceの反実仮想計算には別のMPC instanceを使用します。実際のnominalはinfo['mpc_control']との一致を確認しています。

|初速・seed|Test|steps|MPC steering飽和率|MPC失敗|CBF失敗|終端|
|---|---|---:|---:|---:|---:|---|
|5 m/s・10000|A learned|111|93.69%|0|1|road_departure|
|5 m/s・10000|B 0 rad|86|0%|0|15|road_departure|
|5 m/s・10000|C +0.1 rad|81|0%|0|8|road_departure|
|5 m/s・10000|D −0.1 rad|81|0%|0|17|success|
|10 m/s・10001|A learned|100|94%|0|1|time_limit|
|10 m/s・10001|B 0 rad|43|0%|0|0|road_departure|
|10 m/s・10001|C +0.1 rad|43|0%|0|0|road_departure|
|10 m/s・10001|D −0.1 rad|41|0%|0|0|success|

飽和はabs(steering)>=π/4−1e−4で集計します。CBF失敗はsafe_result.success=Falseで集計しています。Dのsuccessは現行termination判定によるもので、合流全体の安全性や論文再現の成功を意味しません。B/Cはrampを最後まで走る、あるいは外側へ向かうreferenceなのでroad_departureだけをMPC異常の根拠にはできません。Bで交通のCBF介入がheadingを変えれば、それを0へ戻すMPC steeringは0以外になります。heading=0かつreference=0でのisolated steeringは厳密に0でした。

4. 飽和が生じる機序

初速5・seed10000のAでは、最後のpre-step headingは−0.00000286 radなのに、policyがtheta_ref=−0.78439645 radを要求します。MPC steering=−0.78539816、CBF steering correction=+0.78539879、final steering=+0.000000626 rad。MPCは将来のnegative headingを追従しようとし、CBFは境界でそれを抑えるため、同じ構図が繰り返されます。

このstepのoptimized costは0.861229（heading term 0.761373、steering term 0.099144など）、zero-control sequenceのcostは6.771628です。Aの全stepでoptimized costはzero-control costより低く、単純にcostを増やす方向へsolverが走っているわけではありません。Aのcold solveは全step成功し、ほぼ同じcontrolになります。

5. Policy compatibilityと未確定事項

bestPolicy.ptのcheckpoint keysはactor/critic/optimizerだけです。環境版、action normalization、termination、CBF設定、学習seedの証明情報はありません。ファイルのLastWriteTimeはbestPolicy.pt=2026-10-03 09:54:53、termination.py=10:33:40、observation.py=11:29:28、roadCBF.py=11:57:07です。時刻は補助証拠であり、学習時環境の完全な証明ではありません。

trainingLog.csvの最後100 episodeはlength=3〜48 step、success=89/100、平均CBF intervention rate≈0.07887です。230 mから310 mに到達する現行success条件と非常に短い成功episodeの組合せは、以前のtermination条件下での学習という説明を支持します。ただし、それだけでnegative Actor biasの学習原因を断定しません。

初速5のAではt=1.4 sにCBFが制約を満たせず、minimum_constraint_value≈−0.238223のcontrolが適用されました。これはMPC飽和とは別の安全制御上の未解決点です。交通・道路contextの切替・constraint feasibility・solver失敗時の処理を切り分ける必要があります。nominalが既存CBF制約を満たしたstepでは、今回の全8 episodeでcorrectionは0でした。「既存CBF制約を満たすこと」と「実際の道路全体で安全であること」は同一とは限りません。

6. 論文照合と引用文献の扱い

ユーザー指定のメインPDF（IEEE pp.8919–8932）を直接読み、Eq.(1)、(14)〜(18)、(19)〜(25)、(27)〜(31)、Algorithm 1を照合しました。今回のaction/control ordering、global state、beta、Euler update、tracking objectiveはメイン論文だけで確認できるため、添付された他論文から追加の値・処理は導入していません。headingの物理範囲、Q/R、prediction horizon、tanh Gaussian、low-level correctionからActor demonstrationへの変換を、論文で規定された値・処理として扱っていません。

再現上の追加注意点として、Eq.(17)の印刷上のsum上限はn_p−1ですが、現行costはN+1 stateすべてにtracking termを課しています。また、PDF Eq.(25)は(1+gamma_hat)、Eq.(20)からの代数変形は(1−gamma_hat)です。現行safeControllerは後者を使っていますが、docstringは「Eq.(25) literally」と説明しており、出典説明が実際のPDFと一致していません。これらは今回の負reference発生源ではないため変更していません。論文に忠実な実装を完成させる段階では、別途明示的な判断が必要です。

HighwayEnv公式仕様・一次ソース：

- [Kinematics](https://highway-env.farama.org/v1.8.1/dynamics/vehicle/kinematics/)
- [Vehicle.step source](https://github.com/Farama-Foundation/HighwayEnv/blob/main/highway_env/vehicle/kinematics.py)

7. 回帰項目と変更一覧

|要求項目|結果|
|---|---|
|theta_ref=0、ego heading=0→steering≈0|PASS、速度5/10とも0|
|reference>ego heading→正steering|PASS、ego heading=0/±0.2で確認|
|reference<ego heading→負steering|PASS、同上|
|過去の飽和sequenceからzero referenceへ復帰|PASS、abs(steering)<0.000135 rad|
|MPCとHighwayEnv step符号・更新一致|PASS|
|nominal安全→CBFほぼ0、unsafe→介入|isolated fixtureでPASS。rollout中のnominal-feasible stepでもcorrection=0|
|learned policyで恒常的limit張り付きを解消|未達。旧policyのまま93.69%/94%|
|learned policyの恒常的道路境界走行・逸脱を解消|未達。新policyの学習・検証は実施していない|

|追加ファイル|関数|対応した問題|内容|根拠・仮定|テスト結果|
|---|---|---|---|---|---|
|diagnostics/testMPCReference.py|run_case/cost_components|Actor→reference→nominal→CBFを一緒に観測できなかった|A〜D CSV、cost内訳、solver結果、warm/cold比較、制約残差を記録|主論文のstate/action/control定義。manual testは診断fixture|全8 episode完走、cost合計・adapter逆変換・適用control一致assertがPASS|
|diagnostics/testMPCReference.py|isolated_checks|policy/trafficを切り離した符号・安全介入の検証がなかった|zero/±heading、nonzero ego heading、飽和warm start、Vehicle.step比較、CBF安全/unsafe fixture|人工境界fixtureはコードにIMPLEMENTATION ASSUMPTIONを明記|21件の記録を保存、assertはすべてPASS|
|diagnostics/MPCReferenceReport.md|該当なし|原因・未解決・論文差異の混同|本報告書|観測事実と推測を区別|production source/checkpoint SHA-256不変を確認|

CSVはmpcReference/A.csv〜D.csvおよびmpcReference_speed10_seed10001/A.csv〜D.csv。summary.jsonにはproduction sourceとcheckpointのSHA-256を保存しています。追加のisolated checksはmpcReference/regressionChecks.jsonに保存しています。

今回の症状に対するproduction修正候補はありません。現在の環境でpolicyを新規学習し、同じA diagnosticで比較することが次の検証になります。ただし再学習で必ず解消するとは未確認です。既存bestPolicyの上書きや、reference clipping・steering bound・reward・Q/R調整による症状隠しは行っていません。CBF solver失敗と論文との差異は別の調査事項として残ります。
