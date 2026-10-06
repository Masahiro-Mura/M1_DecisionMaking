今回のsimulation.mp4と、2026-10-03 15:15〜15:17に保存されたpolicy/logを対象とした再確認です。前回MPCReferenceReport.mdの対象checkpointとは異なります。production code、学習設定、policyは変更していません。

1. 今回の動画の直接原因

現在のbestPolicy.ptはseed=0のPPOAgent生成直後のActor・Criticと全tensorが完全一致しました。optimizer.stateは空です。すなわち、このbestPolicyは未学習の初期policyです。一方finalPolicy.ptはActor tensor 7個がbestと異なり、optimizer.step=80の更新履歴があります。

main/mainLearning.pyでは最初のbuffer-full更新までpolicyを更新しませんが、episodeごとのbest保存はその前にも実行します。今回のtrainingLogはepisode0（47step）とepisode1（91step）でrolling_success_rate=1.0です。確率的な初期policyとCBFが達成した成功率を使って未更新policyを保存し、後のpolicyがこのスコアを超えないと置き換わりません。学習停止後のpartial buffer更新はfinalPolicyのみに反映されます。

最新のtrainingLogは42episode・2000step。updateLog末尾の今回のrunはglobal_step=1024と2000の2回です。updateLogはappend方式なので、先頭には過去runの数値も混在しています。

simulationLogは44step、theta_ref=+0.041624〜+0.063940rad、y=8.047618〜10.063701m。HighwayEnvでは本線はrampのy=8より小さいy側です。今回の小さいpositive referenceは本線の反対側へ向かう指令です。evaluationLogの50episodeはsuccess=0、本線進入=0。動画中盤と最終frameもこの動きと一致します。

初期状態・seed10000でnormalized actionを比較するとbest=[0.01953683,0.04730172]、final=[−0.0365864,−0.17516784]でした。

2. 2000stepの影響

メイン論文Table III（PDF9ページ、IEEE p.8927）との比較：

|項目|論文|現在|
|---|---:|---:|
|Total timesteps|150000|2000|
|Number of epoch when optimizing|40|5|
|Replay buffer size|128×24=3072|1024|
|Training batch size|128|128|
|Learning rate|0.0006|0.0006|
|Discount rate|0.99|0.99|
|CBF constant|0.1|0.1|

2000stepはTable IIIの1/75であり、現在のbuffer設定ではデータ収集・policy更新サイクルが2回しかありません。学習不足は十分あり得ます。ただしstep数を増やせば必ず合流できるとは確認していません。本文IV-Aには500000/600000stepで収束との記載もあり、Table IIIの150000と整合しない箇所があります。ここではTable IIIの値を明示し、本文との不一致も隠しません。

finalPolicyの再評価コマンド：

```powershell
python diagnostics/testMPCReference.py --model results/models/finalPolicy.pt --output diagnostics/paperAudit/finalPolicyReference
```

A（速度5m/s・seed10000）の115stepではy最小1.63644mとなり、本線側へ移動しました。MPC飽和率0%、MPC/CBF solver失敗0、CBF介入0。ただしx>=310付近ではy≈1.7〜2.1mにあり、車線中央からの距離がsuccess許容値1m以内に落ち着かず、最後はx=465.647mの道路終端を越えてroad_departureとなりました。最初から合流方向に動かないbestと、動くが完了できないfinalを区別してください。この1seedだけでfinalの一般的な成功率は判定できません。

3. 確定した実装上の問題

|ファイル・関数|問題|影響・根拠|
|---|---|---|
|main/mainLearning.py:train|未更新policyをbestとして保存可能。保存順位は確率的training rolloutの直近10episodeスコアで、deterministicな独立評価ではない。最後のpartial updateもbestに反映されない|今回のbestは初期重みと完全一致。動画の直接原因。論文の数式違反というより、学習成果の保存・評価を妨げる実装問題|
|rl/advantage.py:compute_gae、rl/rolloutBuffer.py、main/mainLearning.py:train|GAEの再帰をterminatedだけで切り、truncatedのepisode境界を保存していない。bufferはepisodeをまたぐ|time-limit終了ではV(next_state)のbootstrapを許すことと、次のreset episodeのGAEを足すことは別。後者は遮断が必要。人工例rewards=[1,100], values=next_values=0, terminated=[0,1]で最初のepisodeがtruncatedだと現行advantage=[95.05,100]。最初は1であるべき。今回の動画の直接原因とまでは断定しない|
|parameters/rlParameters.py|step/epoch/bufferを論文値から変更|学習量・更新条件が異なる。変更が意図的でも、論文と同じ実験条件とは扱えない|

4. 論文と異なる部分、および論文未記載の仮定

|ファイル・関数|現行処理|論文との関係|
|---|---|---|
|rl/cbfGuidedLoss.py:build_demonstration_action|normalized reference actionに、物理acceleration/steering補正をcontrol範囲でnormalizeして加算・clip|最も重要な再現上の仮定。reference speedとacceleration、reference headingとsteeringは異なる量で、この加算は安全なreferenceを復元する保証がない。Eq.(27)〜(31)はCBF補正行動の模倣を述べるが、low-level correctionからreference actionへの逆変換は未記載|
|cbf/roadCBF.py:left_road_barrier/right_road_barrier|h1=W/2−d、h2=W/2+d|Eq.(23)/(24)の文字通りの実装ではなく、HighwayEnvのlocal lane座標に合わせた解釈。literal式ではlane centerでh1<0となる曖昧さを補っている。安全集合の意図に沿う修正理由はあるが、論文そのままとは言えない|
|cbf/safeController.py:_build_constraints|h_next−(1−gamma)h_current>=0|PDF Eq.(25)は(1+gamma)、Eq.(20)の代数変形は(1−gamma)。現行はEq.(20)と整合するが、「Eq.(25) literally」というコード説明は誤り。plusへ機械的に戻すことを推奨する意味ではない|
|mpc/mpcCost.py:trajectory_cost|N controlsにN+1 statesすべてのtracking costを課す|Eq.(17)のsum上限はn_p−1。末端cost追加は現行の解釈であり、論文がその末端termを指定したわけではない|
|environment/termination.py:check_success、parameters/environmentParameters.py|x>=310、main lane、中心距離<=1mでsuccessとなり即termination|具体的な位置・閾値は既存仮定。論文III-Dはtime-limit/collision/departureを停止条件として説明し、IV-Aでは合流後も一定時間failureがないことを成功判定に含める。現行は合流後の継続安定性を確認しない|
|environment/actionAdapter.py、parameters/environmentParameters.py|v_ref∈[0,20]、theta_ref∈[−π/4,+π/4]|Eq.(1)はnormalized[-1,1]で、物理範囲は未記載。範囲の選択は仮定。今回の変換に符号反転・index異常は確認されていない|
|parameters/mpcParameters.py、parameters/vehicleParameters.py|horizon10、Q=diag(1,1)、R=diag(0.1,0.1)、WHEELBASE2.5、SLSQP|数値・solverは論文未記載の仮定。診断で符号・step整合は確認済みだが、論文で指定された値とは扱わない|
|rl/actor.py、parameters/rlParameters.py|tanh Gaussian、state-independent log_std、初期log_std−0.5など|bounded samplingやstdの詳細は論文未記載。Fig.6の2×Dense256とlinear mean出力は対応しているが、確率分布の詳細まで再現できたとは言えない|
|rl/critic.py|V(s)だけを入力|Eq.(12)のV(s)には対応。ただしFig.6にはvalue networkへのRL action入力も描かれており、本文・図の曖昧さが残る。直ちにQ(s,a)へ変更する根拠にはしない|

道路初期化・車両数範囲・観測normalization・周辺車両の選択順・欠損front/rear処理にも既存仮定があります。論文が指定した値と混同しないことが必要です。今回は主論文で確認できる箇所を主に調べ、GAEは主論文に推定法の指定がなく[47]を引用しているため、添付PPO元論文のAlgorithm/advantage説明も参照しました。別論文の値や処理をproductionへ追加していません。

優先する順序は、未更新bestの選択を防いでfinal/独立deterministic評価と比較すること、GAEのepisode境界を直すこと、CBF教師actionの意味を明示すること、その上でTable IIIに合わせた学習と複数seed評価です。論文値への復帰は再現条件をそろえるためであり、既知の実装問題をハイパーパラメータで隠すためではありません。
